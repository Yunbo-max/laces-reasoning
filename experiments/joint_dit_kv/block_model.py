"""Block-causal continuous latent diffusion for single-prompt multi-stream reasoning.

Within each block all latent slots see one another bidirectionally.
Across blocks attention is strictly causal: block b cannot be read by earlier
blocks, and a predicted block sees only the prompt and previously completed blocks.

Adapted *blockwise causal structure* from Arriola et al. (BD3-LMs, ICLR 2025);
unlike BD3-LMs, this implementation uses rectified flow over continuous
autoencoder latents, not discrete token-space diffusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .model import JointConfig, JointSystem


@dataclass
class BlockConfig(JointConfig):
    streams_per_block: int = 2
    max_blocks: int = 4
    stop_threshold: float = 0.50

    @property
    def block_positions(self) -> int:
        return self.streams_per_block * self.slots


def block_causal_mask(group_ids: torch.Tensor) -> torch.Tensor:
    """True means BLOCKED; previous groups may never read later groups."""
    if group_ids.ndim != 1:
        raise ValueError("group_ids must be a vector")
    return group_ids[None, :] > group_ids[:, None]


class BlockCausalDenoiser(nn.Module):
    """Single active noisy block, fully visible within itself; clean historical blocks."""
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.latent_dim
        self.type_embed = nn.Embedding(3, d)  # prompt, past-clean, current-noisy
        self.block_embed = nn.Embedding(cfg.max_blocks + 1, d)
        self.inblock_embed = nn.Embedding(cfg.block_positions, d)
        self.time_embed = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.heads, dim_feedforward=4*d,
            batch_first=True, dropout=0.0, activation="gelu", norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.dit_layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def token_groups(self, prompt_slots: int, prior_blocks: int, device):
        s = self.cfg.block_positions
        return torch.tensor(
            ([0] * prompt_slots) +
            [i + 1 for i in range(prior_blocks) for _ in range(s)] +
            ([prior_blocks + 1] * s),
            device=device, dtype=torch.long,
        )

    def forward(self, noisy: torch.Tensor, t: torch.Tensor,
                prompt: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        # noisy: [B, S, d]; prompt [B, P, d]; previous [B, b, S, d]
        b, s, d = noisy.shape
        if s != self.cfg.block_positions or d != self.cfg.latent_dim:
            raise ValueError("Noisy block shape must be [batch, block_positions, latent_dim]")
        if prompt.ndim != 3 or prompt.size(0) != b or prompt.size(2) != d:
            raise ValueError("Prompt shape mismatch")
        if previous.ndim != 4 or previous.shape[0] != b or previous.shape[2:] != (s, d):
            raise ValueError("Previous blocks must have shape [batch, prior_blocks, S, d]")
        h = previous.size(1)
        if h >= self.cfg.max_blocks:
            raise ValueError("Exceeds max_blocks")
        ids = self.token_groups(prompt.size(1), h, noisy.device)
        attention_mask = block_causal_mask(ids)
        x_prompt = prompt + self.type_embed.weight[0][None, None]
        components = [x_prompt]
        for j in range(h):
            positions = self.inblock_embed.weight[None]
            components.append(previous[:, j] + positions
                              + self.type_embed.weight[1][None, None]
                              + self.block_embed.weight[j + 1][None, None])
        active = (
            noisy + self.inblock_embed.weight[None]
            + self.type_embed.weight[2][None, None]
            + self.block_embed.weight[h + 1][None, None]
            + self.time_embed(t.reshape(b, 1))[:, None]
        )
        components.append(active)
        embeddings = torch.cat(components, dim=1)
        out = self.blocks(embeddings, mask=attention_mask)
        return self.out(out[:, -s:])


class BlockFlow(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.cfg = cfg
        self.denoiser = BlockCausalDenoiser(cfg)
        d = cfg.latent_dim
        self.stop = nn.Sequential(
            nn.LayerNorm(d * 2), nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1)
        )

    def loss_one_block(self, clean: torch.Tensor, prompt: torch.Tensor,
                       previous: torch.Tensor):
        clean = clean.detach()
        eps = torch.randn_like(clean)
        t = torch.rand(clean.size(0), device=clean.device).clamp(0.02, 0.98)
        x = (1 - t[:, None, None]) * clean + t[:, None, None] * eps
        velocity = self.denoiser(x, t, prompt, previous)
        return F.mse_loss(velocity, eps - clean)

    def sample_one_block(self, prompt: torch.Tensor, previous: torch.Tensor,
                         steps: int | None = None) -> torch.Tensor:
        steps = self.cfg.sample_steps if steps is None else steps
        if steps < 1:
            raise ValueError("sampling steps must be >=1")
        b = prompt.size(0)
        x = torch.randn(b, self.cfg.block_positions, self.cfg.latent_dim,
                        device=prompt.device, dtype=prompt.dtype)
        for n in range(steps):
            t = torch.full((b,), 1.0 - (n + 0.5) / steps,
                           device=prompt.device, dtype=prompt.dtype)
            x = x - self.denoiser(x, t, prompt, previous) / steps
        return x

    def stop_logit(self, block: torch.Tensor, prompt: torch.Tensor):
        return self.stop(torch.cat([block.mean(dim=1), prompt.mean(dim=1)], dim=-1)).squeeze(-1)


class BlockJointSystem(JointSystem):
    """Reuse the exact AE/KV writer/receiver interface from the fixed-length prototype."""
    def __init__(self, receiver, tokenizer, cfg: BlockConfig):
        super().__init__(receiver, tokenizer, cfg)
        self.flow = BlockFlow(cfg)
        self.cfg = cfg

    def prompt_latents(self, question: str, device: torch.device):
        return self.encode(question, device)

    def target_blocks(self, steps: Sequence[str], device: torch.device) -> torch.Tensor:
        """Gold steps are training targets only, NEVER inference conditions."""
        if not steps:
            raise ValueError("Training requires at least one reasoning step")
        if len(steps) > self.cfg.streams_per_block * self.cfg.max_blocks:
            # Keep final steps and terminal answer, but limit number of blocks.
            steps = list(steps[-self.cfg.streams_per_block * self.cfg.max_blocks:])
        steps = list(steps)
        total = ((len(steps) + self.cfg.streams_per_block - 1)
                 // self.cfg.streams_per_block) * self.cfg.streams_per_block
        steps += ["[NO_STEP]"] * (total - len(steps))
        # Per step: P x d; per block: (streams_per_block * P) x d.
        latents = [self.encode(str(s), device)[0] for s in steps]
        tensor = torch.stack(latents).reshape(
            total // self.cfg.streams_per_block, self.cfg.block_positions,
            self.cfg.latent_dim)
        return tensor.unsqueeze(0).detach()

    def sample_blocks(self, prompt: torch.Tensor, *,
                      force_blocks: int | None = None,
                      max_blocks: int | None = None):
        """Autoregressive across blocks; each denoising call is bidirectional within a block."""
        cap = min(self.cfg.max_blocks, max_blocks or self.cfg.max_blocks)
        if force_blocks is not None and not 1 <= force_blocks <= cap:
            raise ValueError("force_blocks outside 1..max_blocks")
        b = prompt.size(0)
        history = prompt.new_zeros((b, 0, self.cfg.block_positions, self.cfg.latent_dim))
        stop_logits = []
        for i in range(cap):
            active = self.flow.sample_one_block(prompt, history)
            history = torch.cat([history, active[:, None]], dim=1)
            logit = self.flow.stop_logit(active, prompt)
            stop_logits.append(logit)
            if force_blocks is not None:
                if i + 1 == force_blocks:
                    break
            elif bool((logit.sigmoid() >= self.cfg.stop_threshold).all()):
                break
        return history, torch.stack(stop_logits, dim=1)

    def block_losses(self, question: str, steps: Sequence[str], answer: str,
                     device: torch.device, *,
                     ce_weight: float = 1.0,
                     stop_weight: float = 0.2):
        prompt = self.prompt_latents(question, device)
        gold = self.target_blocks(steps, device)
        n = gold.size(1)
        # Teacher-force prior blocks ONLY (never future blocks); active block noisy.
        idx = int(torch.randint(n, (1,), device=device).item())
        prior = gold[:, :idx]
        flow_mse = self.flow.loss_one_block(gold[:, idx], prompt, prior)
        # Receiver CE is trained on actual autoregressively SAMPLED blocks.
        generated, stop_logits = self.sample_blocks(prompt, force_blocks=n)
        loss_ce = self.receiver_loss(generated, question, answer, device)
        # STOP supervision on generated states to avoid clean-vs-sampled mismatch.
        labels = torch.zeros_like(stop_logits)
        labels[:, -1] = 1.0
        stop_bce = F.binary_cross_entropy_with_logits(stop_logits, labels)
        total = flow_mse + ce_weight * loss_ce + stop_weight * stop_bce
        return total, {
            "flow_mse": flow_mse.detach(),
            "receiver_ce": loss_ce.detach(),
            "stop_bce": stop_bce.detach(),
            "target_blocks": n,
        }

    @torch.no_grad()
    def answer(self, question: str, device: torch.device,
               max_new_tokens: int = 48, *, force_blocks: int | None = None,
               max_blocks: int | None = None):
        prompt = self.prompt_latents(question, device)
        latents, logits = self.sample_blocks(prompt, force_blocks=force_blocks,
                                             max_blocks=max_blocks)
        text = self.generate(latents, question, device,
                             max_new_tokens=max_new_tokens)
        return text, {
            "num_blocks": int(latents.size(1)),
            "stop_probabilities": logits.sigmoid().flatten().tolist(),
            "virtual_kv_positions": int(latents.size(1) * self.cfg.block_positions),
        }

"""Joint multi-agent latent flow + trainable virtual-KV prefix for a frozen causal LM.

Flow matching (not DDPM) is used as a computationally small continuous diffusion prior.
Gold-answer latents are NEVER used as inference-time conditioning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class JointConfig:
    latent_dim: int = 128
    slots: int = 4
    max_agents: int = 8
    max_text_tokens: int = 96
    heads: int = 4
    dit_layers: int = 4
    sample_steps: int = 4


class QueryEncoder(nn.Module):
    """Learned queries recover order-sensitive evidence from frozen token embeddings."""
    def __init__(self, embedding_dim: int, cfg: JointConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.latent_dim
        self.proj = nn.Linear(embedding_dim, d)
        self.positions = nn.Embedding(cfg.max_text_tokens, d)
        self.queries = nn.Parameter(torch.randn(cfg.slots, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, input_ids: torch.Tensor, frozen_embeddings: nn.Module) -> torch.Tensor:
        if input_ids.size(1) > self.cfg.max_text_tokens:
            input_ids = input_ids[:, :self.cfg.max_text_tokens]
        with torch.no_grad():
            emb = frozen_embeddings(input_ids)
        h = self.proj(emb.float())
        pos = torch.arange(h.size(1), device=h.device)
        h = h + self.positions(pos)[None]
        q = self.queries[None].expand(h.size(0), -1, -1)
        attended, _ = self.cross(q, h, h, need_weights=False)
        z = q + attended
        return self.norm(z + self.ff(z))


class JointDiT(nn.Module):
    """All agent-specific + global answer slots exchange information bidirectionally."""
    def __init__(self, cfg: JointConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.latent_dim
        self.agent_pos = nn.Embedding(cfg.max_agents + 1, d)
        self.slot_pos = nn.Embedding(cfg.slots, d)
        self.context_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.noise_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        block = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.heads, dim_feedforward=d*4,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=cfg.dit_layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # noisy and cond: [B, M+1, P, D], last agent slot is global answer.
        if noisy.shape != cond.shape or noisy.ndim != 4:
            raise ValueError("noisy/cond must share shape [B, agents+1, slots, dim]")
        b, groups, p, d = noisy.shape
        if groups > self.cfg.max_agents + 1 or p != self.cfg.slots or d != self.cfg.latent_dim:
            raise ValueError("latent group/slot/dimension mismatch")
        position = (
            self.agent_pos(torch.arange(groups, device=noisy.device))[:, None, :]
            + self.slot_pos(torch.arange(p, device=noisy.device))[None, :, :]
        ).reshape(1, groups * p, d)
        tim = self.time_mlp(t.reshape(b, 1)).unsqueeze(1)
        ctx = cond.reshape(b, groups*p, d) + position + self.context_type
        noisy_slots = noisy.reshape(b, groups*p, d) + position + self.noise_type + tim
        out = self.blocks(torch.cat([ctx, noisy_slots], dim=1))
        return self.out(out[:, groups*p:]).reshape(b, groups, p, d)


class JointFlow(nn.Module):
    """Rectified flow: x(t)=(1-t)*clean + t*noise, dx/dt=noise-clean."""
    def __init__(self, cfg: JointConfig):
        super().__init__()
        self.cfg = cfg
        self.net = JointDiT(cfg)

    def training_loss(self, clean: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        clean = clean.detach()  # moving AE targets must not collapse toward the prior
        noise = torch.randn_like(clean)
        t = torch.rand(clean.size(0), device=clean.device).clamp(0.02, 0.98)
        t4 = t[:, None, None, None]
        x = (1 - t4) * clean + t4 * noise
        return F.mse_loss(self.net(x, t, cond), noise - clean)

    def sample(self, cond: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        steps = self.cfg.sample_steps if steps is None else steps
        if steps < 1:
            raise ValueError("Flow sampling steps must be positive")
        x = torch.randn_like(cond)  # NO oracle answer information at inference
        for i in range(steps):
            t = torch.full((cond.size(0),), 1.0 - (i + 0.5) / steps,
                           device=cond.device, dtype=cond.dtype)
            x = x - self.net(x, t, cond) / steps
        return x


class KVWriter(nn.Module):
    """Small shared writer. Each latent slot synthesizes one post-RoPE KV prefix position."""
    def __init__(self, cfg: JointConfig, decoder_config):
        super().__init__()
        self.cfg = cfg
        self.decoder_config = decoder_config
        self.layers = int(decoder_config.num_hidden_layers)
        self.kv_heads = int(decoder_config.num_key_value_heads)
        self.head_dim = int(getattr(decoder_config, "head_dim", None)
                            or decoder_config.hidden_size // decoder_config.num_attention_heads)
        width = self.layers * self.kv_heads * self.head_dim * 2
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim),
            nn.Linear(cfg.latent_dim, cfg.latent_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.latent_dim * 2, width),
        )
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)
        self.log_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        # [B, P, d] -> [layers, K/V, B, n_kv, P, head_dim].
        b, p, _ = slots.shape
        output = self.net(slots.float()) * self.log_scale.exp()
        output = output.reshape(b, p, self.layers, 2, self.kv_heads, self.head_dim)
        return output.permute(2, 3, 0, 4, 1, 5).contiguous()

    def make_cache(self, slots: torch.Tensor, dtype: torch.dtype):
        from transformers.cache_utils import DynamicCache
        try:
            cache = DynamicCache()
        except TypeError:
            cache = DynamicCache(config=self.decoder_config)
        tensors = self(slots).to(dtype=dtype)
        for layer in range(self.layers):
            cache.update(tensors[layer, 0], tensors[layer, 1], layer)
        return cache


class JointSystem(nn.Module):
    """Frozen Qwen receiver + AE/flow/writer; answer logits supervised through virtual KV."""
    def __init__(self, receiver: nn.Module, tokenizer, cfg: JointConfig):
        super().__init__()
        self.receiver = receiver.eval()
        self.receiver.requires_grad_(False)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.encoder = QueryEncoder(receiver.get_input_embeddings().embedding_dim, cfg)
        self.flow = JointFlow(cfg)
        self.writer = KVWriter(cfg, receiver.config)

    def encode(self, text: str, device: torch.device) -> torch.Tensor:
        ids = self.tokenizer(text, add_special_tokens=False,
                             truncation=True, max_length=self.cfg.max_text_tokens,
                             return_tensors="pt")["input_ids"].to(device)
        if ids.numel() == 0:
            raise ValueError("Cannot encode empty text")
        return self.encoder(ids, self.receiver.get_input_embeddings())

    def encode_conditions(self, agents: List[str], question: str, device: torch.device):
        if not (1 <= len(agents) <= self.cfg.max_agents):
            raise ValueError("Agent count outside configured bounds")
        local = torch.stack([self.encode(s, device)[0] for s in agents], dim=0)[None]
        q = self.encode(question, device)[:, None]
        return torch.cat([local, q], dim=1)

    def gold_targets(self, conditions: torch.Tensor, answer: str, device: torch.device):
        answer_slots = self.encode(answer, device)[:, None]
        return torch.cat([conditions[:, :-1], answer_slots], dim=1).detach()

    def latent(self, conditions: torch.Tensor, strategy: str):
        if strategy == "joint":
            return self.flow.sample(conditions)
        if strategy == "independent":
            # No cross-agent denoising: independent local encodings followed by concatenation.
            return conditions
        if strategy == "mean":
            return conditions.mean(dim=1, keepdim=True).expand_as(conditions)
        raise ValueError("Unknown latent strategy: " + strategy)

    def _receiver_forward(self, slots: torch.Tensor, input_ids: torch.Tensor,
                          use_cache: bool):
        prefix = slots.shape[1]
        cache = self.writer.make_cache(slots, next(self.receiver.parameters()).dtype)
        n = input_ids.shape[1]
        positions = torch.arange(prefix, prefix + n, device=input_ids.device)
        return self.receiver(
            input_ids=input_ids,
            past_key_values=cache,
            attention_mask=torch.ones((1, prefix + n), device=input_ids.device, dtype=torch.long),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            use_cache=use_cache, return_dict=True,
        )

    def token_ce(self, slots: torch.Tensor, prompt: str, target: str,
                 device: torch.device, max_target_tokens: int = 64):
        prefix_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
            "input_ids"].to(device)
        gold = self.tokenizer(target, add_special_tokens=False,
                              truncation=True, max_length=max_target_tokens,
                              return_tensors="pt")["input_ids"].to(device)
        if prefix_ids.shape[1] < 1 or gold.shape[1] < 1:
            raise ValueError("Empty supervised prompt or target")
        # Last prompt-token logit predicts FIRST target token (no boundary-token omission).
        sequence = torch.cat([prefix_ids, gold[:, :-1]], dim=1)
        output = self._receiver_forward(slots, sequence, use_cache=False)
        logits = output.logits[:, prefix_ids.shape[1] - 1:]
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), gold.reshape(-1))

    def receiver_loss(self, latents: torch.Tensor, question: str, answer: str,
                      device: torch.device):
        slots = latents.flatten(1, 2)
        return self.token_ce(slots, "Question: " + question + "\nAnswer:",
                             " " + answer, device)

    def ae_loss(self, evidence: str, device: torch.device):
        z = self.encode(evidence, device)
        return self.token_ce(z, "Reproduce the exact evidence:\n", " " + evidence,
                             device, max_target_tokens=self.cfg.max_text_tokens)

    @torch.no_grad()
    def generate(self, latents: torch.Tensor, question: str, device: torch.device,
                 max_new_tokens: int = 48, temperature: float = 0.0,
                 top_p: float = 0.9) -> str:
        self.receiver.eval()
        slots = latents.flatten(1, 2)
        ids = self.tokenizer("Question: " + question + "\nAnswer:",
                             add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        prefix = slots.shape[1]
        out = self._receiver_forward(slots, ids, use_cache=True)
        cache = out.past_key_values
        next_logits = out.logits[:, -1]
        generated = []
        at = prefix + ids.shape[1]
        for _ in range(max_new_tokens):
            if temperature <= 0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits.float() / temperature, dim=-1)
                sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
                remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
                sorted_probs = sorted_probs.masked_fill(remove, 0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                sample_idx = torch.multinomial(sorted_probs, 1)
                next_id = sorted_indices.gather(-1, sample_idx)
            token_id = int(next_id.item())
            if token_id == self.tokenizer.eos_token_id:
                break
            generated.append(token_id)
            pos = torch.tensor([at], dtype=torch.long, device=device)
            out = self.receiver(
                input_ids=next_id, past_key_values=cache,
                attention_mask=torch.ones((1, at+1), dtype=torch.long, device=device),
                position_ids=pos.unsqueeze(0), cache_position=pos,
                use_cache=True, return_dict=True,
            )
            cache, next_logits = out.past_key_values, out.logits[:, -1]
            at += 1
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def compact_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("receiver.")}

    def train(self, mode: bool = True):
        super().train(mode)
        self.receiver.eval()  # no LM dropout, but grads still flow to virtual KV
        return self

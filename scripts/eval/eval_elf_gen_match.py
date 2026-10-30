#!/usr/bin/env python3
"""ELF (Embedded Language Flows) generate-then-match harness for the ELF baselines.

ELF is NOT a causal LM and NOT a discrete diffusion LM. It is a continuous
flow-matching model over **frozen T5-small sentence embeddings**:

    text --T5 encoder--> (L, 512) latents --/0.2--> bottleneck(128) --> DiT denoiser
                                                                       (flow matching)
    denoised latent --model's own CE head--> token ids --T5 tokenizer--> text

Consequences that make the sibling harnesses unusable for this model (and that are
the reason this file exists):

  * `AutoModelForCausalLM.generate()` cannot run it at all -- there is no HF config,
    only a `checkpoint_<step>` torch.save payload plus `config.yml`.
  * MDLM/SEDD's discrete denoising loops do not apply: the state being denoised is a
    float (L, 512) tensor, not a token grid.
  * The tokenizer is **T5 sentencepiece** (the unembed head is 512x32100), NOT GPT-2.
    `run_baseline_evals.TOK_OVERRIDE` points ELF at gpt2-medium, which is wrong.

Architecture is taken from the official port that ships with this repo
(`baseline/ELF-pytorch/src/modules/{layers,model}.py`, org `embedded-language-flows`)
and is reimplemented here so the script is self-contained on the network-less
compute nodes. Verified 2026-09-20 on both released checkpoints:
`load_state_dict(..., strict=True)` -> 0 missing / 0 unexpected / 0 shape mismatch;
and the decoder head round-trips T5-encoded text back to the input token ids at
95-100% top-1 (ELF-L EMA = 100%), which independently confirms architecture,
EMA weights, encoder and latent_std are all correct.

Sampling follows the official `utils/{sampling_utils,generation_utils}.py` exactly
(SDE/ODE flow rollout + self-cond CFG token + DLM decode head).

NOTE on conditioning: ELF-L/M-owt-torch are the **unconditional OpenWebText**
models -- there is no conditional ELF-L/M release (only ELF-B-xsum / ELF-B-de-en).
They are run here with prefix inpainting (`restore_cond`), which is the same code
path the official `test_generation_cond` uses for the conditional checkpoints.

Usage:
  python scripts/eval/eval_elf_gen_match.py \
      --model_dir  .../models/ELF-L-owt-torch \
      --task_data_dir .../baseline/Cola-DLM/generate_task_data \
      --output_dir results/benchmarks/baseline-elf-l \
      --tasks mmlu,obqa --max_samples 200 --max_new_tokens 32

  # end-to-end sampler validation against the paper's released uncond numbers
  # (Gen. PPL under gpt2-large: ELF-L 23.3, ELF-M 21.7; unigram entropy 5.28 / 5.18)
  python scripts/eval/eval_elf_gen_match.py --model_dir ... --mode uncond \
      --num_samples 64 --output_dir /tmp/elf_uncond
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Official hyperparameters (from the released config.yml / train_owt_ELF-*.yml)
# ----------------------------------------------------------------------------
TEXT_ENCODER_DIM = 512          # t5-small d_model
LATENT_MEAN = 0.0
LATENT_STD = 0.2
DENOISER_NOISE_SCALE = 2.0
T_EPS = 0.05
DENOISER_P_MEAN = -1.5
DENOISER_P_STD = 0.8
SELF_COND_PROB = 0.5
MAX_LENGTH = 1024
T5_VOCAB = 32100                # sentencepiece size (HF config says 32128 padded)

ARCH_PRESETS = {
    # name: (depth, hidden_size, num_heads)   -- official model.py factories
    "ELF-B": (12, 768, 12),
    "ELF-M": (24, 1056, 16),
    "ELF-L": (32, 1280, 16),
}


# ============================================================================
# ELF module (mirrors baseline/ELF-pytorch/src/modules/{layers,model}.py)
# ============================================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


class TextRotaryEmbeddingFast(nn.Module):
    """1D RoPE with `num_empty_token` leading no-rotation slots for prefix tokens."""

    def __init__(self, dim: int, pt_seq_len: int = 512, theta: float = 10000.0,
                 num_empty_token: int = 0):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.theta = theta
        self.num_empty_token = num_empty_token
        c, s = self._compute_freqs()
        self.register_buffer("freqs_cos", c, persistent=False)
        self.register_buffer("freqs_sin", s, persistent=False)

    def _compute_freqs(self):
        dim = self.dim
        freqs = 1.0 / (self.theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim))
        # ft_seq_len == pt_seq_len in the released configs, so pos == arange(n):
        # positions are plain integers and a shorter canvas is exactly faithful.
        pos = torch.arange(self.pt_seq_len, dtype=torch.float32)
        freqs_main = torch.einsum('..., f -> ... f', pos, freqs)
        freqs_main = torch.repeat_interleave(freqs_main, 2, dim=-1)
        D = freqs_main.shape[-1]
        cos_parts, sin_parts = [], []
        if self.num_empty_token > 0:
            cos_parts.append(torch.ones((self.num_empty_token, D), dtype=freqs.dtype))
            sin_parts.append(torch.zeros((self.num_empty_token, D), dtype=freqs.dtype))
        cos_parts.append(torch.cos(freqs_main))
        sin_parts.append(torch.sin(freqs_main))
        c = torch.cat(cos_parts, 0) if len(cos_parts) > 1 else cos_parts[0]
        s = torch.cat(sin_parts, 0) if len(sin_parts) > 1 else sin_parts[0]
        return c, s

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        S = t.shape[-2]
        return t * self.freqs_cos[:S].to(t.dtype) + rotate_half(t) * self.freqs_sin[:S].to(t.dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(dt)
        return self.weight.to(dt) * (x * inv)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True):
        super().__init__()
        self.dim, self.num_heads, self.qk_norm = dim, num_heads, qk_norm
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, rope_fn=None, attention_mask=None):
        B, N, C = x.shape
        hd = self.dim // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope_fn is not None:
            q, k = rope_fn(q), rope_fn(k)
        bm = None
        if attention_mask is not None:
            bm = attention_mask
            if bm.dim() == 2:
                bm = bm[:, None, None, :]
            elif bm.dim() == 3:
                bm = bm[:, None, :, :]
            bm = bm.bool()
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bm)
        return self.proj(x.permute(0, 2, 1, 3).reshape(B, N, C))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        eff = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * eff, bias=True)
        self.w3 = nn.Linear(eff, dim, bias=True)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.mlp_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period)
                          * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].to(torch.float32) * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp_2(F.silu(self.mlp_0(self.timestep_embedding(t, self.frequency_embedding_size))))


class BottleneckTextProj(nn.Module):
    def __init__(self, text_encoder_dim: int, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.proj1 = nn.Linear(text_encoder_dim, bottleneck_dim, bias=False)
        self.proj2 = nn.Linear(bottleneck_dim, hidden_size, bias=True)

    def forward(self, x):
        return self.proj2(self.proj1(x))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        return self.linear(self.norm_final(x))


class ELFBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qk_norm=True)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio))

    def forward(self, x, rope_fn, attention_mask=None):
        x = x + self.attn(self.norm1(x), rope_fn, attention_mask=attention_mask)
        return x + self.mlp(self.norm2(x))


class ELF(nn.Module):
    def __init__(self, text_encoder_dim=TEXT_ENCODER_DIM, max_length=MAX_LENGTH,
                 hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4.0,
                 bottleneck_dim=128, num_time_tokens=4, num_self_cond_cfg_tokens=4,
                 num_model_mode_tokens=0, vocab_size=0):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.bottleneck_dim = bottleneck_dim
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size

        self.self_cond_proj = nn.Linear(2 * text_encoder_dim, text_encoder_dim, bias=True)
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(
                torch.empty(1, num_self_cond_cfg_tokens, hidden_size))
        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.empty(1, num_model_mode_tokens, hidden_size))

        head_dim = hidden_size // num_heads
        prefix_total = num_model_mode_tokens + num_time_tokens
        if num_self_cond_cfg_tokens > 0:
            prefix_total += num_self_cond_cfg_tokens
        self.feat_rope = TextRotaryEmbeddingFast(dim=head_dim, pt_seq_len=max_length,
                                                 num_empty_token=prefix_total)
        self.blocks = nn.ModuleList([ELFBlock(hidden_size, num_heads, mlp_ratio)
                                     for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, text_encoder_dim)
        bn = text_encoder_dim
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, bn))
        self.proj_bias = nn.Parameter(torch.empty(bn))
        self.unembed_kernel = nn.Parameter(torch.empty(bn, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))

    def build_context(self, t, self_cond_cfg_scale=None):
        B = t.shape[0]
        out = [self.t_emb_tokens.expand(B, -1, -1) + self.t_embedder(t).unsqueeze(1)]
        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            out.append(self.self_cond_cfg_tokens.expand(B, -1, -1)
                       + self.self_cond_cfg_embedder(self_cond_cfg_scale).unsqueeze(1))
        return out

    def forward(self, x, t, attention_mask=None, self_cond_cfg_scale=None,
                decoder_step_active=None):
        B = x.shape[0]
        if x.shape[-1] == 2 * self.text_encoder_dim:
            x = self.self_cond_proj(x.float())
        x = self.text_proj(x.float())
        ctx = self.build_context(t, self_cond_cfg_scale)

        mode_off = 0
        if self.num_model_mode_tokens > 0:
            gate = 0.0 if decoder_step_active is None else float(decoder_step_active)
            x = torch.cat([self.mode_tokens.expand(B, -1, -1) * gate, x], dim=1)
            mode_off = self.num_model_mode_tokens
            if attention_mask is not None:
                attention_mask = torch.cat([
                    torch.ones((B, mode_off), dtype=attention_mask.dtype,
                               device=attention_mask.device), attention_mask], dim=1)
        prefix_len = 0
        if ctx:
            pt = torch.cat(ctx, dim=1)
            prefix_len = pt.shape[1]
            x = torch.cat([pt, x], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    torch.ones((B, prefix_len), dtype=attention_mask.dtype,
                               device=attention_mask.device), attention_mask], dim=1)
        for blk in self.blocks:
            x = blk(x, rope_fn=self.feat_rope, attention_mask=attention_mask)
        x = x[:, prefix_len + mode_off:]

        logits = None
        if decoder_step_active is not None:
            h = F.gelu(x.float() @ self.proj_kernel + self.proj_bias, approximate="tanh")
            logits = h @ self.unembed_kernel + self.unembed_bias
        return self.final_layer(x.float()), logits


# ============================================================================
# Architecture inference / loading
# ============================================================================
def infer_arch(params: dict) -> dict:
    """Read the architecture straight out of the checkpoint's own parameter shapes.

    The released dirs ship a config.yml, but that file is 1 KB inside a 5 GB HF
    snapshot and is easy to lose; every architectural constant is recoverable from
    the tensors themselves, so the loader trusts the tensors.
    """
    depth = 1 + max(int(k.split('.')[1]) for k in params if k.startswith("blocks."))
    b0 = "blocks.0."
    hidden = params[b0 + "norm1.weight"].shape[0]
    head_dim = params[b0 + "attn.q_norm.weight"].shape[0]
    if hidden % head_dim:
        raise ValueError(f"hidden {hidden} not divisible by head_dim {head_dim}")
    hidden_eff = params[b0 + "mlp.w3.weight"].shape[1]          # int(int(h*ratio)*2/3)
    ratio = hidden_eff * 3 / 2 / hidden
    arch = dict(
        depth=depth,
        hidden_size=hidden,
        num_heads=hidden // head_dim,
        mlp_ratio=round(ratio),
        vocab_size=params["unembed_kernel"].shape[1],
        num_time_tokens=params["t_emb_tokens"].shape[1],
        num_self_cond_cfg_tokens=params["self_cond_cfg_tokens"].shape[1],
        num_model_mode_tokens=params["mode_tokens"].shape[1],
        bottleneck_dim=params["text_proj.proj1.weight"].shape[0],
        text_encoder_dim=params["text_proj.proj1.weight"].shape[1],
    )
    # sanity: the fused SwiGLU shape must be exactly reproducible from mlp_ratio
    if params[b0 + "mlp.w12.weight"].shape[0] != 2 * hidden_eff:
        raise ValueError("SwiGLU w12 width does not match 2 * w3 input width")
    if hidden_eff != int(int(hidden * arch["mlp_ratio"]) * 2 / 3):
        raise ValueError(f"mlp_ratio={arch['mlp_ratio']} does not reproduce mlp width "
                         f"{hidden_eff} for hidden={hidden}")
    return arch


def find_checkpoint(model_dir: Path) -> Path:
    if model_dir.is_file():
        return model_dir
    cands = sorted([p for p in model_dir.iterdir() if p.name.startswith("checkpoint_")],
                   key=lambda p: int(re.search(r"(\d+)$", p.name).group(1)))
    if not cands:
        raise FileNotFoundError(f"no checkpoint_* in {model_dir}")
    return cands[-1]


def build_elf_model(model_dir: Path, use_ema: bool, device, dtype=torch.float32):
    """Return (model, meta). Raises if `strict=True` does not pass."""
    ckpt_path = find_checkpoint(Path(model_dir))
    ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    if "params" not in ck:
        raise ValueError(f"{ckpt_path} has no 'params' key; keys={list(ck.keys())}")
    params = ck["params"]
    arch = infer_arch(params)
    model = ELF(**arch)

    # Both tensors and module are constructed from the same inferred arch, so any
    # mismatch here is a real architecture error. strict=True is deliberate: a
    # silent partial load would score a half-random model.
    model.load_state_dict(params, strict=True)
    src = "params"
    if use_ema:
        if "ema_params1" not in ck:
            raise ValueError("--use_ema requested but checkpoint has no ema_params1")
        model.load_state_dict(ck["ema_params1"], strict=True)
        src = "ema_params1"

    meta = dict(arch=arch, ckpt=str(ckpt_path), weight_source=src,
                step=int(ck["step"]) if "step" in ck else None,
                epoch=int(ck["epoch"]) if "epoch" in ck else None,
                has_ema="ema_params1" in ck)
    del ck
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


# ============================================================================
# Flow-matching sampler (mirrors utils/sampling_utils.py + generation_utils.py)
# ============================================================================
def get_sampling_steps(n_steps, time_schedule, p_mean, p_std, device, dtype, generator=None):
    """Length-(n_steps+1) ascending t in [0,1]; endpoints pinned to 0 and 1."""
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype, device=device)
    z = torch.randn((n_steps - 1,), generator=generator, dtype=dtype) * p_std + p_mean
    steps = torch.sort(torch.sigmoid(z.to(device))).values
    return torch.cat([torch.zeros(1, dtype=dtype, device=device), steps,
                      torch.ones(1, dtype=dtype, device=device)], dim=0)


def restore_cond(z_updated, cond_seq, cond_seq_mask):
    """Clamp already-known (conditioning) positions to their clean values."""
    if cond_seq is None:
        return z_updated
    mask = cond_seq_mask
    while mask.dim() < z_updated.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def net_out_to_v_x(net_out, z, t, t_eps=T_EPS):
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    x = net_out
    v = (x - z) / torch.clamp(1.0 - t.reshape(-1, 1, 1), min=t_eps)
    return v, x


class ELFSampler:
    """Official SDE/ODE flow rollout with self-conditioning."""

    def __init__(self, model: ELF, sampling_method="sde", num_sampling_steps=64,
                 sde_gamma=1.0, self_cond_cfg_scale=3.0, time_schedule="logit_normal",
                 use_bf16=True, noise_scale=DENOISER_NOISE_SCALE):
        self.model = model
        self.method = sampling_method
        self.n_steps = num_sampling_steps
        self.gamma = sde_gamma
        self.sc_cfg = self_cond_cfg_scale
        self.time_schedule = time_schedule
        self.use_bf16 = use_bf16 and next(model.parameters()).is_cuda
        self.noise_scale = noise_scale
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    def _forward(self, z, t_batch, x_pred_prev, cond_seq, cond_seq_mask):
        """One guided velocity/x0 estimate (mirrors `_forward_sample`)."""
        if self.model.num_self_cond_cfg_tokens > 0:
            if x_pred_prev is None:
                x_pred_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
            zin = torch.cat([z, x_pred_prev], dim=-1)
            sc = torch.full((z.shape[0],), float(self.sc_cfg), dtype=z.dtype, device=z.device)
            out = self.model(zin, t_batch, self_cond_cfg_scale=sc)
            v, x = net_out_to_v_x(out, z, t_batch)
            return restore_cond(v, cond_seq, cond_seq_mask), restore_cond(x, cond_seq, cond_seq_mask)

        zin = torch.cat([z, restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)], dim=-1)
        out = self.model(zin, t_batch)
        v, x = net_out_to_v_x(out, z, t_batch)
        return restore_cond(v, cond_seq, cond_seq_mask), restore_cond(x, cond_seq, cond_seq_mask)

    def _step(self, z, x_pred, t, t_next, cond_seq, cond_seq_mask, generator, method=None):
        if (method or self.method) == "sde":
            h = float(t_next - t)
            alpha = max(0.0, min(1.0, 1.0 - self.gamma * h))
            t_back = alpha * float(t)
            eps = torch.randn(z.shape, generator=generator, dtype=z.dtype).to(z.device) * self.noise_scale
            z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
        else:
            t_back, z_back = float(t), z
        tb = torch.full((z.shape[0],), t_back, dtype=z.dtype, device=z.device)
        v, x = self._forward(z_back, tb, x_pred, cond_seq, cond_seq_mask)
        return z_back + (t_next - t_back) * v, x

    @torch.no_grad()
    def sample(self, cond_seq, cond_seq_mask, canvas_len, generator, attention_mask=None):
        """Denoise a (B, canvas_len, 512) canvas with `cond_seq` clamped.

        Returns the final latent z (t -> 1).
        """
        B = cond_seq.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones((B, canvas_len), dtype=torch.float32, device=self.device)
        z = (torch.randn((B, canvas_len, self.model.text_encoder_dim),
                         generator=generator, dtype=self.dtype) * self.noise_scale).to(self.device)
        z = restore_cond(z, cond_seq, cond_seq_mask)
        x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

        t_steps = get_sampling_steps(self.n_steps, self.time_schedule,
                                     DENOISER_P_MEAN, DENOISER_P_STD,
                                     self.device, self.dtype, generator)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
            for i in range(self.n_steps - 1):
                z, x_pred = self._step(z, x_pred, t_steps[i].item(), t_steps[i + 1].item(),
                                       cond_seq, cond_seq_mask, generator)
            # official: the final step is always ODE
            z, x_pred = self._step(z, x_pred, t_steps[-2].item(), t_steps[-1].item(),
                                   cond_seq, cond_seq_mask, generator, method="ode")
        return z, t_steps[-1].item()

    @torch.no_grad()
    def decode_ids(self, z, t_final):
        """DLM decode head: latent -> token ids (mirrors `_dlm_decode_batch`)."""
        B = z.shape[0]
        t_batch = torch.full((B,), float(t_final), dtype=z.dtype, device=z.device)
        sc = (torch.full((B,), float(self.sc_cfg), dtype=z.dtype, device=z.device)
              if self.model.num_self_cond_cfg_tokens > 0 else None)
        zin = torch.cat([z, torch.zeros_like(z)], dim=-1) if SELF_COND_PROB > 0 else z
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
            _, logits = self.model(zin, t_batch, self_cond_cfg_scale=sc,
                                   decoder_step_active=True)
        return logits.argmax(dim=-1)


# ============================================================================
# T5 text encoder (frozen; the ELF latents live in its output space)
# ============================================================================
class T5TextEncoder:
    def __init__(self, model_name="t5-small", device="cpu", use_bf16=True):
        from transformers import T5EncoderModel
        # Prefer the fast (Rust `tokenizers`) T5 tokenizer: T5Tokenizer needs the
        # `sentencepiece` wheel, which is not in shared_pylibs and cannot be pip-installed
        # on the network-less compute nodes. t5-small ships tokenizer.json, so the fast
        # path is equivalent and dependency-free.
        try:
            from transformers import T5TokenizerFast as _T5Tok
        except ImportError:
            from transformers import T5Tokenizer as _T5Tok
        self.tok = _T5Tok.from_pretrained(model_name, local_files_only=True)
        self.enc = T5EncoderModel.from_pretrained(model_name, local_files_only=True)
        self.enc.eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.enc = self.enc.to(device)
        self.device = device
        self.use_bf16 = use_bf16 and torch.device(device).type == "cuda"

    @torch.no_grad()
    def encode(self, texts, max_tokens):
        """list[str] -> (B, max_len) ids, (B, max_len) mask, (B, max_len, 512) latents."""
        enc = self.tok(list(texts), add_special_tokens=False, padding=True,
                       truncation=True, max_length=max_tokens, return_tensors="pt")
        ids = enc.input_ids.to(self.device)
        mask = enc.attention_mask.to(self.device).float()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
            lat = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        lat = (lat.float() - LATENT_MEAN) / LATENT_STD
        return ids, mask, lat


def mask_after_eos(ids, eos_id, pad_id):
    keep = (ids == eos_id).to(torch.int32).cumsum(dim=1) == 0
    return torch.where(keep, ids, torch.full_like(ids, pad_id))


def shift_left(x, shift_per_sample, pad_value=0):
    seq_len = x.shape[1]
    base = torch.arange(seq_len, device=x.device)[None, :]
    idx = (shift_per_sample[:, None].to(x.device) + base)
    valid = idx < seq_len
    out = torch.gather(x, 1, idx.clamp(0, seq_len - 1))
    return torch.where(valid, out, torch.full_like(out, pad_value))


def clean_text(txt, tokenizer):
    for special in getattr(tokenizer, "all_special_tokens", ()) or ():
        txt = txt.replace(special, " ")
    return re.sub(r"[ \t]{2,}", " ", txt).strip()


# ============================================================================
# Task harness
# ============================================================================
_OPTION_MARKER_RE = re.compile(r"\(A\)")


def inline_choices(prompt: str, item: dict) -> str:
    """Inline the item's ``choices`` when the prompt does not already carry them.

    Same defect and same guard as eval_generic_gen_match.inline_choices: the six
    tasks arc_easy/arc_challenge/winogrande/sciq/commonsenseqa/piqa ship ``choices``
    but no option list in ``prompt``. Tasks that already carry a ``(A) ...`` block
    are returned byte-identical, so this is idempotent.
    """
    choices = item.get("choices") if isinstance(item, dict) else None
    if not isinstance(choices, (list, tuple)) or not choices:
        return prompt
    m = _OPTION_MARKER_RE.search(prompt)
    if m is not None and str(choices[0]).strip().lower() in prompt[m.start():].lower():
        return prompt  # already carries an option list -- never add a second copy
    block = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
    cut = prompt.rfind("Answer:")
    if cut == -1:
        return f"{prompt}\n{block}\nAnswer:"
    return f"{prompt[:cut].rstrip()}\n{block}\nAnswer:"


def build_prompt(task: str, item: dict) -> str:
    """Mirror of eval_generic_gen_match.build_prompt (same prompt construction)."""
    if isinstance(item.get("prompt"), str) and item["prompt"]:
        prefix = item.get("few_shot_prefix")
        if isinstance(prefix, str) and prefix and not item["prompt"].startswith(prefix):
            return inline_choices(prefix + item["prompt"], item)
        return inline_choices(item["prompt"], item)
    parts = [str(item.get("few_shot_prefix") or ""), str(item.get("context") or ""),
             str(item.get("question") or ""), str(item.get("prompt") or "")]
    return inline_choices("\n".join(p for p in parts if p), item)


def get_ground_truth(item: dict) -> str:
    return str(item.get("ground_truth", item.get("answer", item.get("target", ""))))


@torch.no_grad()
def run_tasks(args, model, meta, sampler, txt_enc):
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    eos_id = txt_enc.tok.eos_token_id
    pad_id = txt_enc.tok.pad_token_id

    max_prompt = args.max_prompt_tokens if args.max_prompt_tokens > 0 else MAX_LENGTH - args.max_new_tokens
    max_prompt = min(max_prompt, MAX_LENGTH - args.max_new_tokens)

    summary = dict(model_dir=str(args.model_dir), checkpoint=meta["ckpt"],
                   weight_source=meta["weight_source"], arch=meta["arch"],
                   max_samples=args.max_samples, max_new_tokens=args.max_new_tokens,
                   sampling_method=sampler.method, num_sampling_steps=sampler.n_steps,
                   sde_gamma=sampler.gamma, self_cond_cfg_scale=sampler.sc_cfg,
                   max_prompt_tokens=max_prompt, protocol="generate-then-match (prefix inpainting)",
                   tasks={})
    print(f"[elf] arch={meta['arch']} weights={meta['weight_source']} step={meta['step']}", flush=True)

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        path = Path(args.task_data_dir) / f"{task}.jsonl"
        if not path.exists():
            print(f"[{task}] SKIP (no {path})", flush=True)
            continue
        items = []
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if len(items) >= args.max_samples:
                    break
                line = line.strip()
                if line:
                    items.append((i, json.loads(line)))
        t0 = time.time()
        out_path = out_root / f"{task}.jsonl"
        n_trunc = 0

        # Length-bucket: the diffusion canvas is (longest prompt in batch + max_new_tokens),
        # so one long RACE prompt in an unsorted batch inflates the cost of every row.
        # Sort by prompt length, then restore the original order when writing.
        prompts = [build_prompt(task, it) for _, it in items]
        # verbose=False: this call measures the *untruncated* length, so it trips the
        # tokenizer's "longer than model_max_length (512)" warning for RACE. Harmless --
        # the real encode below passes truncation=True with the 992-token ELF cap.
        raw_lens = [len(txt_enc.tok(p, add_special_tokens=False, verbose=False)["input_ids"])
                    for p in prompts]
        order = sorted(range(len(items)), key=lambda k: raw_lens[k])
        records: dict[int, dict] = {}

        for start in range(0, len(order), args.batch_size):
            sel = order[start:start + args.batch_size]
            chunk_prompts = [prompts[k] for k in sel]
            ids, mask, lat = txt_enc.encode(chunk_prompts, max_prompt)
            lens = [int(m.sum().item()) for m in mask]
            n_trunc += sum(1 for L in lens if L >= max_prompt)
            longest = max(lens)
            canvas = min(MAX_LENGTH, longest + args.max_new_tokens)
            B = len(sel)

            cond_seq = torch.zeros((B, canvas, model.text_encoder_dim),
                                   dtype=torch.float32, device=device)
            cond_mask = torch.zeros((B, canvas), dtype=torch.float32, device=device)
            for b, L in enumerate(lens):
                cond_seq[b, :L] = lat[b, :L]
                cond_mask[b, :L] = 1.0
            attn = torch.zeros((B, canvas), dtype=torch.float32, device=device)
            attn[:, :longest] = 1.0

            z, t_final = sampler.sample(cond_seq, cond_mask, canvas, gen, attention_mask=attn)
            pred = sampler.decode_ids(z, t_final)
            # official cond path: drop the conditioning prefix, keep the generated tail
            cond_len = cond_mask.sum(dim=1).to(torch.int64)
            pred = shift_left(pred, cond_len, 0)[:, :args.max_new_tokens]
            pred = mask_after_eos(pred, eos_id, pad_id)

            for b, k in enumerate(sel):
                orig_i, item = items[k]
                text = clean_text(
                    txt_enc.tok.decode(pred[b].detach().cpu().numpy(),
                                       skip_special_tokens=True), txt_enc.tok)
                rec = dict(item)
                rec.update({
                    "id": item.get("id", orig_i), "prompt": prompts[k],
                    "generate": text, "ground_truth": get_ground_truth(item),
                    "choices": item.get("choices", []),
                    "prompt_tokens": lens[b], "generated_tokens": int(args.max_new_tokens),
                    "method": f"elf-{sampler.method}{sampler.n_steps}",
                    "cfg_scale": 0.0, "steps": sampler.n_steps,
                    "baseline_model": str(args.model_dir)})
                records[k] = rec
            print(f"[{task}] {min(start + args.batch_size, len(order))}/{len(order)} "
                  f"({time.time() - t0:.0f}s, canvas={canvas})", flush=True)

        with out_path.open("w", encoding="utf-8") as fo:
            for k in range(len(items)):
                fo.write(json.dumps(records[k], ensure_ascii=False) + "\n")
        n_lines = sum(1 for _ in out_path.open())
        summary["tasks"][task] = dict(n=n_lines, elapsed_s=round(time.time() - t0, 1),
                                      truncated_prompts=n_trunc)
        print(f"[{task}] done n={n_lines} truncated={n_trunc} "
              f"in {time.time() - t0:.0f}s -> {out_path}", flush=True)

    (out_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


# ============================================================================
# Unconditional mode: end-to-end sampler validation vs the released numbers
# ============================================================================
@torch.no_grad()
def run_uncond(args, model, meta, sampler, txt_enc):
    """Sample from the model with no conditioning and report gen-PPL under gpt2-large.

    The ELF paper/README report, for 64-step SDE (SC-CFG=3, gamma=1.0) on
    OpenWebText: ELF-M Gen.PPL 21.7 / entropy 5.18, ELF-L 23.3 / 5.28.
    Reproducing those is the strongest available evidence that the sampler and
    weights are right -- it does not depend on any key-name reverse engineering.
    """
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    eos_id, pad_id = txt_enc.tok.eos_token_id, txt_enc.tok.pad_token_id

    texts = []
    t0 = time.time()
    done = 0
    canvas = args.uncond_max_length
    while done < args.num_samples:
        b = min(args.batch_size, args.num_samples - done)
        cond_seq = torch.zeros((b, canvas, model.text_encoder_dim), device=device)
        cond_mask = torch.zeros((b, canvas), device=device)
        z, t_final = sampler.sample(cond_seq, cond_mask, canvas, gen)
        pred = mask_after_eos(sampler.decode_ids(z, t_final), eos_id, pad_id)
        for r in pred:
            texts.append(txt_enc.tok.decode(r.detach().cpu().numpy(), skip_special_tokens=True))
        done += b
        print(f"[uncond] {done}/{args.num_samples} ({time.time() - t0:.0f}s)", flush=True)

    out_path = out_root / "all_generated_uncond.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"id": i, "generated": t}, ensure_ascii=False) + "\n")

    result = dict(model_dir=str(args.model_dir), n=len(texts),
                  method=f"{sampler.method}{sampler.n_steps}", sc_cfg=sampler.sc_cfg,
                  gamma=sampler.gamma, entropy=None, ppl=None)
    nonempty = [t for t in texts if t.strip()]
    print(f"[uncond] {len(nonempty)}/{len(texts)} non-empty; sample: {nonempty[0][:200] if nonempty else ''}",
          flush=True)
    if nonempty:
        if args.ppl_model:
            p, h = compute_gen_ppl_and_entropy(nonempty, args.ppl_model,
                                               args.ppl_batch_size, device)
            result["ppl"] = p
            result["entropy"] = h
            print(f"[uncond] Gen.PPL ({args.ppl_model}) = {p:.3f}", flush=True)
            print(f"[uncond] mean per-sample unigram entropy = {h:.3f}", flush=True)
        else:
            result["entropy"] = compute_unigram_entropy(nonempty)
            print(f"[uncond] mean unigram entropy (whitespace) = {result['entropy']:.3f}", flush=True)
    (out_root / "uncond_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def compute_unigram_entropy(texts):
    """Fallback entropy when no GPT-2 tokenizer is available (whitespace tokens).

    NOTE: NOT comparable to the released 5.18/5.28 numbers -- see
    `compute_gen_ppl_and_entropy` for the published definition.
    """
    from collections import Counter
    c = Counter()
    for t in texts:
        c.update(t.split())
    n = sum(c.values())
    if not n:
        return 0.0
    return float(-sum((v / n) * math.log(v / n) for v in c.values()))


@torch.no_grad()
def compute_gen_ppl_and_entropy(texts, model_name, batch_size, device):
    """Gen. PPL + mean per-sample unigram entropy, as the official metrics_utils does.

    `mean_entropy` in the official `Metrics.record_generative_perplexity` is the mean over
    samples of the Shannon entropy of that sample's **GPT-2 token-id** histogram, not a
    whitespace-word entropy over the pooled corpus. (Those two differ a lot: 7.3 vs 5.2.)
    """
    import numpy as np
    from collections import Counter
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    lm = lm.to(device).eval()
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    tot_nll, tot_tok = 0.0, 0
    entropies = []
    for s in range(0, len(texts), batch_size):
        chunk = texts[s:s + batch_size]
        enc = tok(chunk, return_tensors="np", padding=True, truncation=True,
                  max_length=1024, return_token_type_ids=False)
        ids_np = np.asarray(enc["input_ids"])
        am_np = np.asarray(enc["attention_mask"])
        for i in range(ids_np.shape[0]):
            vlen = int(am_np[i].sum())
            counts = np.unique(ids_np[i, :vlen], return_counts=True)[1]
            p = counts.astype(np.float32) / counts.sum()
            entropies.append(float(-np.sum(p * np.log(p + 1e-10))))
        ids = torch.from_numpy(ids_np).to(device)
        am = torch.from_numpy(am_np).to(device)
        shift_logits = lm(input_ids=ids, attention_mask=am).logits[:, :-1].float().contiguous()
        shift_labels = ids[:, 1:].contiguous()
        shift_mask = am[:, 1:].contiguous().float()
        nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nll = nll.view(shift_labels.shape) * shift_mask
        tot_nll += float(nll.sum())
        tot_tok += int(shift_mask.sum())
    del lm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (float(math.exp(tot_nll / max(tot_tok, 1))),
            float(sum(entropies) / max(len(entropies), 1)))


# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True,
                   help="dir containing checkpoint_<step> + config.yml (e.g. models/ELF-L-owt-torch)")
    p.add_argument("--task_data_dir", default=None,
                   help="dir of <task>.jsonl (required unless --mode uncond)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", default="mmlu,obqa,siqa,story_cloze,hellaswag,piqa,arc_easy,"
                                      "arc_challenge,winogrande,sciq,commonsenseqa,lambada,"
                                      "squad,race")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    # ELF sampler knobs (defaults = the official 64-step uncond_SDE config)
    p.add_argument("--sampling_method", default="sde", choices=["sde", "ode"])
    p.add_argument("--num_sampling_steps", "--steps", dest="num_sampling_steps", type=int, default=64)
    p.add_argument("--sde_gamma", type=float, default=1.0)
    p.add_argument("--self_cond_cfg_scale", type=float, default=3.0)
    p.add_argument("--time_schedule", default="logit_normal", choices=["logit_normal", "uniform"])
    p.add_argument("--no_ema", dest="use_ema", action="store_false", default=True,
                   help="load 'params' instead of 'ema_params1' (official eval uses EMA)")
    p.add_argument("--max_prompt_tokens", type=int, default=0,
                   help="0 = no extra cap (bounded by 1024 - max_new_tokens)")
    p.add_argument("--t5_model", default="t5-small",
                   help="frozen T5 encoder used to build ELF latents")
    p.add_argument("--no_bf16", dest="use_bf16", action="store_false", default=True)
    # uncond validation mode
    p.add_argument("--mode", default="tasks", choices=["tasks", "uncond"])
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--uncond_max_length", type=int, default=MAX_LENGTH,
                   help="uncond canvas; the released models were trained at 1024, so only "
                        "lower this for smoke tests")
    p.add_argument("--ppl_model", default=None,
                   help="causal LM for gen-PPL in --mode uncond (e.g. a local gpt2-large dir)")
    p.add_argument("--ppl_batch_size", type=int, default=8)
    # accepted for CLI parity with the sibling harnesses; ELF ignores them
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--tokenizer_dir", default=None)
    args = p.parse_args()
    if args.mode != "uncond" and not args.task_data_dir:
        p.error("--task_data_dir is required in --mode tasks")
    if args.tokenizer_dir:
        print(f"[elf] WARNING: --tokenizer_dir={args.tokenizer_dir} ignored; ELF's unembed head "
              f"is T5 sentencepiece (32100), not GPT-2. Using {args.t5_model}.", flush=True)
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"[elf] loading checkpoint from {args.model_dir}", flush=True)
    model, meta = build_elf_model(Path(args.model_dir), args.use_ema, device)
    print(f"[elf] strict=True load OK; arch={meta['arch']} weights={meta['weight_source']}", flush=True)
    txt_enc = T5TextEncoder(args.t5_model, device, args.use_bf16)
    sampler = ELFSampler(model, sampling_method=args.sampling_method,
                         num_sampling_steps=args.num_sampling_steps,
                         sde_gamma=args.sde_gamma,
                         self_cond_cfg_scale=args.self_cond_cfg_scale,
                         time_schedule=args.time_schedule, use_bf16=args.use_bf16)
    if args.mode == "uncond":
        run_uncond(args, model, meta, sampler, txt_enc)
    else:
        run_tasks(args, model, meta, sampler, txt_enc)


if __name__ == "__main__":
    main()

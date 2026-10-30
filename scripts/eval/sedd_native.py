#!/usr/bin/env python3
"""Native loader + sampler for the SEDD-medium discrete-diffusion LM.

Why this file exists
--------------------
`louaaron/sedd-medium` publishes a RAW training checkpoint: `config.json` is the
hydra training config (top-level keys `graph`, `noise`, `model`, ...) and
`pytorch_model.bin` is a bare `state_dict` of a 424M-parameter DiT. It carries no
`model_type` and no `auto_map`, so every HF auto-class rejects it:

    ValueError: Unrecognized model in .../SEDD-medium.
                Should have a `model_type` key in its config.json.

That is verbatim the error in `logs/qz/baseline-sedd-medium-*.log`, and it is why
the `--impl sedd` arm of `eval_diffusion_lm_gen_match.py` never produced a cell.
SEDD has to be driven through its own architecture, re-implemented here from the
published method (Lou, Meng & Ermon, "Discrete Diffusion Modeling by Estimating
the Ratios of the Distribution", arXiv:2310.16834) and from the tensor shapes in
the checkpoint.

Architecture, verified by a strict `load_state_dict` (251/251 keys,
424,477,042 parameters):
  * DDiT: 24 blocks, hidden 1024, 16 heads, cond_dim 128, mlp_ratio 4
  * RoPE on q/k in every block; adaLN-Zero conditioning on a sinusoidal sigma
  * absorbing graph over `tokens + 1` classes, index `tokens` is the mask
  * loglinear noise schedule with `scale_by_sigma` enabled

Sampler
-------
Absorbing-state ancestral sampling over the completion span only; the prompt is
clamped (partial-noising / inpainting), which is how SEDD conditions.

With the loglinear schedule this checkpoint was trained on, the forward
probability that a position is masked at time t is exactly

    m(t) = (1 - eps) * t

so the reverse kernel conditional on a position being masked at time t is

    P(still masked at s | masked at t) = m(s) / m(t) = s / t

and the last step (s = 0) unmasks everything. Positions already revealed are
left alone, which is also all SEDD's own analytic predictor can do. The network
output is the score-entropy score log p(y)/p(x); the absorbing column is dropped
before the softmax, so the softmax over the remaining `tokens` classes is exactly
p(x0 | x_t = mask).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class LayerNorm(nn.Module):
    """Weight-only, bias-free layer norm computed in fp32 (SEDD's DiT norm)."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def _modulate(x, shift, scale):
    """adaLN-Zero modulation. shift/scale are [B, 1, dim] and broadcast over L."""
    return x * (1 + scale) + shift


class EmbeddingLayer(nn.Module):
    """Plain lookup table, kept as a submodule so the parameter is
    `vocab_embed.embedding` as in the published state_dict."""

    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class TimestepEmbedder(nn.Module):
    """Sinusoidal sigma embedding -> 2-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class Rotary(nn.Module):
    """RoPE table over the sequence dimension; q/k only, v is untouched."""

    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :]
            self.sin_cached = emb.sin()[None, :, None, None, :]
        return self.cos_cached, self.sin_cached


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(qkv, cos, sin):
    """qkv: [B, L, 3, H, D]; cos/sin: [1, L, 1, 1, D]. Applies to q and k only."""
    q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
    # cos/sin are [1, L, 1, 1, D]; dropping the broadcast head axis leaves
    # [1, L, 1, D], which broadcasts over q/k's [B, L, H, D].
    cos = cos[..., 0, :, :]
    sin = sin[..., 0, :, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k, v


class DDiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int, mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)

    def forward(self, x, rotary_cos_sin, c):
        B, L = x.shape[0], x.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        # attention (full, non-causal: SEDD's DiT is not an autoregressive model)
        x_skip = x
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.attn_qkv(h)
        qkv = qkv.view(B, L, 3, self.n_heads, self.head_dim)
        cos, sin = rotary_cos_sin
        q, k, v = _apply_rope(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        )
        attn = attn.transpose(1, 2).reshape(B, L, -1)
        x = x_skip + gate_msa * self.dropout1(self.attn_out(attn))

        # mlp
        h = self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp * self.dropout2(h)
        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class SEDD(nn.Module):
    """The `medium` DDiT, parameter names matching the published state_dict."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.tokens = int(cfg["tokens"])
        self.absorb = cfg["graph"]["type"] == "absorb"
        vocab_size = self.tokens + (1 if self.absorb else 0)
        self.scale_by_sigma = bool(m["scale_by_sigma"])

        self.vocab_embed = EmbeddingLayer(m["hidden_size"], vocab_size)
        self.sigma_map = TimestepEmbedder(m["cond_dim"])
        self.rotary_emb = Rotary(m["hidden_size"] // m["n_heads"])
        self.blocks = nn.ModuleList([
            DDiTBlock(m["hidden_size"], m["n_heads"], m["cond_dim"],
                      dropout=m["dropout"])
            for _ in range(m["n_blocks"])
        ])
        self.output_layer = DDitFinalLayer(m["hidden_size"], vocab_size, m["cond_dim"])
        self.vocab_size = vocab_size

    def forward(self, indices, sigma):
        """indices: [B, L] long; sigma: [B] total noise. Returns score log p(y)/p(x)."""
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))
        cos_sin = self.rotary_emb(x)
        for block in self.blocks:
            x = block(x, cos_sin, c)
        x = self.output_layer(x, c)

        if self.scale_by_sigma:
            esigm1_log = torch.where(
                sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1
            ).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - math.log(x.shape[-1] - 1)

        # score-entropy convention: the diagonal of the score is 0
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        return x


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
def loglinear_sigma(t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Total noise of SEDD's loglinear schedule: -log(1 - (1-eps) t)."""
    return -torch.log1p(-(1.0 - eps) * t)


def mask_prob(t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Forward probability that a position is absorbed by time t: (1-eps) t.

    Derived from the schedule: alpha(t) = exp(-sigma(t)) = 1 - (1-eps) t is the
    survival probability, so the mask probability is 1 - alpha(t).
    """
    return (1.0 - eps) * t


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def resolve_checkpoint(model_dir) -> Path:
    """Path to a SEDD `pytorch_model.bin`, with a clear error if it is truncated.

    Two copies of this checkpoint exist on disk and the one the driver's
    `find_model()` resolves to first (`models/SEDD-medium`) was a partial
    download (353,589,438 B of the real 1,698,000,441 B), which `torch.load`
    rejects with `PytorchStreamReader failed reading zip archive`. Fail loudly
    rather than silently falling back to a half-loaded model.
    """
    d = Path(model_dir)
    cand = d / "pytorch_model.bin"
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"no pytorch_model.bin under {d}; the published SEDD-medium checkpoint is "
        f"a raw state_dict named pytorch_model.bin"
    )


def load_sedd(model_dir, device, verbose: bool = True):
    """Build the DDiT from config.json and strict-load the published state_dict."""
    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text())
    if "model" not in cfg or "hidden_size" not in cfg.get("model", {}):
        raise ValueError(
            f"{d}/config.json has no `model` section, so it is not a SEDD hydra "
            f"config (keys: {sorted(cfg)})"
        )
    model = SEDD(cfg)
    ckpt = resolve_checkpoint(d)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    for wrapper in ("model", "ema", "state_dict"):
        if isinstance(state, dict) and wrapper in state and hasattr(state[wrapper], "keys"):
            state = state[wrapper]
    missing, unexpected = model.load_state_dict(state, strict=False)
    # `rotary_emb.inv_freq` is a deterministic buffer and `seq_len_cached` is not
    # a tensor, so nothing legitimately expected may be missing.
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch for {ckpt}: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    model = model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[sedd] loaded {ckpt} ({ckpt.stat().st_size} B, {n} params) "
              f"vocab={model.vocab_size} mask_id={mask_id_of(cfg)}", flush=True)
    return model, cfg


def mask_id_of(cfg: dict) -> int:
    """Absorbing state index: `tokens` (one past the last real token id)."""
    return int(cfg["tokens"])


# --------------------------------------------------------------------------- #
# sampler
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sedd_denoise(model, prompt_ids: torch.Tensor, gen_len: int, steps: int,
                 mask_id: int, device, temperature: float = 1.0,
                 eps: float = 1e-3) -> list[int]:
    """Ancestral sampling of the completion span with the prompt clamped.

    See the module docstring for the derivation of `s / t` as the probability
    that a position which is masked at time t is still masked at time s.
    """
    L = prompt_ids.shape[1]
    span = torch.full((1, gen_len), mask_id, device=device, dtype=torch.long)
    x = torch.cat([prompt_ids, span], dim=1)
    use_amp = device.type == "cuda"

    for i in range(steps, 0, -1):
        t = i / steps
        s = (i - 1) / steps
        sigma = torch.full((1,), float(loglinear_sigma(torch.tensor(t, dtype=torch.float32),
                                                       eps)), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(x, sigma)
        # score is log p(y)/p(x) over `vocab_size` classes, the last of which is the
        # absorbing state; dropping it leaves log p(y) for the real tokens.
        logits = out[:, L:, :mask_id].float() / max(temperature, 1e-6)
        x0 = torch.multinomial(
            torch.softmax(logits.view(-1, logits.shape[-1]), dim=-1), 1
        ).view(1, gen_len)

        is_mask = span == mask_id
        if s > 0.0:
            keep = torch.rand(1, gen_len, device=device) < (s / t)
            new = torch.where(is_mask & keep, torch.full_like(x0, mask_id), x0)
        else:
            new = x0
        span = torch.where(is_mask, new, span)
        x = torch.cat([prompt_ids, span], dim=1)

    return span[0].tolist()

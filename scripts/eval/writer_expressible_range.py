#!/usr/bin/env python3
"""Expressible range of the fixed-dictionary vs dynamic low-rank state writer.

Why this exists: the writing-interface ablation reports task accuracy, which mixes
the writer's reach with everything downstream of it (prior, decoder margin, prompt).
This script measures the writer alone. Both writers are pure functions of the plan
``z``, so nothing here needs the decoder, the corpus, or a GPU.

The two writers, verified against ``models/state_hijacking_dit.py:1150-1178``:

  fixed    (``alpha_type='linear'`` => ``features = z``, ``alpha_trunk is None``)
           M_l(z) = sum_k [W_l z + b_l]_k B_{l,k},  B fixed after training
  dynamic  h = GELU(L2(GELU(L1(z))));  M_l(z) = U_l(h) V_l(h)^T

The distinction the ablation turns on is *not* the rank of one written matrix -- both
families write a rank<=32 update. It is the dimension of the set of matrices the writer
can reach as the plan varies. For the fixed writer the coefficients are affine in z and
the bases are frozen, so every written state lies in one affine subspace of dimension at
most K=16 -- one universal condition, re-weighted. The dynamic writer is quadratic in a
128-dimensional trunk feature, so its reachable set is a nonlinear manifold whose linear
span is not bounded by any small constant. We therefore sweep the number of sampled
plans N: a writer whose reach is a K-dimensional subspace saturates at K and stays there,
while one that re-aims per input keeps adding directions as N grows.

Metrics, per layer, over N plans z ~ N(0,I) (the prior the sampler draws from):

  reach_dim    dim span{M(z_i) - mean_i M(z_i)}, i.e. how many distinct state directions
               the writer can reach at all. Compared against N and against the ambient
               dimension (64*64 = 4096 per head).
  reach_erank  participation ratio exp(H) of that span's squared spectrum -- the soft
               version of reach_dim, robust to near-degenerate directions.
  vary_frac    ||M - mean_z M||_F / ||M||_F: the share of the write that depends on the
               plan rather than being the same offset for every input.
  mat_erank    effective rank of ONE written 64x64 matrix (what the decoder receives).
  dir_overlap  normalised overlap of the top-r column spaces of M(z_i) and M(z_j),
               1.0 = the write directions never move with the input, 0 = fully re-aimed.

Measured on one head per layer: alpha_l is shared across heads for the fixed writer and
V_l is shared across heads for the dynamic one, so the per-head figure is the per-layer
figure at 1/40 the cost. float64 throughout: in float32 the fixed writer's reach_dim
reads as ~126 instead of 16, which is rounding noise, not reach.

Usage:  OMP_NUM_THREADS=8 python3 scripts/eval/writer_expressible_range.py
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning"
PAIR = {
    "fixed": f"{ROOT}/outputs_dynamic_basis/laces-2.9B-fixed-r32-global-birwkv768x8-b16-50k/step_00050000/model.pt",
    "dynamic": f"{ROOT}/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-birwkv768x8-b16-50k/step_00050000/model.pt",
}
# The two runs differ only in s1_writer_type: same denoiser (birwkv 768x8), same batch
# size, same corpus, same step. config_name reads "dynlowrank" in BOTH checkpoints, so
# it is not a usable discriminator; s1_writer_type is.
WANT = {"fixed": "fixed", "dynamic": "dynlowrank"}

NUM_LAYERS, NUM_HEADS, HEAD_DIM, K, RANK, LATENT = 32, 40, 64, 16, 32, 32
HEAD = 0
LAYERS = [0, 8, 16, 24, 31]
N_SWEEP = [16, 32, 64, 128, 256]
# The dynamic writer returns N-1 at every N in N_SWEEP, i.e. the sample size is the
# binding constraint there, not the writer. N_SATURATE pushes past the ambient dimension
# (64*64 = 4096 per head) on a few layers to find where its reach actually stops.
N_SATURATE = [512, 1024, 2048, 4096]
SATURATE_LAYERS = [0, 16, 31]
N_MAIN = 256
RTOL = 1e-9  # relative to the top singular value

Z_ALL = torch.randn(max(N_SWEEP + N_SATURATE + [N_MAIN]), LATENT,
                    generator=torch.Generator().manual_seed(0))


def erank(sv: torch.Tensor, rel: float = 1e-13) -> float:
    """Participation ratio of the squared spectrum."""
    s = sv.clamp_min(0) ** 2
    s = s[s > s.max() * rel]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def load(tag: str):
    obj = torch.load(PAIR[tag], map_location="cpu", weights_only=False)
    cfg = obj["config"]
    assert f"s1_writer_type: {WANT[tag]}" in cfg, f"{tag}: writer type mismatch"
    return obj["trainable_state"], obj["state_basis"], float(obj["state_scale"].item())


def write_states(tag: str, sd, basis, scale, Z: torch.Tensor, layer: int) -> torch.Tensor:
    """Written states for one head of one layer: [N, HEAD_DIM, HEAD_DIM]."""
    if tag == "fixed":
        alpha = Z @ sd[f"alpha_heads.{layer}.weight"].double().T + sd[f"alpha_heads.{layer}.bias"].double()
        return torch.einsum("nk,kde->nde", alpha, basis[layer, :, HEAD].double()) * scale
    h = Z
    for i in (0, 2):  # Sequential(Linear, GELU, Linear, GELU)
        h = F.gelu(h @ sd[f"s1_trunk.{i}.weight"].double().T + sd[f"s1_trunk.{i}.bias"].double())
    n = HEAD_DIM * RANK
    u0 = (layer * NUM_HEADS * HEAD_DIM + HEAD * HEAD_DIM) * RANK
    v0 = layer * HEAD_DIM * RANK
    U = (h @ sd["s1_u_head.weight"][u0:u0 + n].double().T
         + sd["s1_u_head.bias"][u0:u0 + n].double()).reshape(len(Z), HEAD_DIM, RANK)
    V = (h @ sd["s1_v_head.weight"][v0:v0 + n].double().T
         + sd["s1_v_head.bias"][v0:v0 + n].double()).reshape(len(Z), HEAD_DIM, RANK)
    return torch.einsum("ndr,ner->nde", U, V) * scale


def analytic_fixed_reach(sd, basis) -> dict:
    """The fixed writer's reach is bounded by K by construction; verify it exactly.

    M_l(z) = sum_k alpha_{l,k}(z) B_{l,k} with alpha affine in z, so the centred
    reachable set is span{B_{l,k}} intersected with the image of the centred
    coefficients: its dimension is min(rank(centred alpha), rank(B_l)) <= K, for ANY
    number of sampled plans. Computed on the full 40-head state, N=1024.
    """
    Z = torch.randn(1024, LATENT, generator=torch.Generator().manual_seed(1))
    a_rank, b_rank = [], []
    for l in range(NUM_LAYERS):
        a = Z @ sd[f"alpha_heads.{l}.weight"].double().T  # bias drops out when centred
        a_rank.append(int(torch.linalg.matrix_rank(a, rtol=RTOL)))
        B = basis[l].double().reshape(K, -1)  # K x (40*64*64) = K x 163840
        b_rank.append(int(torch.linalg.matrix_rank(B @ B.T, rtol=RTOL)))
    return {
        "alpha_rank": {"min": min(a_rank), "max": max(a_rank)},
        "basis_rank": {"min": min(b_rank), "max": max(b_rank)},
        "n_plans": 1024,
        "ambient_full_state": NUM_HEADS * HEAD_DIM * HEAD_DIM,
    }


def main():
    out = {}
    for tag in ("fixed", "dynamic"):
        t0 = time.time()
        sd, basis, scale = load(tag)
        res = {"sweep": {}, "per_layer": {}}

        for N in N_SWEEP:
            dims = []
            for l in LAYERS:
                M = write_states(tag, sd, basis, scale, Z_ALL[:N], l).reshape(N, -1)
                sv = torch.linalg.svdvals(M - M.mean(0, keepdim=True))
                dims.append(int((sv > sv[0] * RTOL).sum()))
            res["sweep"][N] = dims
            print(f"{tag:8s} N={N:4d}  reach_dim per layer {dims}", flush=True)

        # Past the ambient dimension, on fewer layers: does the reach ever stop growing?
        res["saturate"] = {}
        for N in N_SATURATE:
            dims = []
            for l in SATURATE_LAYERS:
                M = write_states(tag, sd, basis, scale, Z_ALL[:N], l).reshape(N, -1)
                sv = torch.linalg.svdvals(M - M.mean(0, keepdim=True))
                dims.append(int((sv > sv[0] * RTOL).sum()))
            res["saturate"][N] = dims
            print(f"{tag:8s} N={N:4d}  reach_dim layers {SATURATE_LAYERS} {dims}", flush=True)

        Z = Z_ALL[:N_MAIN]
        for l in LAYERS:
            M = write_states(tag, sd, basis, scale, Z, l)
            Fl = M.reshape(N_MAIN, -1)
            C = Fl - Fl.mean(0, keepdim=True)
            sv = torch.linalg.svdvals(C)
            ov = []
            for i, j in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]:
                Qi = torch.linalg.svd(M[i])[0][:, :RANK]
                Qj = torch.linalg.svd(M[j])[0][:, :RANK]
                ov.append(float((torch.linalg.svdvals(Qi.T @ Qj) ** 2).sum() / RANK))
            res["per_layer"][l] = {
                "reach_dim": int((sv > sv[0] * RTOL).sum()),
                "reach_erank": round(erank(sv), 3),
                "vary_frac": round(float(C.norm() / Fl.norm()), 4),
                "mat_erank": round(sum(erank(torch.linalg.svdvals(M[i])) for i in range(8)) / 8, 3),
                "dir_overlap": round(sum(ov) / len(ov), 4),
            }
        keys = list(res["per_layer"][LAYERS[0]])
        res["mean"] = {k: round(sum(res["per_layer"][l][k] for l in LAYERS) / len(LAYERS), 3) for k in keys}
        if tag == "fixed":
            res["analytic"] = analytic_fixed_reach(sd, basis)
            print(f"{tag:8s} analytic:", json.dumps(res["analytic"]), flush=True)
        out[tag] = res
        print(f"{tag:8s} ({time.time() - t0:.0f}s) mean over layers {LAYERS}: "
              f"{json.dumps(res['mean'])}", flush=True)

    out["meta"] = {
        "pair": "laces-2.9B-{fixed,dynlowrank}-r32-global-birwkv768x8-b16-50k, step 50000",
        "single_factor": "s1_writer_type (same denoiser, batch size, corpus, step)",
        "K": K, "rank": RANK, "latent_dim": LATENT,
        "layers": LAYERS, "head": HEAD, "n_sweep": N_SWEEP, "n_main": N_MAIN,
        "n_saturate": N_SATURATE, "saturate_layers": SATURATE_LAYERS,
        "ambient_per_head": HEAD_DIM * HEAD_DIM,
        "dtype": "float64", "rank_rtol": RTOL,
    }
    dst = f"{ROOT}/results/paper_ledger/writer_expressible_range.json"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()

"""Probe: resample S2 from the MEAN of a sampled 16-block trajectory.

cond0 (S0 encode of prefix) -> z1 [16,32] -> text1
z_mean = z1.mean(dim=1)      -> z2 [16,32] -> text2   (resample from latent mean)
z_mean2 = z2.mean(dim=1)     -> z3 [16,32] -> text3   (one more iteration)

Same DDIM generator across z1/z2/z3 so any text change is due to the changed
condition (cond0 vs z_mean vs z_mean2), not the random seed.

Runs on the qz cluster (H100). Writes JSON to --output.
"""
from __future__ import annotations
import sys, os, math, json, time
from pathlib import Path
import torch

REPO = "/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/laces-reasoning"
sys.path.insert(0, REPO)

from laces_posttrain.native import load_native
from laces_posttrain.policy import PolicyConfig, ddim_sample

CKPT = os.path.join(REPO, "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000")
RWKV = os.path.join(REPO, "models/RWKV7-Goose-World3-2.9B-HF")  # real safetensors

STEPS = 32
MAX_TOKENS = 64

PROMPTS = [
    "Question: Why does the sky appear blue? Answer in one short sentence:",
    "The history of the lightbulb begins with",
    "A good recipe for pasta carbonara is:",
]


def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-8))


def main():
    t0 = time.time()
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/laces-reasoning/results/latent_mean_resample.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} steps={STEPS} max_tokens={MAX_TOKENS}", flush=True)
    native = load_native(CKPT, device, rwkv_path=RWKV, expected_step=30000,
                         expected_writer="dynlowrank", blend=0.7)
    print(f"loaded in {time.time()-t0:.1f}s horizon={native.horizon} chunk={native.chunk}", flush=True)
    cfg = PolicyConfig(steps=STEPS, eta=0.3, min_std=0.02, cfg_scale=2.0)

    results = []
    for p, prompt in enumerate(PROMPTS):
        prefix, cond0 = native.encode_prompt(prompt, seed=42)
        cond0 = cond0.to(native.device)

        gen = torch.Generator(device=native.device).manual_seed(0)
        z1 = ddim_sample(native.s2, cond0, native.horizon, cfg, generator=gen)
        z_mean = z1.mean(dim=1, keepdim=False).to(native.device)
        z2 = ddim_sample(native.s2, z_mean, native.horizon, cfg, generator=torch.Generator(device=native.device).manual_seed(0))
        z_mean2 = z2.mean(dim=1, keepdim=False).to(native.device)
        z3 = ddim_sample(native.s2, z_mean2, native.horizon, cfg, generator=torch.Generator(device=native.device).manual_seed(0))

        _, t1 = native.generate(prefix, z1.to(native.device), MAX_TOKENS)
        _, t2 = native.generate(prefix, z2.to(native.device), MAX_TOKENS)
        _, t3 = native.generate(prefix, z3.to(native.device), MAX_TOKENS)

        row = {
            "prompt": prompt,
            "text_from_cond0": t1,
            "text_from_mean_z1": t2,
            "text_from_mean_z2": t3,
            "cos_z1_z2": cos(z1, z2),
            "cos_z2_z3": cos(z2, z3),
            "cos_cond0_mean_z1": cos(cond0, z_mean),
            "cos_cond0_mean_z2": cos(cond0, z_mean2),
            "cond0_norm": float(cond0.norm()),
            "mean_z1_norm": float(z_mean.norm()),
            "mean_z2_norm": float(z_mean2.norm()),
            "z1_norm": float(z1.norm()),
            "z2_norm": float(z2.norm()),
            "z3_norm": float(z3.norm()),
        }
        results.append(row)
        print(f"\n=== prompt {p} ===", flush=True)
        print(f"[z1 from cond0]    : {t1!r}", flush=True)
        print(f"[z2 from mean(z1)] : {t2!r}", flush=True)
        print(f"[z3 from mean(z2)] : {t3!r}", flush=True)
        print(f"cos(z1,z2)={row['cos_z1_z2']:.4f} cos(z2,z3)={row['cos_z2_z3']:.4f} "
              f"cos(cond0,mean1)={row['cos_cond0_mean_z1']:.4f}", flush=True)
        print(f"norms: cond0={row['cond0_norm']:.3f} mean1={row['mean_z1_norm']:.3f} "
              f"mean2={row['mean_z2_norm']:.3f} z1={row['z1_norm']:.3f} z2={row['z2_norm']:.3f} z3={row['z3_norm']:.3f}", flush=True)

    payload = {"device": device, "steps": STEPS, "max_tokens": MAX_TOKENS,
               "elapsed_seconds": time.time() - t0, "results": results}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}", flush=True)
    native.close()


if __name__ == "__main__":
    main()

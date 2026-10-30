#!/usr/bin/env python3
"""GSM8K under TRAJECTORY-mode relay injection (chunked checkpoints).

The existing GSM8K harnesses cover the two ends of the relay axis:
  - eval_gsm8k_raw.py                : frozen backbone, no injection
  - eval_chain_reasoning_numeric.py  : single-z latent chain (p(z_next|z_cond))

Neither exercises the trajectory pipeline that traj32x16 checkpoints are trained
with. This script fills that gap: it samples a trajectory plan with
sample_trajectory_cfg and decodes chunk-by-chunk, injecting the predicted RWKV
state at each chunk boundary, then scores the generated continuation with the
SAME numeric extraction and matching as the other two harnesses so the three
numbers are directly comparable.

Answer extraction/matching and task loading are imported from
eval_chain_reasoning_numeric so all three GSM8K paths share one protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.diag_clean_z_trajectory import _set_pad_token  # noqa: E402
from scripts.eval.eval_chain_reasoning_numeric import _extract_num, _load_task, _match  # noqa: E402
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    generate_answer_trajectory,
)
from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    encode_prefix,
    sample_trajectory_cfg,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", default="gsm8k", choices=["gsm8k", "svamp"])
    p.add_argument("--n", type=int, default=1319)
    p.add_argument("--offset", type=int, default=0,
                   help="skip the first N items; lets a run be split across shards")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--trajectory_state_blend", type=float, default=None)
    p.add_argument("--trajectory_sampler", default=None, choices=[None, "rf_heun"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--progress_every", type=int, default=20)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16

    model, _rwkv, tokenizer, _ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model.eval()
    _set_pad_token(tokenizer)
    model._prefix_suffix_trajectory_s2 = True
    model._training_stage = 2
    model._trajectory_sampler = args.trajectory_sampler
    # cfg_drop_prob lives under `training` in the checkpoint config, not under the
    # architecture section that `model.config` exposes.
    model._cfg_drop_prob = float(cfg.training.get("cfg_drop_prob", 0.0) or 0.0)
    if args.trajectory_state_blend is not None:
        model.trajectory_state_blend = float(args.trajectory_state_blend)
        model.config.trajectory_state_blend = float(args.trajectory_state_blend)

    gen_args = SimpleNamespace(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
    )

    # _load_task has no offset argument, so over-read and slice; the global item
    # index is preserved so per-item seeding stays shard-independent.
    items = _load_task(args.task, args.offset + args.n)[args.offset:]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    samples_path = out.with_suffix(".samples.jsonl")

    correct = 0
    records = []
    with samples_path.open("w", encoding="utf-8") as sf:
        for qi, (question, gt) in enumerate(items):
            input_ids = tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
            attention_mask = torch.ones_like(input_ids)
            # Same per-item seeding scheme as the other relay harnesses, so the
            # sampled plan is reproducible per question index.
            torch.manual_seed(args.seed + 1000 * (args.offset + qi))
            z_prefix, prefix_cache, prefix_logits = encode_prefix(model, input_ids, attention_mask)
            z_traj = sample_trajectory_cfg(
                model, z_prefix, args.steps, args.cfg_scale, args.device, dtype,
                diffusion_sampler=str(getattr(model, "_gen_type", None) or "ddpm"),
            )
            generated, _ptok, _gtok = generate_answer_trajectory(
                model, tokenizer, input_ids, prefix_cache, prefix_logits, z_traj, gen_args,
            )
            pred = _extract_num(generated)
            ok = _match(pred, gt)
            correct += int(ok)
            rec = {"id": args.offset + qi, "question": question, "ground_truth": gt,
                   "generate": generated, "pred_num": pred, "ok": ok}
            records.append(rec)
            sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (qi + 1) % args.progress_every == 0:
                print(f"[{qi+1}/{len(items)}] acc={correct}/{qi+1} = {correct/(qi+1):.3f}", flush=True)

    summary = {
        "ckpt_dir": args.ckpt_dir, "task": args.task, "n": len(items),
        "offset": args.offset,
        "method": "trajectory_cfg", "sampling": "trajectory",
        "steps": args.steps, "cfg_scale": args.cfg_scale,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "accuracy": correct / max(1, len(items)), "correct": correct,
        "trajectory_horizon": int(getattr(model, "trajectory_horizon", 0)),
        "trajectory_chunk_size": int(getattr(model, "trajectory_chunk_size", 0)),
        "trajectory_state_blend": float(getattr(model, "trajectory_state_blend", 1.0)),
    }
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

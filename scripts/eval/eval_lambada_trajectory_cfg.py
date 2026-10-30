#!/usr/bin/env python3
"""Trajectory-mode LAMBADA eval: scores the target suffix under CHUNK-WISE
trajectory state injection (the mode the chunked checkpoints were trained in).

Same metric keys as eval_prefix_suffix_cfg_lambada.py (exact suffix / last token /
per-token) so numbers stay comparable, plus "mode": "trajectory".
Injection idiom mirrors run_cola_dlm_tasks_prefix_suffix_trajectory_cfg
(s1_mode=independent): inject chunk h's states into the prefix cache, refresh
logits by re-running the context under the injected cache, advance by chunk_size.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.eval.relay_utils import get_data_dir, list_npz_files, load_relay_model  # noqa: E402
from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    apply_generated_state, encode_prefix, sample_trajectory_cfg,
)
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    recompute_logits_from_injected_cache,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_samples", type=int, default=5153)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--scales", default="1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    return p.parse_args()


def _metrics(pred_tokens, target_tokens):
    exact = int(torch.equal(pred_tokens, target_tokens))
    last_ok = int(pred_tokens[-1].item() == target_tokens[-1].item())
    tok_ok = int((pred_tokens == target_tokens).sum().item())
    return exact, last_ok, tok_ok, int(len(target_tokens))


@torch.no_grad()
def score_raw(model, ids, mask, target_start):
    out = model.rwkv_model(input_ids=ids, attention_mask=mask.bool(), return_dict=True)
    pred_logits = out.logits[0, target_start - 1:ids.shape[1] - 1]
    pred_tokens = pred_logits.argmax(dim=-1)
    target_tokens = ids[0, target_start:ids.shape[1]]
    return _metrics(pred_tokens, target_tokens)


@torch.no_grad()
def score_traj(model, ids, mask, target_start, z_traj, device):
    """Greedy next-token predictions for ids[target_start:] under chunk-wise injection."""
    prefix_ids = ids[:, :target_start]
    prefix_mask = mask[:, :target_start]
    blend = float(model.config.get("trajectory_state_blend", getattr(model, "trajectory_state_blend", 1.0)))
    chunk_size = int(model.trajectory_chunk_size)

    _, past_kv, _ = encode_prefix(model, prefix_ids, prefix_mask)
    target = ids[0, target_start:]
    ctx_ids = list(ids[0, :target_start].tolist())

    preds = []
    pos = 0
    for h in range(z_traj.shape[1]):
        if pos >= target.numel():
            break
        states_h = model.predict_states(z_traj[:, h])
        past_kv = apply_generated_state(model, past_kv, states_h, blend)
        ctx = torch.tensor([ctx_ids], device=device, dtype=torch.long)
        past_kv, logits_batch = recompute_logits_from_injected_cache(
            model, ctx, torch.ones_like(ctx), past_kv
        )
        logits = logits_batch[0]
        for _ in range(chunk_size):
            if pos >= target.numel():
                break
            preds.append(int(logits.argmax(dim=-1).item()))
            out = model.rwkv_model(
                input_ids=target[pos:pos + 1].view(1, 1),
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
            ctx_ids.append(int(target[pos].item()))
            pos += 1
    if not preds:
        return 0, 0, 0, int(target.numel())
    pred_tokens = torch.tensor(preds, device=target.device, dtype=target.dtype)
    return _metrics(pred_tokens, target[:len(preds)])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = torch.bfloat16
    scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    model, _rwkv, _tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model_any = cast(Any, model)
    cfg_any = cast(Any, cfg)
    model_any._prefix_suffix_s2 = True
    model_any._prefix_suffix_trajectory_s2 = True
    model_any._training_stage = 2
    model_any._cfg_drop_prob = float(cfg_any.training.get("cfg_drop_prob", 0.0))
    data_dir = args.data_dir or get_data_dir("DIFFRWKV_DATA_DIR", "preprocessed_data/lambada/test")
    files = list_npz_files(data_dir, args.max_samples)
    results = {
        "ckpt_dir": args.ckpt_dir, "step": ckpt.get("step", -1), "n_files": len(files),
        "steps": args.steps, "scales": scales, "mode": "trajectory",
        "trajectory_horizon": int(getattr(model_any, "trajectory_horizon")),
        "trajectory_chunk_size": int(getattr(model_any, "trajectory_chunk_size")),
        "trajectory_state_blend": float(getattr(model_any, "trajectory_state_blend", 1.0)),
    }
    n = max(1, len(files))
    stats = {"last": 0, "tok": 0, "ntok": 0}
    correct_raw = 0
    t0 = time.time()
    for fp in files:
        d = np.load(fp)
        ids = torch.from_numpy(d["input_ids"].astype(np.int64)).unsqueeze(0).to(args.device)
        mask = torch.from_numpy(d["attention_mask"].astype(np.int64)).unsqueeze(0).to(args.device)
        exact, last_ok, tok_ok, n_tok = score_raw(model, ids, mask, int(d["target_start"][0]))
        correct_raw += exact
        stats["last"] += last_ok
        stats["tok"] += tok_ok
        stats["ntok"] += n_tok
    results["acc_raw_suffix"] = correct_raw / n
    results["correct_raw_suffix"] = correct_raw
    results["acc_raw_last"] = stats["last"] / n
    results["acc_raw_token"] = stats["tok"] / max(1, stats["ntok"])
    results["elapsed_raw_suffix_s"] = round(time.time() - t0, 1)
    print(f"raw: exact={results['acc_raw_suffix']:.4f} last={results['acc_raw_last']:.4f} "
          f"token={results['acc_raw_token']:.4f}", flush=True)

    for i, scale in enumerate(scales):
        correct = 0
        cstats = {"last": 0, "tok": 0, "ntok": 0}
        t0 = time.time()
        for j, fp in enumerate(files):
            d = np.load(fp)
            ids = torch.from_numpy(d["input_ids"].astype(np.int64)).unsqueeze(0).to(args.device)
            mask = torch.from_numpy(d["attention_mask"].astype(np.int64)).unsqueeze(0).to(args.device)
            target_start = int(d["target_start"][0])
            prefix_ids = ids[:, :target_start]
            prefix_mask = mask[:, :target_start]
            torch.manual_seed(args.seed + 100000 * (i + 1) + j)
            z_prefix, _pc, _pl = encode_prefix(model, prefix_ids, prefix_mask)
            z_traj = sample_trajectory_cfg(model, z_prefix, args.steps, scale, args.device, dtype)
            exact, last_ok, tok_ok, n_tok = score_traj(model, ids, mask, target_start, z_traj, args.device)
            correct += exact
            cstats["last"] += last_ok
            cstats["tok"] += tok_ok
            cstats["ntok"] += n_tok
        key = f"cfg{scale:g}"
        results[f"acc_{key}"] = correct / n
        results[f"correct_{key}"] = correct
        results[f"acc_{key}_last"] = cstats["last"] / n
        results[f"acc_{key}_token"] = cstats["tok"] / max(1, cstats["ntok"])
        results[f"elapsed_{key}_s"] = round(time.time() - t0, 1)
        print(f"{key}: exact={results[f'acc_{key}']:.4f} last={results[f'acc_{key}_last']:.4f} "
              f"token={results[f'acc_{key}_token']:.4f}", flush=True)
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
    output_json = json.dumps(results, indent=2)
    print(output_json)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_json)


if __name__ == "__main__":
    main()

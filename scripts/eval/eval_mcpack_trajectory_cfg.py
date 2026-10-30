#!/usr/bin/env python3
"""Trajectory-mode mcpack eval: NLL ranking of choices under CHUNK-WISE trajectory
state injection (sample_trajectory_cfg + predict_states + blend), which is the
mode the chunked checkpoints (trajectory_mode=True, prefix_suffix_trajectory_s2=True)
were trained in.

Same result keys as eval_prefix_suffix_cfg_multichoice.py (acc_raw_suffix /
acc_cfg{scale}) so numbers are directly comparable, plus "mode": "trajectory".
The state-injection idiom mirrors
run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.generate_answer_trajectory
(s1_mode=independent): inject chunk h's states, refresh logits by re-running the
context under the injected cache, decode chunk_size tokens, advance to chunk h+1.
Here the choice tokens are teacher-forced instead of sampled so they can be scored.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.eval.relay_utils import list_npz_files, load_relay_model  # noqa: E402
from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    apply_generated_state, encode_prefix, sample_trajectory_cfg,
)
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    recompute_logits_from_injected_cache,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--task_dir", required=True)
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--scales", default="1")
    p.add_argument("--length_normalize", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    return p.parse_args()


def span_nll(logits, ids, mask, choice_start, length_normalize):
    if choice_start <= 0:
        choice_start = 1
    pred_logits = logits[0, choice_start - 1:ids.shape[1] - 1].float()
    targets = ids[0, choice_start:ids.shape[1]]
    valid = mask[0, choice_start:ids.shape[1]].bool()
    if valid.sum() == 0:
        return float("inf")
    log_probs = F.log_softmax(pred_logits, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    nll = nll[valid]
    total = nll.sum().item()
    if length_normalize:
        return total / max(1, nll.numel())
    return total


@torch.no_grad()
def raw_score(model, ids, mask, choice_start, length_normalize):
    out = model.rwkv_model(input_ids=ids, attention_mask=mask.bool(), return_dict=True)
    return span_nll(out.logits, ids, mask, choice_start, length_normalize)


@torch.no_grad()
def traj_score(model, ids, mask, choice_start, z_traj, device, length_normalize):
    """Teacher-forced NLL of ids[choice_start:] under chunk-wise trajectory injection."""
    if choice_start <= 0:
        choice_start = 1
    prefix_ids = ids[:, :choice_start]
    prefix_mask = mask[:, :choice_start]
    blend = float(model.config.get("trajectory_state_blend", getattr(model, "trajectory_state_blend", 1.0)))
    chunk_size = int(model.trajectory_chunk_size)

    _, past_kv, _ = encode_prefix(model, prefix_ids, prefix_mask)
    suffix = ids[0, choice_start:]
    ctx_ids = list(ids[0, :choice_start].tolist())

    nll_sum, n_tok = 0.0, 0
    pos = 0
    for h in range(z_traj.shape[1]):
        if pos >= suffix.numel():
            break
        states_h = model.predict_states(z_traj[:, h])
        past_kv = apply_generated_state(model, past_kv, states_h, blend)
        ctx = torch.tensor([ctx_ids], device=device, dtype=torch.long)
        past_kv, logits_batch = recompute_logits_from_injected_cache(
            model, ctx, torch.ones_like(ctx), past_kv
        )
        logits = logits_batch[0]
        for _ in range(chunk_size):
            if pos >= suffix.numel():
                break
            log_probs = F.log_softmax(logits.float(), dim=-1)
            nll_sum += float(-log_probs[suffix[pos]])
            n_tok += 1
            out = model.rwkv_model(
                input_ids=suffix[pos:pos + 1].view(1, 1),
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
            ctx_ids.append(int(suffix[pos].item()))
            pos += 1
    if n_tok == 0:
        return float("inf")
    return nll_sum / n_tok if length_normalize else nll_sum


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
    files = list_npz_files(args.task_dir, args.max_samples)
    task_name = os.path.basename(args.task_dir.rstrip("/"))
    results = {
        "ckpt_dir": args.ckpt_dir, "step": ckpt.get("step", -1), "task": task_name,
        "task_dir": args.task_dir, "n_files": len(files), "steps": args.steps,
        "scales": scales, "mode": "trajectory",
        "trajectory_horizon": int(getattr(model_any, "trajectory_horizon")),
        "trajectory_chunk_size": int(getattr(model_any, "trajectory_chunk_size")),
        "trajectory_state_blend": float(getattr(model_any, "trajectory_state_blend", 1.0)),
    }
    modes = [None] + scales
    for mode_i, mode in enumerate(modes):
        correct = 0
        t0 = time.time()
        for file_i, fp in enumerate(files):
            d = np.load(fp)
            input_ids = d["input_ids"]
            attn_mask = d["attention_mask"]
            choice_start = d["choice_start"]
            label = int(d["label"][0])
            shared_z_traj = None
            if mode is not None:
                first_ids = torch.from_numpy(input_ids[0:1].astype(np.int64)).to(args.device)
                first_mask = torch.from_numpy(attn_mask[0:1].astype(np.int64)).to(args.device)
                first_cs = max(1, int(choice_start[0]))
                torch.manual_seed(args.seed + 100000 * mode_i + 1000 * file_i)
                z_prefix, _pc, _pl = encode_prefix(model, first_ids[:, :first_cs], first_mask[:, :first_cs])
                shared_z_traj = sample_trajectory_cfg(model, z_prefix, args.steps, mode, args.device, dtype)

            scores = []
            for j in range(input_ids.shape[0]):
                ids = torch.from_numpy(input_ids[j:j + 1].astype(np.int64)).to(args.device)
                mask = torch.from_numpy(attn_mask[j:j + 1].astype(np.int64)).to(args.device)
                cs = int(choice_start[j])
                if mode is None:
                    score = raw_score(model, ids, mask, cs, args.length_normalize)
                else:
                    score = traj_score(model, ids, mask, cs, shared_z_traj, args.device, args.length_normalize)
                scores.append(score)
            pred = int(np.argmin(scores))
            correct += int(pred == label)
        key = "raw_suffix" if mode is None else f"cfg{mode:g}"
        results[f"acc_{key}"] = correct / max(1, len(files))
        results[f"correct_{key}"] = correct
        results[f"elapsed_{key}_s"] = round(time.time() - t0, 1)
        print(f"{key}: acc={results[f'acc_{key}']:.4f} correct={correct}/{len(files)}", flush=True)
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

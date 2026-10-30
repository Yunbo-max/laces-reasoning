#!/usr/bin/env python3
"""Alignment experiment: why do the two RWKV-7 2.9B rows disagree?

The LACES paper carries two rows that are nominally the same 2.9B RWKV-7 World3
decoder but score differently on the same 6 option-bearing columns:

  * ``RWKV-7 World3``               (baseline group)  -- 6-col mean 60.75
    produced by scripts/eval/eval_generic_gen_match.py, i.e. HF ``model.generate``
  * ``Raw frozen decoder (no plan)`` (control group)  -- 6-col mean 59.75
    produced by scripts/eval/run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py
    ``--method raw``, i.e. the repo's hand-written greedy loop

Both claim greedy decoding on byte-identical prompts, yet 0/200 generations match.
This script isolates the cause with a controlled 2x2 over the two axes that
actually differ between the harnesses, holding the WEIGHT FILE fixed:

    axis 1  loader dtype : fp32 (what eval_generic_gen_match derives from
                           config.json's ``torch_dtype: float32``)
                     vs.  bf16 (what relay_utils.load_relay_model hardcodes)
    axis 2  decode impl  : ``hf``   = model.generate(do_sample=False, rep_pen)
                     vs.  ``loop`` = the repo's sample_next_token greedy loop

Arms (``--arm``):
    hf_fp32    reproduces the baseline row
    hf_bf16
    loop_bf16  reproduces the control row
    loop_fp32

Every arm writes the SAME jsonl schema the scorers consume, plus ``gen_ids``
(the raw token ids) so arms can be compared at the token level without the text
post-processing differences confounding it.

Extra modes:
    --mode verify_weights   compare the relay-loaded backbone against the HF
                            safetensors tensor-by-tensor (shape/allclose/maxdiff)
    --mode trace            per-step top-5 logits for one prompt under both
                            dtypes, to locate the first divergence

Nothing here mutates the existing harnesses; this is a separate read-only probe.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

HF_DIR = "/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/laces-reasoning/models/RWKV7-Goose-World3-2.9B-HF"
RELAY_CKPT = (
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/outputs_relay/"
    "traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000"
)
TASK_DIR = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/baseline/Cola-DLM/generate_task_data"
TASKS = ["piqa", "arc_easy", "arc_challenge", "winogrande", "sciq", "commonsenseqa"]

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


# ---------------------------------------------------------------- prompt build
# Imported from the shipped harness so the prompts are literally the same code
# path, not a re-implementation.
from scripts.eval.eval_generic_gen_match import build_prompt, get_ground_truth  # noqa: E402


# ------------------------------------------------------------- decode: repo loop
def _safe_apply_repetition_penalty(logits, generated_ids, penalty):
    """Verbatim copy of run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py L52-60."""
    if penalty == 1.0 or not generated_ids:
        return logits
    token_ids = torch.tensor(list(set(generated_ids)), device=logits.device, dtype=torch.long)
    token_ids = token_ids.clamp(0, logits.size(-1) - 1)
    token_scores = logits[token_ids]
    logits = logits.clone()
    logits[token_ids] = torch.where(token_scores < 0, token_scores * penalty, token_scores / penalty)
    return logits


@torch.no_grad()
def decode_loop(model, tokenizer, input_ids, max_new_tokens, rep_pen, eos_mode):
    """Verbatim semantics of generate_raw_answer (L291-313) of the relay harness."""
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    cache = out.past_key_values
    all_ids = list(input_ids[0].tolist())
    new_ids: list[int] = []
    logits = out.logits[0, -1]
    # The relay harness uses tokenizer.eos_token_id (65530 == "\n\n" for the RWKV
    # world tokenizer). HF generate instead uses generation_config.eos_token_id (0).
    eos_id = tokenizer.eos_token_id if eos_mode == "tokenizer" else 0
    for _ in range(max_new_tokens):
        logits = _safe_apply_repetition_penalty(logits.float(), all_ids, rep_pen)
        next_id = int(torch.argmax(logits).item())
        if eos_id is not None and next_id == eos_id:
            break
        new_ids.append(next_id)
        all_ids.append(next_id)
        out = model(
            input_ids=torch.tensor([[next_id]], device=input_ids.device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logits = out.logits[0, -1]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, int(input_ids.shape[1]), new_ids


# --------------------------------------------------------------- decode: HF
@torch.no_grad()
def decode_hf(model, tokenizer, input_ids, max_new_tokens, rep_pen):
    """Verbatim semantics of eval_generic_gen_match.generate_one (L150-180)."""
    prompt_tokens = int(input_ids.shape[1])
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=rep_pen,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen_ids = out[0, prompt_tokens:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    for special in getattr(tokenizer, "all_special_tokens", ()):
        text = text.replace(special, " ")
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text, prompt_tokens, gen_ids


# ------------------------------------------------------------------- loading
def load_backbone(dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_DIR, trust_remote_code=True, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            HF_DIR, trust_remote_code=True, torch_dtype=dtype, local_files_only=True
        )
        .to(device)
        .eval()
    )
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = getattr(tok, "eos_token", "") or tok.unk_token
    got = {p.dtype for p in model.parameters()}
    print(f"[load] requested={dtype} actual_param_dtypes={got}", flush=True)
    return model, tok


# ------------------------------------------------------- mode: verify_weights
def mode_verify_weights(device):
    """Is the 'frozen decoder' inside the relay checkpoint bit-identical to the
    released HF weights?  Compares the relay-loaded backbone against the
    safetensors shards tensor-by-tensor."""
    from safetensors.torch import load_file

    ckpt = torch.load(os.path.join(RELAY_CKPT, "model.pt"), map_location="cpu", weights_only=False)
    ts = ckpt["trainable_state"]
    print(f"[ckpt] step={ckpt.get('step')} n_trainable_tensors={len(ts)}")
    backboneish = [k for k in ts if k.startswith(("rwkv_model.", "backbone.", "decoder.", "rwkv."))]
    print(f"[ckpt] tensors under a backbone prefix (rwkv_model./backbone./decoder./rwkv.): {len(backboneish)}")
    import collections

    print(f"[ckpt] top-level namespaces: {dict(collections.Counter(k.split('.')[0] for k in ts))}")

    idx = json.load(open(os.path.join(HF_DIR, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    ref: dict[str, torch.Tensor] = {}
    for s in shards:
        ref.update(load_file(os.path.join(HF_DIR, s)))
    print(f"[hf] {len(ref)} tensors from {len(shards)} shards; dtypes={set(str(v.dtype) for v in ref.values())}")

    from scripts.eval.relay_utils import load_relay_model

    model, _rwkv, _tok, _c, _cfg = load_relay_model(RELAY_CKPT, device)
    live = {k: v for k, v in model.rwkv_model.state_dict().items()}
    print(f"[relay] model.rwkv_model has {len(live)} tensors")

    n_ok = n_diff = n_shape = 0
    worst = 0.0
    worst_key = None
    missing = []
    for k, rv in ref.items():
        lv = live.get(k)
        if lv is None:
            missing.append(k)
            continue
        if tuple(lv.shape) != tuple(rv.shape):
            n_shape += 1
            print(f"  SHAPE {k}: relay={tuple(lv.shape)} hf={tuple(rv.shape)}")
            continue
        a = lv.detach().to("cpu", torch.float32)
        b = rv.to(torch.float32)
        md = (a - b).abs().max().item()
        if md > worst:
            worst, worst_key = md, k
        if md == 0.0:
            n_ok += 1
        else:
            n_diff += 1
            if n_diff <= 20:
                print(f"  DIFF  {k}: maxdiff={md:.3e} allclose={torch.allclose(a, b)}")
    print()
    print(f"[RESULT] bit-identical tensors : {n_ok}/{len(ref)}")
    print(f"[RESULT] differing tensors     : {n_diff}")
    print(f"[RESULT] shape mismatches      : {n_shape}")
    print(f"[RESULT] present in HF, absent in relay backbone: {len(missing)} {missing[:5]}")
    print(f"[RESULT] global maxdiff        : {worst:.6e}  (worst key: {worst_key})")
    print(f"[RESULT] VERDICT: {'DECODER IS BIT-IDENTICAL' if (n_diff == 0 and n_shape == 0 and not missing) else 'DECODER DIFFERS'}")
    # dtype of the live backbone matters as much as the values
    import collections as _c2

    print(f"[RESULT] relay backbone param dtypes: {dict(_c2.Counter(str(v.dtype) for v in live.values()))}")


# ---------------------------------------------------------------- mode: trace
def mode_trace(device, rep_pen, max_new_tokens):
    """Per-step top-5 under fp32 vs bf16 on one prompt; find the first divergence."""
    item = json.loads(open(os.path.join(TASK_DIR, "arc_easy.jsonl")).readline())
    prompt = build_prompt("arc_easy", item)
    print(f"[trace] prompt = {prompt!r}\n")

    per_dtype = {}
    for name in ("fp32", "bf16"):
        model, tok = load_backbone(DTYPES[name], device)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        out = model(input_ids=ids, use_cache=True, return_dict=True)
        cache = out.past_key_values
        all_ids = list(ids[0].tolist())
        logits = out.logits[0, -1]
        steps = []
        for _ in range(max_new_tokens):
            lg = _safe_apply_repetition_penalty(logits.float(), all_ids, rep_pen)
            top = torch.topk(lg, 5)
            nid = int(torch.argmax(lg).item())
            steps.append(
                {
                    "tok": nid,
                    "piece": tok.decode([nid]),
                    "top5": [(int(i), round(float(v), 4)) for v, i in zip(top.values, top.indices)],
                }
            )
            all_ids.append(nid)
            out = model(
                input_ids=torch.tensor([[nid]], device=device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            logits = out.logits[0, -1]
        per_dtype[name] = steps
        del model
        torch.cuda.empty_cache()

    first = None
    for i, (a, b) in enumerate(zip(per_dtype["fp32"], per_dtype["bf16"])):
        same = a["tok"] == b["tok"]
        if first is None and not same:
            first = i
        g1, g2 = a["top5"][0][1], b["top5"][0][1]
        m = "" if same else "   <-- DIVERGE"
        print(f"step {i:2d} fp32 tok={a['tok']:6d} {a['piece']!r:14s} top1={g1:9.4f} | "
              f"bf16 tok={b['tok']:6d} {b['piece']!r:14s} top1={g2:9.4f}{m}")
        if not same:
            print(f"         fp32 top5={a['top5']}")
            print(f"         bf16 top5={b['top5']}")
    print(f"\n[trace] first divergent step: {first}")


# ----------------------------------------------------------------- mode: run
def mode_run(args, device):
    impl, dt_name = args.arm.split("_")
    model, tok = load_backbone(DTYPES[dt_name], device)
    out_root = Path(args.output_dir) / args.arm
    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "arm": args.arm,
        "decode_impl": impl,
        "loader_dtype": dt_name,
        "weights": HF_DIR,
        "eos_mode": args.eos_mode,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "max_samples": args.max_samples,
        "greedy": True,
        "tasks": {},
    }
    for task in TASKS:
        src = Path(TASK_DIR) / f"{task}.jsonl"
        if not src.exists():
            print(f"[{task}] SKIP missing {src}", flush=True)
            continue
        dst = out_root / f"{task}.jsonl"
        t0 = time.time()
        n = 0
        with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as fo:
            for i, line in enumerate(f):
                if i >= args.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt = build_prompt(task, item)
                ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                if impl == "hf":
                    gen, ptok, gen_ids = decode_hf(
                        model, tok, ids, args.max_new_tokens, args.repetition_penalty
                    )
                else:
                    gen, ptok, gen_ids = decode_loop(
                        model, tok, ids, args.max_new_tokens, args.repetition_penalty, args.eos_mode
                    )
                rec = dict(item)
                rec.update(
                    {
                        "id": item.get("id", i),
                        "prompt": prompt,
                        "generate": gen,
                        "ground_truth": get_ground_truth(item),
                        "choices": item.get("choices", []),
                        "prompt_tokens": ptok,
                        "generated_tokens": len(gen_ids),
                        "gen_ids": gen_ids,
                        "method": "raw",
                        "cfg_scale": 0.0,
                        "steps": 0,
                        "arm": args.arm,
                    }
                )
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 50 == 0:
                    print(f"[{args.arm}/{task}] {n} ({time.time()-t0:.0f}s)", flush=True)
        summary["tasks"][task] = {"n": n, "elapsed_s": round(time.time() - t0, 1)}
        print(f"[{args.arm}/{task}] DONE n={n} {time.time()-t0:.0f}s -> {dst}", flush=True)
    (out_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["run", "verify_weights", "trace"], default="run")
    p.add_argument("--arm", choices=["hf_fp32", "hf_bf16", "loop_bf16", "loop_fp32"], default="hf_bf16")
    p.add_argument("--eos_mode", choices=["tokenizer", "hf"], default="tokenizer",
                   help="tokenizer=65530 ('\\n\\n', what the relay loop uses); hf=0 (generation_config)")
    p.add_argument("--output_dir", default=str(REPO / "results" / "benchmarks" / "harness_align"))
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device)
    if args.mode == "verify_weights":
        mode_verify_weights(device)
    elif args.mode == "trace":
        mode_trace(device, args.repetition_penalty, args.max_new_tokens)
    else:
        mode_run(args, device)


if __name__ == "__main__":
    main()

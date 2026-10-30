#!/usr/bin/env python3
"""Raw GSM8K baseline (no latent chain, no state injection): prompt the frozen
RWKV backbone directly, generate an answer and extract the number.

Companion to eval_chain_reasoning_numeric.py (which iterates the latent chain):
uses the SAME task loading (_load_task), answer extraction (_extract_num) and
matching (_match), and the SAME generation defaults (max_new_tokens=200,
temperature 0.6, top_k 30, top_p 0.9, repetition_penalty 1.2) so raw vs
latent-chain numbers are directly comparable.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.diag_loop1_common import build_model  # noqa: E402
from scripts.eval.eval_chain_reasoning_numeric import (  # noqa: E402
    _load_task, _extract_num, _match,
)
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    generate_raw_answer,
)


@torch.no_grad()
def run(ckpt_dir, device, task, n, max_new_tokens, temperature, top_k, top_p,
        repetition_penalty, output, seed):
    torch.manual_seed(seed)
    model, tokenizer, _dtype, _pad_id = build_model(ckpt_dir, device)
    model.eval()

    gen_args = SimpleNamespace(max_new_tokens=max_new_tokens, temperature=temperature,
                               top_k=top_k, top_p=top_p,
                               repetition_penalty=repetition_penalty)

    items = _load_task(task, n)
    records = []
    correct = 0
    for qi, (question, gt) in enumerate(items):
        ids = tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        gen, _ptok, _gtok = generate_raw_answer(model, tokenizer, ids, gen_args)
        ok = _match(_extract_num(gen), gt)
        correct += int(ok)
        records.append({"id": qi, "question": question, "ground_truth": gt,
                        "generate": gen, "pred_num": _extract_num(gen), "ok": ok})
        if (qi + 1) % 20 == 0:
            print(f"[{qi+1}/{len(items)}] acc={correct}/{qi+1}", flush=True)

    acc = correct / max(1, len(items))
    summary = {
        "ckpt_dir": ckpt_dir, "task": task, "n": len(items), "method": "raw",
        "temperature": temperature, "max_new_tokens": max_new_tokens,
        "accuracy": acc, "correct": correct,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    with out.with_suffix(".samples.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "svamp"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="outputs_eval/gsm8k_raw.json")
    a = ap.parse_args()
    run(a.ckpt_dir, a.device, a.task, a.n, a.max_new_tokens, a.temperature,
        a.top_k, a.top_p, a.repetition_penalty, a.output, a.seed)


if __name__ == "__main__":
    main()

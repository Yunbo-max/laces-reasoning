#!/usr/bin/env python3
"""Build generate-then-match task data from the tokenised mc-pack npz shards.

Why this exists: the npz shards store, per example, one row per candidate choice,
each row being `prompt + choice_j` with **its own `choice_start[j]`**. Slicing every
row with `choice_start[0]` (an easy mistake) mis-aligns rows whose prompt is longer,
which silently corrupts the choices and depresses every model's score. This script
does it per row, strips the RWKV end-of-text marker, drops degenerate rows, and
verifies that the ground-truth text is present in the choice list.

Usage:
  python scripts/preprocess/build_gen_task_data_from_npz.py \
      --npz_root <preprocessed_data> --out_dir <generate_task_data> \
      --tasks piqa arc_easy arc_challenge winogrande sciq commonsenseqa --max_samples 200
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

EOT = "<|rwkv_tokenizer_end_of_text|>"
DEFAULT_TOKENIZER = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/models/"
                     "RWKV7-Goose-World3-2.9B-HF")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace(EOT, " ")).strip()


def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)


def build_task(tok, npz_root: Path, task: str, out_dir: Path, max_samples: int, limit: int | None):
    files = sorted((npz_root / task / "validation").glob("*.npz"))
    if limit:
        files = files[:limit]
    written = skipped = 0
    out_path = out_dir / f"{task}.jsonl"
    with out_path.open("w", encoding="utf-8") as fo:
        for fp in files:
            if written >= max_samples:
                break
            d = np.load(fp)
            ids = d["input_ids"]
            cs = d["choice_start"]
            label = int(d["label"][0])
            if not (0 <= label < len(ids)):
                skipped += 1
                continue
            # per-row split: each candidate row carries its own choice_start
            prompts, choices = [], []
            for j in range(len(ids)):
                start = int(cs[j]) if np.ndim(cs) else int(cs)
                start = max(0, min(start, len(ids[j]) - 1))
                prompts.append(clean(tok.decode(ids[j][:start].astype("int64"))))
                choices.append(clean(tok.decode(ids[j][start:].astype("int64"))))
            prompt = prompts[label] if prompts[label] else (prompts[0] if prompts else "")
            # every choice must share the same prompt; keep the majority text
            prompt = max(set(prompts), key=prompts.count) if prompts else ""
            if not prompt or len(choices) < 2 or any(not c for c in choices):
                skipped += 1
                continue
            if len(set(choices)) != len(choices):
                skipped += 1
                continue
            fo.write(json.dumps({"id": written, "prompt": prompt,
                                 "ground_truth": choices[label], "choices": choices},
                                ensure_ascii=False) + "\n")
            written += 1
    print(f"  {task:16s} wrote {written:4d}  skipped {skipped:4d}  -> {out_path}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tok_path", default=DEFAULT_TOKENIZER)
    ap.add_argument("--tasks", nargs="+",
                    default=["piqa", "arc_easy", "arc_challenge", "winogrande", "sciq", "commonsenseqa"])
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--limit_files", type=int, default=None)
    a = ap.parse_args()

    tok = load_tokenizer(a.tok_path)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        build_task(tok, Path(a.npz_root), task, out_dir, a.max_samples, a.limit_files)


if __name__ == "__main__":
    main()

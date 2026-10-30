#!/usr/bin/env python3
"""Winogrande task data from the original HF dataset.

The tokenised mc-pack shards store winogrande as a cloze pair (`choice_start = 0`),
so no prompt can be recovered from them; we rebuild the generative form from the
source dataset instead: "Question: <sentence with ____>\nAnswer:" with the two
candidate fillers as choices.
"""
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="winogrande_xl")
    ap.add_argument("--max_samples", type=int, default=200)
    a = ap.parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache")
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", a.config, split="validation")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fo:
        for r in ds:
            if n >= a.max_samples:
                break
            sentence = r["sentence"].replace("_", "____")
            choices = [r["option1"], r["option2"]]
            gt = choices[0] if r["answer"] == "1" else choices[1]
            fo.write(json.dumps({"id": n, "prompt": f"Question: {sentence}\nAnswer:",
                                 "ground_truth": gt, "choices": choices}, ensure_ascii=False) + "\n")
            n += 1
    print(f"winogrande: wrote {n} -> {out}")


if __name__ == "__main__":
    main()

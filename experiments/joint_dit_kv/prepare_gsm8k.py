"""Convert official GSM8K *training* examples into single-prompt reasoning blocks.

Source: openai/gsm8k, main, train only. A separate SHA-hash 5% dev
subset is made from the official training set, leaving the official test
sealed. Reasoning targets are NEVER supplied as inference conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def split_gsm8k_answer(raw: str):
    if "####" not in raw:
        raise ValueError("GSM8K training record lacks final #### answer separator")
    rationale, answer = raw.rsplit("####", 1)
    answer = answer.strip().replace(",", "")
    if not answer:
        raise ValueError("Empty final answer")
    # Inline <<operation=result>> annotations are training artifacts, not language.
    rationale = re.sub(r"<<[^>]*>>", "", rationale)
    steps = [line.strip() for line in rationale.splitlines() if line.strip()]
    steps.append("Final answer: " + answer)
    return steps, answer


def split_for_id(example_id: str, holdout_percent: int) -> str:
    bucket = int(hashlib.sha256(example_id.encode()).hexdigest()[:8], 16) % 10000
    return "dev" if bucket < holdout_percent * 100 else "train"


def read_source(args):
    if args.local_jsonl:
        with open(args.local_jsonl) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        from datasets import load_dataset
        yield from load_dataset("openai/gsm8k", "main", split="train", streaming=True)


def convert(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"read": 0, "train": 0, "dev": 0, "skipped": 0,
             "min_steps": 999999, "max_steps": 0}
    trainpath = out / "train.jsonl"
    devpath = out / "dev.jsonl"
    seen = set()
    with trainpath.open("w") as tr, devpath.open("w") as dv:
        for raw in read_source(args):
            stats["read"] += 1
            question = str(raw.get("question", "")).strip()
            if not question:
                stats["skipped"] += 1
                continue
            identifier = str(raw.get("id") or hashlib.sha256(question.encode()).hexdigest())
            if identifier in seen:
                stats["skipped"] += 1
                continue
            seen.add(identifier)
            try:
                steps, gold = split_gsm8k_answer(str(raw["answer"]))
            except (ValueError, KeyError):
                stats["skipped"] += 1
                continue
            row = {"id": identifier, "question": question, "steps": steps, "answer": gold}
            subset = split_for_id(identifier, args.holdout_percent)
            (dv if subset == "dev" else tr).write(json.dumps(row, ensure_ascii=False)+"\n")
            stats[subset] += 1
            stats["min_steps"] = min(stats["min_steps"], len(steps))
            stats["max_steps"] = max(stats["max_steps"], len(steps))
            if args.limit and stats["read"] >= args.limit:
                break
    (out / "manifest.json").write_text(json.dumps({
        "dataset": "openai/gsm8k:main:train",
        "source": args.local_jsonl or "HuggingFace train (streaming)",
        "official_test_used": False,
        "split": "SHA256 of question or provided stable id",
        "holdout_percent": args.holdout_percent,
        "steps_policy": "Each rationale line forms one reasoning step; final answer appended",
        "stats": stats,
    }, indent=2))
    return stats


def make_toy(path: str, train: int = 256, dev: int = 64):
    """Pipeline smoke only; never a benchmark improvement claim."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(417)
    rows = []
    for i in range(train + dev):
        a, b, c = rng.randint(2, 60), rng.randint(3, 80), rng.randint(2, 9)
        v = (a+b)*c
        steps = [f"First add {a} and {b} to get {a+b}.",
                 f"Then multiply {a+b} by {c} to get {v}.",
                 f"Final answer: {v}"]
        rows.append({
            "id": "smoke-" + str(i),
            "question": f"What is ({a} + {b}) * {c}? Give only the number.",
            "steps": steps, "answer": str(v),
        })
    for name, group in (("train", rows[:train]), ("dev", rows[train:])):
        with (out / (name + ".jsonl")).open("w") as f:
            for row in group:
                f.write(json.dumps(row) + "\n")
    (out / "manifest.json").write_text(json.dumps(
        {"origin": "synthetic_smoke_only", "train": train, "dev": dev}, indent=2))
    return {"train": train, "dev": dev}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--make-toy", action="store_true")
    p.add_argument("--train-toy", type=int, default=256)
    p.add_argument("--dev-toy", type=int, default=64)
    p.add_argument("--local-jsonl", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--holdout-percent", type=int, default=5)
    a = p.parse_args()
    if not 1 <= a.holdout_percent <= 40:
        p.error("holdout-percent must be between 1 and 40")
    result = (make_toy(a.out, a.train_toy, a.dev_toy)
              if a.make_toy else convert(a))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

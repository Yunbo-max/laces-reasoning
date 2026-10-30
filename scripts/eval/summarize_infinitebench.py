#!/usr/bin/env python3
"""Build the InfiniteBench per-model x per-dataset table, annotated with truncation.

A bare score is not reportable for this suite: InfiniteBench prompts run ~30k-170k tokens
against windows of 2k-32k, so almost every row is truncated and a low score can mean either
"the model cannot do this" or "we deleted the evidence". Every cell therefore carries the
fraction of the document the model actually saw, and the degeneracy census is printed
alongside so a degenerate-generation zero is never mistaken for a capability measurement.

Usage:
  python3 scripts/eval/summarize_infinitebench.py [--dir results/benchmarks/longctx_fix]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

DATASETS = ["longbook_qa_eng", "longbook_choice_eng", "kv_retrieval", "code_debug", "math_find"]

# Best score obtainable by answering the SAME letter on every row, measured on the evaluated
# 50-row slice with this file's own scorer (see the ledger). These are not the 0.250 uniform
# chance rate: the gold letters are not uniformly distributed in the slice we score. code_debug's
# gold is "A" on 21/50 rows, so a model that only ever says "A" -- which a degenerate repetition
# loop effectively does -- scores 0.420 without reading a single token of the context.
# A score at or below these numbers is not evidence of long-context ability.
#
# IMPORTANT: 0.420 is a property of the `[:50]` SLICE, not of code_debug. P(gold="A") by block of
# 50 across the 394 rows runs 0.420, 0.300, 0.260, 0.300, 0.240, 0.240, 0.240, 0.182 -- the head
# of the file is the most "A"-heavy block in it (overall P(A)=0.274; p=0.018 for >=21 of 50 by
# chance). So this null is the right one for the numbers in this directory, and a 50-row
# code_debug score is not a quantity worth reporting at all. longbook_choice_eng is unaffected
# (0.300 on the first 50 vs 0.297 over all 229).
TRIVIAL_BASELINE = {"longbook_choice_eng": 0.300, "code_debug": 0.420}
# Number of options, for the tasks where a uniform guess is also worth reporting.
UNIFORM_CHANCE = {"longbook_choice_eng": 0.250, "code_debug": 0.250}


def distinct_ngram_ratio(ids: list[int], n: int = 4) -> float:
    """Fraction of distinct n-grams in the generation; a repetition loop drives this to ~0.

    The empty/near-monotoken census below catches only the crudest failures. rwkv7-world3
    passes it 250/250 while emitting "A:\\nThe answer is:\\nA:\\nThe answer is:..." -- a
    phrase-level loop, with many distinct tokens but no content. That output still collects a
    non-zero choice score, because the scorer finds a letter in it. This metric is what
    separates the two.
    """
    if len(ids) < n + 1:
        return 1.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)

# Declared-vs-real window, from the modeling-source audit in logs/qz/INFINITEBENCH_FIX_JOBS.md.
ARCH = {
    "qwen2.5-3b": "positional (32768)", "qwen2.5-1.5b": "positional (131072)",
    "llama-3.2-3b": "positional (131072)", "phi-2": "positional (2048)",
    "gemma-2-2b": "positional (8192)", "gla-1.3b": "recurrent (2048 inert)",
    "rwkv7-world3": "recurrent (2048 inert)", "mamba-2.8b": "recurrent (no field)",
    "global-dit-50k": "recurrent relay (Global)", "champ-dynamic-30k": "recurrent relay (Chunked)",
}


def degeneracy(rows: list[dict]) -> tuple[int, int]:
    """(empty, near-monotoken) counts -- the fingerprint of a generation failure."""
    empty = sum(1 for r in rows if not r.get("prediction", "").strip())
    mono = 0
    for r in rows:
        ids = r.get("prediction_token_ids") or []
        if ids and Counter(ids).most_common(1)[0][1] / len(ids) >= 0.9:
            mono += 1
    return empty, mono


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="results/benchmarks/longctx_fix")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*__infinitebench.json")))
    if not files:
        raise SystemExit(f"no InfiniteBench results in {args.dir}")

    print(f"=== scores ({args.dir}) ===")
    header = f"{'model':20s} " + " ".join(f"{d[:12]:>12s}" for d in DATASETS) + f" {'mean':>7s}  budget"
    print(header)
    print("-" * len(header))
    for path in files:
        name = os.path.basename(path).split("__")[0]
        blob = json.load(open(path))
        agg = blob.get("aggregates", {})
        cfg = blob.get("config", {})
        cells, got = [], []
        for ds in DATASETS:
            entry = agg.get(ds)
            if isinstance(entry, dict) and "score" in entry:
                cells.append(f"{entry['score']:12.3f}")
                got.append(entry["score"])
            else:
                cells.append(f"{'--':>12s}")
        mean = f"{sum(got) / len(got):7.3f}" if got else f"{'--':>7s}"
        budget = cfg.get("effective_input_budget", cfg.get("max_input_tokens", "?"))
        print(f"{name:20s} " + " ".join(cells) + f" {mean}  {budget}")

    print()
    print("=== how much of each document the model actually saw ===")
    print(f"{'model':20s} {'dataset':22s} {'seen':>8s} {'of full':>9s} {'frac':>6s}  trunc  arch")
    for path in files:
        name = os.path.basename(path).split("__")[0]
        blob = json.load(open(path))
        for ds in DATASETS:
            entry = blob.get("aggregates", {}).get(ds)
            if not isinstance(entry, dict):
                continue
            seen = entry.get("mean_input_tokens_seen")
            full = entry.get("mean_prompt_tokens_untruncated")
            frac = entry.get("seen_fraction")
            n_trunc = entry.get("rows_truncated")
            # Pre-fix result files predate these fields; print "--" rather than crashing so
            # the old and new runs can be summarised with the same tool.
            seen_s = "--" if seen is None else str(round(seen))
            full_s = "--" if full is None else str(round(full))
            frac_s = "--" if frac is None else f"{frac:.3f}"
            trunc_s = "--" if n_trunc is None else f"{n_trunc}/{entry.get('n')}"
            print(f"{name:20s} {ds:22s} {seen_s:>8} {full_s:>9} {frac_s:>6}  "
                  f"{trunc_s}  {ARCH.get(name, '?')}")

    print()
    print("=== trivial-baseline check on the two 4-option tasks ===")
    print("best-const = score for answering the same letter on all 50 rows, i.e. zero reading.")
    print("A '<=' verdict means the cell is NOT evidence of long-context ability.")
    print(f"{'model':20s} {'task':20s} {'score':>7s} {'best-const':>11s} {'uniform':>8s}  verdict")
    for path in files:
        name = os.path.basename(path).split("__")[0]
        agg = json.load(open(path)).get("aggregates", {})
        for ds, trivial in TRIVIAL_BASELINE.items():
            entry = agg.get(ds)
            if not isinstance(entry, dict) or "score" not in entry:
                continue
            score = entry["score"]
            verdict = "above trivial" if score > trivial else "<= TRIVIAL"
            print(f"  {name:18s} {ds:20s} {score:7.3f} {trivial:11.3f} "
                  f"{UNIFORM_CHANCE.get(ds, 0.25):8.3f}  {verdict}")

    print()
    print("=== phrase-level repetition (mean distinct-4gram ratio; <=0.30 rows = loops) ===")
    print("A loop can still score on choice tasks, so read this next to the scores above.")
    for path in files:
        name = os.path.basename(path).split("__")[0]
        by_ds: dict[str, list[dict]] = {}
        for row in json.load(open(path)).get("rows", []):
            by_ds.setdefault(row["dataset"], []).append(row)
        parts, loops, total = [], 0, 0
        for ds in DATASETS:
            rows = by_ds.get(ds)
            if not rows:
                continue
            ratios = [distinct_ngram_ratio(r.get("prediction_token_ids") or []) for r in rows]
            parts.append(f"{ds[:11]}={sum(ratios) / len(ratios):.2f}")
            loops += sum(1 for x in ratios if x <= 0.30)
            total += len(rows)
        flag = "  <-- MOSTLY LOOPING" if total and loops > total * 0.5 else ""
        print(f"  {name:20s} " + "; ".join(parts) + f"  loops={loops}/{total}{flag}")

    print()
    print("=== exact hits vs partial credit (questions actually answered) ===")
    print("A nonzero free-form mean does NOT imply one answered question: the free-form branch")
    print("falls back to token-overlap F1, so a rambling generation collects fractional credit.")
    print("champ-dynamic-30k's longbook_qa_eng was 0/50 exact with a nonzero mean -- all overlap.")
    print("Caveat: some exact hits are themselves question-echo false positives (see the ledger),")
    print("so these counts are an UPPER bound on capability, not a corrected score.")
    print(f"{'model':20s} " + " ".join(f"{d[:11]:>11s}" for d in DATASETS) + f" {'total':>7s}")
    for path in files:
        name = os.path.basename(path).split("__")[0]
        by_ds: dict[str, list[dict]] = {}
        for row in json.load(open(path)).get("rows", []):
            by_ds.setdefault(row["dataset"], []).append(row)
        cells, total = [], 0
        for ds in DATASETS:
            rows = by_ds.get(ds)
            if not rows:
                cells.append(f"{'--':>11s}")
                continue
            exact = sum(1 for r in rows if r.get("score", 0.0) >= 1.0)
            total += exact
            cells.append(f"{exact:>7d}/{len(rows)}")
        print(f"{name:20s} " + " ".join(cells) + f" {total:7d}")

    print()
    print("=== degeneracy census (empty / near-monotoken of n) ===")
    print("A model with a large count here did not answer badly -- it failed to generate.")
    for path in files:
        name = os.path.basename(path).split("__")[0]
        blob = json.load(open(path))
        by: dict[str, list[dict]] = {}
        for row in blob.get("rows", []):
            by.setdefault(row["dataset"], []).append(row)
        parts = []
        for ds in DATASETS:
            rows = by.get(ds)
            if not rows:
                continue
            empty, mono = degeneracy(rows)
            flag = " DEGENERATE" if (empty + mono) > len(rows) * 0.5 else ""
            parts.append(f"{ds}={empty}/{mono} of {len(rows)}{flag}")
        print(f"  {name:20s} " + "; ".join(parts))


if __name__ == "__main__":
    main()

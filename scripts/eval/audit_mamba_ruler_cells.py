#!/usr/bin/env python3
"""Row-level audit of mamba-2.8b's RULER cells against PRE-REGISTERED criteria.

Written before the 16384 lanes finished, so the thresholds cannot have been chosen to flatter the
result. See ``results/paper_ledger/LONGCTX_BLANKROWS_LEDGER.md`` section 7a, "PRE-REGISTERED audit
criteria".

Why this exists rather than an eyeball of the log: the lane logs print no sample predictions
(``grep -c 'pred|sample|gold'`` on them is 0), so a 0.0000 cell cannot be distinguished from a broken
one without reading ``rows`` from the merged JSON.

Deliberate design choices:

* It reports the **unambiguous empty-string count** and the **distinct-prediction count** as plain
  numbers and never emits a "degenerate: yes/no" verdict. Degeneracy detectors on this suite are
  threshold-dependent by ~55x, and for ``niah_single_1`` the gold IS a bare number, so a loose
  detector flags *correct* answers as broken.
* It prints verbatim samples, because criterion (c) is that a human reads real predictions.
* It checks the budget/decode stamps from the TOP LEVEL of the JSON (``decode_path``,
  ``decode_is_trustworthy``), not from ``config`` -- looking only in ``config`` previously produced a
  false "unstamped file" alarm.
* ``aggregates`` entries are dicts in some files and bare floats in others across this corpus, so
  every read of them is shape-guarded.

Usage:
  python3 scripts/eval/audit_mamba_ruler_cells.py \
      --file results/benchmarks/longctx_fix/mamba-2.8b__ruler.json [--samples 5]
"""
from __future__ import annotations

import argparse
import collections
import json

# Pre-established reference shapes at the three 16384 cells, from files that already existed before
# mamba's 16k lanes finished. Criterion (d): mamba should look like the other recurrent models.
REFERENCE_16K = {
    "gla-1.3b":      {"16384_niah_single_1": 0.0000, "16384_vt": 0.0000, "16384_cwe": 0.0000},
    "rwkv7-world3":  {"16384_niah_single_1": 0.9900, "16384_vt": 0.0000, "16384_cwe": 0.0580},
    "llama-3.2-3b":  {"16384_niah_single_1": 1.0000, "16384_vt": 0.9620, "16384_cwe": 0.4185},
}
# Prediction 1: vt is the cell that truncates for everyone (~17.0k tokens vs a 16384 cap).
EXPECT_TRUNCATED = {"16384_vt"}


def score_of(entry) -> float:
    """aggregates values are sometimes dicts, sometimes bare floats. Never assume."""
    if isinstance(entry, dict):
        return entry.get("score", float("nan"))
    return float(entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    blob = json.load(open(args.file))
    cfg = blob.get("config", {})

    print("=" * 78)
    print(f"AUDIT {args.file}")
    print("=" * 78)
    # Stamps live at the top level, not in config.
    print(f"decode_path           = {blob.get('decode_path')!r}")
    print(f"decode_is_trustworthy = {blob.get('decode_is_trustworthy')!r}")
    print(f"effective_input_budget= {cfg.get('effective_input_budget')!r}")
    print(f"truncation_reason     = {cfg.get('truncation_reason')!r}")
    print(f"positional_window     = {cfg.get('positional_window')!r}")
    print(f"max_input_tokens(CLI) = {cfg.get('max_input_tokens')!r}   <- raw arg, NOT the clamp")
    print(f"seed                  = {cfg.get('seed')!r}")
    print()

    rows_by_cell: dict[str, list[dict]] = collections.defaultdict(list)
    for row in blob.get("rows", []):
        rows_by_cell[row.get("dataset")].append(row)

    aggregates = blob.get("aggregates", {})
    print(f"cells={len(aggregates)}  rows={len(blob.get('rows', []))}")
    print()

    header = (f"{'cell':26s} {'score':>7s} {'n':>5s} {'trunc':>6s} {'seen':>7s} "
              f"{'empty':>6s} {'distinct':>9s}")
    print(header)
    print("-" * len(header))

    failures: list[str] = []
    for cell in sorted(aggregates):
        entry = aggregates[cell]
        rows = rows_by_cell.get(cell, [])
        score = score_of(entry)
        n = entry.get("n") if isinstance(entry, dict) else len(rows)
        trunc = entry.get("rows_truncated") if isinstance(entry, dict) else None
        seen = entry.get("seen_fraction") if isinstance(entry, dict) else None

        preds = [(r.get("prediction") or "") for r in rows]
        empty = sum(1 for p in preds if not p.strip())
        distinct = len(set(preds))

        seen_s = f"{seen:.4f}" if isinstance(seen, (int, float)) else "  --  "
        print(f"{cell:26s} {score:7.4f} {n if n is not None else -1:5d} "
              f"{trunc if trunc is not None else -1:6d} {seen_s:>7s} {empty:6d} {distinct:9d}")

        # Criterion (a): completeness, and truncation only where predicted.
        if n != 200:
            failures.append(f"{cell}: n={n}, expected 200")
        if trunc:
            if cell in EXPECT_TRUNCATED:
                if not (isinstance(seen, float) and seen >= 0.95):
                    failures.append(f"{cell}: truncated with seen_fraction {seen}, expected >=0.95")
            else:
                failures.append(
                    f"{cell}: {trunc} rows truncated but truncation was NOT predicted here -- "
                    f"re-open the tokenizer-dependence question before reporting")
        elif cell in EXPECT_TRUNCATED:
            failures.append(
                f"{cell}: 0 rows truncated, but the ledger predicts this is the universally "
                f"truncated cell. Prediction 1 is falsified; the vt claim must be re-derived.")

    print()
    print("Reference shapes at the 16k cells (pre-established, criterion (d)):")
    for model, cells in REFERENCE_16K.items():
        got = "  ".join(f"{k.replace('16384_','')}={v:.4f}" for k, v in cells.items())
        print(f"  {model:16s} {got}")
    print()

    # Criterion (c): verbatim samples. A 0.0000 cell is unreadable without these.
    for cell in sorted(aggregates):
        rows = rows_by_cell.get(cell, [])
        if not rows:
            continue
        print(f"--- {cell}: first {args.samples} rows verbatim ---")
        for row in rows[:args.samples]:
            pred = (row.get("prediction") or "")
            # The gold key on RULER rows is `answers` (a list). It is NOT `gold` and NOT `answer`:
            # asking for those returns None for every row, which looks like missing data rather than
            # a wrong key. This is the same class of trap as the bAbILong JSONL using `generate`
            # instead of `prediction`, which once made me report "100/100 blank" on a clean cell.
            # Verified schema: ['answers','arm','dataset','dropped_tokens','id','input_tokens',
            # 'mean_logp','prediction','prediction_token_ids','prompt_tokens_untruncated','score',
            # 'suite'].
            if "answers" not in row:
                raise SystemExit(
                    f"REFUSING TO AUDIT: row in {cell} has no 'answers' key (keys: "
                    f"{sorted(row)}). The gold field was renamed; fix this script rather than "
                    f"reading an audit whose gold column is silently empty."
                )
            gold = row["answers"]
            print(f"    score={row.get('score')}  in_tok={row.get('input_tokens')} "
                  f"untrunc={row.get('prompt_tokens_untruncated')} "
                  f"dropped={row.get('dropped_tokens')}")
            print(f"      gold = {gold!r}")
            print(f"      pred = {pred[:200]!r}")
        print()

    print("=" * 78)
    if failures:
        print("PRE-REGISTERED CRITERIA FAILED -- affected cells stay `--`:")
        for f in failures:
            print(f"  * {f}")
        raise SystemExit(1)
    print("All pre-registered mechanical criteria PASSED.")
    print("Criterion (c) is NOT automatable: a human must still read the samples above and confirm")
    print("that any 0.0000 cell contains wrong-but-plausible answers rather than empty strings.")


if __name__ == "__main__":
    main()

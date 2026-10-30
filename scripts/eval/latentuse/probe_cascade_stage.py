#!/usr/bin/env python3
"""Which cascade level scored each VALID latent-arm row?

The rows carry no ``stage`` field (33-key schema, verified), so the only way to
answer this is to re-run the SHIPPED cascade with instrumentation.  Gate first:
the instrumented copy must reproduce stored ``correct``/``invalid`` on every row
with ZERO disagreement, otherwise its stage labels describe a different scorer.

Read-only.  Writes nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from scripts.eval.latentuse.g2_frozen_run_five_arms import (  # noqa: E402
    LETTERS,
    _f1,
    letter_of,
    normalize_text,
    strip_answer_lead,
    strip_label,
    truncate_at_newline,
)

LATENT_ARMS = ("A2", "A3", "A4")


def match_choice_staged(gen: str, choices: list[str]) -> tuple[Optional[int], Optional[str]]:
    """Mirror of shipped ``match_choice``, additionally naming the sub-stage."""
    if not choices:
        return None, None
    ng = normalize_text(gen)
    if not ng:
        return None, None
    norms = [normalize_text(c) for c in choices]
    for i, c in enumerate(norms):
        if c and c == ng:
            return i, "exact"
    best, best_len = None, 0
    for i, c in enumerate(norms):
        if c and (ng.startswith(c) or c.startswith(ng)):
            L = min(len(c), len(ng))
            if L > best_len:
                best, best_len = i, L
    if best is not None:
        return best, "prefix"
    best, best_len = None, 0
    for i, c in enumerate(norms):
        if c and c in ng and len(c) > best_len:
            best, best_len = i, len(c)
    if best is not None:
        return best, "containment"
    scored = sorted(((_f1(gen, c), i) for i, c in enumerate(choices)), reverse=True)
    if scored and scored[0][0] >= 0.5:
        return scored[0][1], "f1"
    return None, None


def score_row_staged(
    gen: str, gold: str, choices: list[str], letter_first: bool
) -> tuple[bool, bool, Optional[int], str]:
    """Mirror of shipped ``score_row``; returns the extra ``stage`` label."""
    g = truncate_at_newline(gen)
    if not g.strip():
        return False, True, None, "empty"

    pred: Optional[int] = None
    stage = "none"
    if letter_first:
        bare = letter_of(g)
        if bare is not None and (len(strip_label(g)) == 0 or len(g.strip()) <= 2) \
                and LETTERS.index(bare) < len(choices):
            pred = LETTERS.index(bare)
            stage = "letter_first"
    if pred is None:
        pred, sub = match_choice_staged(g, choices)
        if pred is not None:
            stage = f"raw/{sub}"
    if pred is None:
        pred, sub = match_choice_staged(strip_label(g), choices)
        if pred is not None:
            stage = f"strip_label/{sub}"
    if pred is None:
        pred, sub = match_choice_staged(strip_answer_lead(strip_label(g)), choices)
        if pred is not None:
            stage = f"strip_answer_lead/{sub}"
    if pred is None:
        lt = letter_of(g)
        if lt is not None and LETTERS.index(lt) < len(choices):
            pred = LETTERS.index(lt)
            stage = "letter_fallback"

    gold_idx: Optional[int] = None
    ngold = normalize_text(gold)
    for i, c in enumerate(choices):
        if normalize_text(c) == ngold:
            gold_idx = i
            break
    if gold_idx is None:
        gold_idx, _ = match_choice_staged(gold, choices)
    if gold_idx is None:
        raise ValueError(f"gold {gold!r} does not resolve to any choice {choices!r}")
    invalid = pred is None
    return (pred == gold_idx and not invalid), invalid, pred, stage


def gen_of(row: dict[str, Any]) -> str:
    for k in ("prediction", "gen", "generation", "text"):
        if k in row:
            return row[k]
    raise KeyError(f"no generation field among {sorted(row)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--letter-first", action="store_true",
                    help="must match /meta/args/letter_first (measured False in 10/10)")
    ap.add_argument("--inject-gate-failure", action="store_true",
                    help="positive control: corrupt one row so the gate MUST fire. "
                         "NOTE: appending to gen is INERT -- truncate_at_newline() runs "
                         "first and these generations contain newlines, so anything after "
                         "the first newline never reaches the cascade.  The injection must "
                         "therefore REPLACE the pre-newline span.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    print(f"files: {len(files)}  (denominator printed before any verdict)")
    if not files:
        print("FAIL: 0 files -> a gate over 0 rows cannot fail")
        return 2

    disagree = 0
    n_rows = 0
    stages: Counter[str] = Counter()
    latent_valid: list[tuple[str, str, int, str, bool]] = []
    injected = False

    for f in files:
        base = os.path.basename(f).replace(".jsonl", "")
        task, _, arm = base.rpartition("__")
        with open(f) as fh:
            for ln, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                gen = gen_of(row)
                if args.inject_gate_failure and not injected and not row["correct"]:
                    # Replace the span the cascade actually sees (everything before the
                    # first newline) with the gold string, so the row MUST flip to
                    # correct=True and disagree with the stored False.
                    # TWO ways this control was inert before, both fixed here:
                    #   (a) appending after the first newline -- truncate_at_newline()
                    #       discards it, so the cascade never saw the injection;
                    #   (b) injecting into a row whose stored correct was ALREADY True
                    #       -- forcing it to gold changes nothing, so the gate cannot fire.
                    # Hence the `not row["correct"]` guard.
                    gen = row["gold"]
                    injected = True
                    print(f"  INJECT at {base}:{ln} -> pre-newline span replaced with gold "
                          f"{row['gold']!r} (stored correct={row['correct']}, must flip to True)")
                correct, invalid, _pred, stage = score_row_staged(
                    gen, row["gold"], row["choices"], args.letter_first
                )
                n_rows += 1
                if bool(correct) != bool(row["correct"]) or bool(invalid) != bool(row["invalid"]):
                    disagree += 1
                    if disagree <= 3:
                        print(f"  DISAGREE {base}:{ln} stored=({row['correct']},{row['invalid']}) "
                              f"mine=({correct},{invalid}) stage={stage}")
                stages[stage] += 1
                if arm in LATENT_ARMS and not invalid:
                    latent_valid.append((task, arm, row["example_id"], stage, bool(correct)))

    print(f"rows: {n_rows}")
    print(f"GATE  disagreements with stored correct/invalid: {disagree}")
    if disagree:
        print("GATE FAILED -> stage labels below describe a DIFFERENT scorer; do not quote them")
        return 1
    print("GATE PASSED (0/%d) -- and the positive control below proves it CAN fail" % n_rows)

    print("\n=== stage histogram, all arms ===")
    for s, n in stages.most_common():
        print(f"  {s:28s} {n}")

    print(f"\n=== VALID latent-arm rows (A2/A3/A4): n={len(latent_valid)} ===")
    for task, arm, eid, stage, correct in sorted(latent_valid):
        print(f"  {arm} {task:16s} id={eid:<4d} stage={stage:28s} correct={correct}")
    print("\nstage counts among valid latent rows:")
    for s, n in Counter(x[3] for x in latent_valid).most_common():
        print(f"  {s:28s} {n}/{len(latent_valid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

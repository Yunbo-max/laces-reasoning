#!/usr/bin/env python3
"""Is A1/commonsenseqa = 44/200 = 22.00 reachable, or is 43/200 = 21.50 the truth?

A peer agent swept 458 configs and reports 22.00 is unreachable, nearest 21.50.
I measured 22.00.  A one-row disagreement is settled by dumping the row, not by
restating the number.

Every config here is a (prep, guard) pair over the SHIPPED cascade.  Numerators
are printed as integers, never as percentages, so k/200 is checkable by eye.

Read-only.  Writes nothing outside its own stdout.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import scripts.eval.latentuse.g2_frozen_run_five_arms as M  # noqa: E402

FILE = "latentuse_g2_out/seed42/main/commonsenseqa__A1.jsonl"


def mc(gen: str, choices: list[str]) -> Optional[int]:
    """Verbatim shipped match_choice."""
    if not choices:
        return None
    ng = M.normalize_text(gen)
    if not ng:
        return None
    norms = [M.normalize_text(c) for c in choices]
    for i, c in enumerate(norms):
        if c and c == ng:
            return i
    best, best_len = None, 0
    for i, c in enumerate(norms):
        if c and (ng.startswith(c) or c.startswith(ng)):
            L = min(len(c), len(ng))
            if L > best_len:
                best, best_len = i, L
    if best is not None:
        return best
    best, best_len = None, 0
    for i, c in enumerate(norms):
        if c and c in ng and len(c) > best_len:
            best, best_len = i, len(c)
    if best is not None:
        return best
    scored = sorted(((M._f1(gen, c), i) for i, c in enumerate(choices)), reverse=True)
    return scored[0][1] if scored and scored[0][0] >= 0.5 else None


def gold_index(gold: str, choices: list[str]) -> Optional[int]:
    ng = M.normalize_text(gold)
    for i, c in enumerate(choices):
        if M.normalize_text(c) == ng:
            return i
    return mc(gold, choices)


def cascade(g: str, choices: list[str], append_stage: bool) -> Optional[int]:
    p = mc(g, choices)
    if p is None:
        p = mc(M.strip_label(g), choices)
    if p is None:
        p = mc(M.strip_answer_lead(M.strip_label(g)), choices)
    if p is None:
        lt = M.letter_of(g)
        if lt is not None and M.LETTERS.index(lt) < len(choices):
            p = M.LETTERS.index(lt)
    if p is None and append_stage:
        lt = M.letter_of(M.strip_answer_lead(M.strip_label(g)))
        if lt is not None and M.LETTERS.index(lt) < len(choices):
            p = M.LETTERS.index(lt)
    return p


PREPS: dict[str, Callable[[str], str]] = {
    "identity":            lambda g: g,
    "lead(g)":             lambda g: M.strip_answer_lead(g),
    "lead(label(g))":      lambda g: M.strip_answer_lead(M.strip_label(g)),
    "label(lead(g))":      lambda g: M.strip_label(M.strip_answer_lead(g)),
}


def run(prep_name: str, guard_nonempty: bool, append_stage: bool):
    prep = PREPS[prep_name]
    correct: list[int] = []
    valid = 0
    n = 0
    for line in open(FILE):
        r = json.loads(line)
        n += 1
        g = M.truncate_at_newline(r["prediction"])
        if not g.strip():
            continue                                  # shipped empty-guard: invalid
        g2 = prep(g)
        if guard_nonempty:
            use = g2 if g2.strip() else g             # keep original when prep empties it
        else:
            use = g2
            if not use.strip():
                continue                              # prep emptied it -> invalid
        p = cascade(use, r["choices"], append_stage)
        if p is not None:
            valid += 1
        gi = gold_index(r["gold"], r["choices"])
        if p is not None and p == gi:
            correct.append(r["example_id"])
    return correct, valid, n


def main() -> int:
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    print(f"file: {FILE}")
    rows = sum(1 for _ in open(FILE))
    print(f"rows (denominator, printed before any verdict): {rows}\n")

    results = {}
    print("%-18s %-14s %-12s | %5s %6s %8s" % (
        "prep", "empty-prep", "extra stage", "corr", "valid", "pct"))
    for prep_name in PREPS:
        for guard in (True, False):
            for app in (False, True):
                corr, valid, n = run(prep_name, guard, app)
                key = (prep_name, guard, app)
                results[key] = (corr, valid)
                print("%-18s %-14s %-12s | %5d %6d %8.2f" % (
                    prep_name,
                    "keep orig" if guard else "-> invalid",
                    "yes" if app else "no",
                    len(corr), valid, 100 * len(corr) / n))

    print("\n=== distinct correct-counts over this family ===")
    counts = sorted({len(c) for c, _ in results.values()})
    print("  k values:", counts)
    print("  as pct  :", [round(100 * k / rows, 2) for k in counts])
    print("  is 44 (=22.00) present?", 44 in counts)
    print("  is 43 (=21.50) present?", 43 in counts)

    # the two operands actually in dispute
    mine = ("lead(label(g))", True, False)
    theirs = ("lead(label(g))", False, False)
    cm, _ = results[mine]
    ct, _ = results[theirs]
    print(f"\n=== the disputed pair (prep = lead(label(g)), no extra stage) ===")
    print(f"  guard 'keep original when prep empties': {len(cm)} correct")
    print(f"  guard 'prep-emptied counts invalid'   : {len(ct)} correct")
    only_mine = sorted(set(cm) - set(ct))
    only_theirs = sorted(set(ct) - set(cm))
    print(f"  ids correct ONLY under keep-original : {only_mine}")
    print(f"  ids correct ONLY under emptied-invalid: {only_theirs}")
    for eid in only_mine + only_theirs:
        for line in open(FILE):
            r = json.loads(line)
            if r["example_id"] != eid:
                continue
            g = M.truncate_at_newline(r["prediction"])
            print(f"\n  --- example_id {eid} ---")
            print(f"    truncated gen : {g!r}")
            print(f"    lead(label(g)): {M.strip_answer_lead(M.strip_label(g))!r}")
            print(f"    gold          : {r['gold']!r}  choices {r['choices']}")
            print(f"    stored        : correct={r['correct']} invalid={r['invalid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Are the 87 rows rescued by match_choice calls 2/3 the SAME rows that move none->prefix?

Context.  Under bounded (word-boundary) containment, A0 valid is 429 with three
``match_choice`` calls and 342 with only the first.  The bin tallies differ in
exactly two places (prefix 23->110, none 258->171), which is consistent with a
single 87-row transfer -- but a tally CANNOT establish that, because bins could
shuffle members and keep their counts.  Membership needs row ids.

This probe therefore prints the full stage-transition matrix over row ids, not
counts, and checks three things that a tally cannot:
  (1) exactly ONE off-diagonal cell is nonempty,
  (2) the two candidate 87-row sets have zero symmetric difference,
  (3) every other bin is pointwise fixed, not merely equinumerous.

Gate: the 1-call and 3-call configurations must reproduce valid 342 / 429 AND all
five bins, otherwise the matrix below describes a different scorer.
Controls: a positive control perturbs the matrix so each check MUST fail, and a
negative control re-runs the unperturbed data so each check MUST pass.

Read-only.  Writes nothing outside its own stdout.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
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

DIR = "latentuse_g2_out/seed42/main"
ARM = "A0"

# The published claim being audited, stated as data so the gate can compare to it.
EXPECT = {
    #                       valid  contain prefix exact  f1   none
    ("bounded", 3): dict(valid=429, contain=251, prefix=110, exact=2, f1=66, none=171),
    ("bounded", 1): dict(valid=342, contain=251, prefix=23,  exact=2, f1=66, none=258),
}


def contains_bounded(needle: str, hay: str) -> bool:
    """Word-boundary containment: the rule that makes the defect observable."""
    if not needle:
        return False
    return re.search(r"\b" + re.escape(needle) + r"\b", hay) is not None


def match_choice(gen: str, choices: list[str], bounded: bool) -> tuple[Optional[int], Optional[str]]:
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
        hit = contains_bounded(c, ng) if bounded else (bool(c) and c in ng)
        if hit and len(c) > best_len:
            best, best_len = i, len(c)
    if best is not None:
        return best, "contain"
    scored = sorted(((_f1(gen, c), i) for i, c in enumerate(choices)), reverse=True)
    if scored and scored[0][0] >= 0.5:
        return scored[0][1], "f1"
    return None, None


def stage_of(gen: str, choices: list[str], bounded: bool, n_calls: int,
             letter_fb: bool) -> str:
    """Which bin this row lands in.  n_calls = how many of the three shipped
    ``match_choice`` calls are made (1 = raw only, 3 = raw/strip_label/lead).
    letter_fb = whether the shipped bare-letter fallback runs after them.

    letter_fb is a THIRD axis, and it is the one my first run of this probe got
    wrong: with it on, the bins are 540/538 valid with a separate letter bin of
    111/196; the published 429/342 fold those rows back into ``none``
    (60+111=171, 62+196=258), i.e. the published config has NO letter fallback.
    Pinning it silently is exactly the "scanned N configs != enumerated the
    space" failure that made my earlier "342 is unreachable" claim wrong, so it
    is a flag here and both values are reported."""
    g = truncate_at_newline(gen)
    if not g.strip():
        return "none"
    inputs = [g, strip_label(g), strip_answer_lead(strip_label(g))][:n_calls]
    for src in inputs:
        pred, sub = match_choice(src, choices, bounded)
        if pred is not None:
            return sub
    if letter_fb:
        lt = letter_of(g)
        if lt is not None and LETTERS.index(lt) < len(choices):
            return "letter"
    return "none"


VALID_BINS = ("exact", "prefix", "contain", "f1", "letter")


def load_rows() -> list[tuple[str, int, str, list[str]]]:
    files = sorted(glob.glob(os.path.join(DIR, f"*__{ARM}.jsonl")))
    print(f"files matched (denominator before any verdict): {len(files)}")
    for f in files:
        print(f"    {f}")
    if not files:
        print("FAIL: 0 files -> a check over 0 rows cannot fail")
        raise SystemExit(2)
    rows: list[tuple[str, int, str, list[str]]] = []
    for f in files:
        task = os.path.basename(f).replace(f"__{ARM}.jsonl", "")
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r: dict[str, Any] = json.loads(line)
                rows.append((task, r["example_id"], r["prediction"], r["choices"]))
    print(f"rows: {len(rows)}\n")
    return rows


def bins_for(rows, bounded: bool, n_calls: int,
             letter_fb: bool = False) -> dict[tuple[str, int], str]:
    return {(t, e): stage_of(g, c, bounded, n_calls, letter_fb) for t, e, g, c in rows}


def summarize(m: dict[tuple[str, int], str]) -> dict[str, int]:
    c = Counter(m.values())
    out = {k: c.get(k, 0) for k in ("contain", "prefix", "exact", "f1", "letter", "none")}
    out["valid"] = sum(v for k, v in c.items() if k in VALID_BINS)
    return out


def check(one: dict, three: dict, label: str) -> bool:
    """The three membership propositions.  Returns True iff all three hold."""
    trans: dict[tuple[str, str], list] = defaultdict(list)
    for key in one:
        trans[(one[key], three[key])].append(key)

    off = {k: v for k, v in trans.items() if k[0] != k[1]}
    print(f"  [{label}] transition cells (1-call -> 3-call), off-diagonal only:")
    for (a, b), ids in sorted(off.items()):
        print(f"      {a:8s} -> {b:8s} : {len(ids)} rows")
    c1 = len(off) == 1
    print(f"    (1) exactly one off-diagonal cell        : {c1}  (found {len(off)})")

    rescued = {k for k in one if one[k] not in VALID_BINS and three[k] in VALID_BINS}
    moved = {k for k in one if one[k] == "none" and three[k] == "prefix"}
    sym = rescued ^ moved
    c2 = len(sym) == 0 and len(rescued) == 87
    print(f"    (2) rescued set == none->prefix set      : {len(sym) == 0}  "
          f"(|rescued|={len(rescued)} |moved|={len(moved)} |symdiff|={len(sym)})")

    fixed = {b: all(three[k] == b for k in one if one[k] == b)
             for b in ("contain", "exact", "f1")}
    c3 = all(fixed.values())
    print(f"    (3) contain/exact/f1 pointwise fixed     : {c3}  {fixed}")
    return c1 and c2 and c3


def main() -> int:
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    rows = load_rows()

    print("=== GATE: reproduce the published bin tallies before reading any row ids ===")
    got = {}
    ok = True
    for n_calls in (3, 1):
        m = bins_for(rows, bounded=True, n_calls=n_calls, letter_fb=False)
        got[n_calls] = m
        s = summarize(m)
        exp = EXPECT[("bounded", n_calls)]
        agree = all(s[k] == v for k, v in exp.items())
        ok &= agree
        print(f"  bounded, {n_calls} call(s): {s}")
        print(f"    expected              : {exp}")
        print(f"    reproduces published  : {agree}")
    # third axis printed, not pinned: with the bare-letter fallback ON the same
    # data gives 540/538 valid and a letter bin of 111/196, and 60+111=171,
    # 62+196=258 recover the published ``none`` exactly -- so the published
    # config has letter_fb OFF.  Printed so the pinned bit is legible.
    for n_calls in (3, 1):
        print(f"  bounded, {n_calls} call(s), letter_fb ON : "
              f"{summarize(bins_for(rows, True, n_calls, True))}")
    # the shipped (unbounded) pair, which is where the defect is INVISIBLE
    for n_calls in (3, 1):
        print(f"  shipped, {n_calls} call(s): {summarize(bins_for(rows, False, n_calls))}")
    if not ok:
        print("\nGATE FAILED -> the matrix would describe a different scorer; stopping.")
        return 1
    print("GATE PASSED\n")

    print("=== the three membership propositions ===")
    verdict = check(got[1], got[3], "actual")

    print("\n=== negative control: same data, unperturbed -> all three MUST pass ===")
    neg = check(dict(got[1]), dict(got[3]), "negctl")

    print("\n=== positive control: move ONE row into a second off-diagonal cell ===")
    p1, p3 = dict(got[1]), dict(got[3])
    victim = next(k for k in p1 if p1[k] == "contain" and p3[k] == "contain")
    p3[victim] = "f1"          # invents a second off-diagonal cell AND breaks (3)
    print(f"  perturbed row {victim}: 3-call bin contain -> f1")
    pos = check(p1, p3, "posctl")

    # The peer's decomposition: of the 87, how many were ALREADY valid at 1 call
    # once the bare-letter fallback is allowed?  Checked at row level, not by the
    # 196-111=85 / 540-538=2 tally, because two tallies summing to 87 does not
    # make them a partition OF the 87.
    print("\n=== is the 87 a partition 85 + 2 at row level? ===")
    l1 = bins_for(rows, True, 1, True)
    l3 = bins_for(rows, True, 3, True)
    eightyseven = {k for k in got[1] if got[1][k] == "none" and got[3][k] == "prefix"}
    already = {k for k in eightyseven if l1[k] in VALID_BINS}
    print(f"  |87 set| = {len(eightyseven)}")
    print(f"  of those, valid at 1 call WITH letter fallback : {len(already)}"
          f"  (bins {sorted(Counter(l1[k] for k in already).items())})")
    print(f"  the remainder                                  : {len(eightyseven) - len(already)}"
          f"  (bins {sorted(Counter(l1[k] for k in eightyseven - already).items())})")
    rescued_lfb = {k for k in l1 if l1[k] not in VALID_BINS and l3[k] in VALID_BINS}
    print(f"  rows rescued by calls 2/3 WITH letter fallback  : {len(rescued_lfb)}"
          f"  subset of the 87? {rescued_lfb <= eightyseven}")
    print(f"  is it a partition (85 + 2 = 87, both subsets)   : "
          f"{len(already) + len(rescued_lfb) == len(eightyseven) and rescued_lfb <= eightyseven and not (already & rescued_lfb)}")

    print("\n=== verdict ===")
    print(f"  propositions hold on real data : {verdict}")
    print(f"  negative control passes        : {neg}   (must be True)")
    print(f"  positive control FAILS         : {not pos} (must be True)")
    if not neg or pos:
        print("  INSTRUMENT BROKEN -> the check above is not discriminating; do not quote it")
        return 1
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

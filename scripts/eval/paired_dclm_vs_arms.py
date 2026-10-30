#!/usr/bin/env python3
"""Paired (McNemar) comparison of DCLM-1B against our arms on the SAME capability rows.

WHY NOT TWO ACCURACIES. Comparing DCLM's 71.6 SQuAD against an arm's 35.5 by eyeballing two
marginal numbers -- or two marginal confidence intervals -- says nothing about whether the two
models differ on the rows actually evaluated. This repo already recorded the consequence: a
"three wins" claim built from marginal comparisons collapsed once the same rows were paired,
and a model that looked 7-for-7 dominant was beating its opponent only on the rows carrying a
scoring artefact. The only admissible test here is paired on identical rows.

ROW IDENTITY IS CHECKED, NOT ASSUMED. Equal `n` is not equal rows: two arms can both have
n=200 and share no ordering. Rows are aligned on `id` AND the ground truth is required to
match for that id; any id whose ground truth disagrees is reported and dropped rather than
silently paired. If too few ids survive, the comparison is refused instead of reported.

SCORING IS THE PUBLISHED SCORING. Per-row correctness is computed by calling the very helpers
score_gen_match.score_file uses (truncate / match_choice / strip_label / strip_answer_lead /
letter_of / normalize_text / f1), in the same order, so the paired table always reconciles
with the aggregate table. Re-implementing the rules here would risk a test that measures a
different metric than the one published.

TEST CHOICE depends on what the metric is, which differs by task:
  * choice tasks + lambada -> per-row score is binary, so exact two-sided McNemar (binomial on
    the discordant pairs). No normal approximation: discordant counts here are often < 25.
  * squad -> per-row score is a continuous F1, where McNemar does not apply. Reported as a
    Wilcoxon signed-rank test over the per-row F1 differences (ties dropped), plus the paired
    mean difference. The binary EM column is additionally given an exact McNemar.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
sys.path.insert(0, str(REPO / "scripts/eval"))

import score_gen_match as S  # noqa: E402

CHOICE_TASKS = ["mmlu", "obqa", "race", "story_cloze", "hellaswag", "siqa",
                "arc_easy", "arc_challenge", "winogrande", "sciq", "commonsenseqa", "piqa"]
FREEFORM = ["squad", "lambada"]


def row_scores(path: str) -> dict[int, tuple[str, float, float]]:
    """{id: (ground_truth, primary_score, exact_match)} using score_file's exact rules."""
    task = os.path.basename(path)[:-6]
    out: dict[int, tuple[str, float, float]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            gen = S.truncate(r.get("generate", ""))
            gt = str(r.get("ground_truth", ""))
            choices = r.get("choices") or []
            if choices:
                pred = S.match_choice(gen, choices)
                if pred is None:
                    pred = S.match_choice(S.strip_label(gen), choices)
                if pred is None:
                    pred = S.match_choice(S.strip_answer_lead(S.strip_label(gen)), choices)
                if pred is None:
                    lt = S.letter_of(gen)
                    if lt is not None and S.LETTERS.index(lt) < len(choices):
                        pred = S.LETTERS.index(lt)
                gold, ng = None, S.normalize_text(gt)
                for i, c in enumerate(choices):
                    if S.normalize_text(c) == ng:
                        gold = i
                        break
                if gold is None:
                    gold = S.match_choice(gt, choices)
                hit = float(pred is not None and gold is not None and pred == gold)
                out[r.get("id")] = (gt, hit, hit)
            elif task.startswith("lambada"):
                first = S.normalize_text(gen).split()
                head = " ".join(first[:3])
                prim = 1.0 if S.normalize_text(gt) and S.normalize_text(gt) in head else 0.0
                em = float(bool(first) and first[0] == S.normalize_text(gt))
                out[r.get("id")] = (gt, prim, em)
            else:
                out[r.get("id")] = (gt, S.f1(gen, gt),
                                    float(S.normalize_text(gen) == S.normalize_text(gt)))
    return out


def exact_mcnemar(a_only: int, b_only: int) -> float:
    """Two-sided exact binomial on discordant pairs; 1.0 when there are none."""
    d = a_only + b_only
    if d == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(d, i) for i in range(k + 1)) / 2 ** d)


def wilcoxon(diffs: list[float]) -> tuple[int, float | None]:
    """Signed-rank over non-zero diffs; normal approximation with tie-corrected ranks.

    Returned p is None when fewer than 6 non-zero differences remain, where the normal
    approximation is not defensible and no p should be quoted at all.
    """
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n < 6:
        return n, None
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    mu = n * (n + 1) / 4
    sd = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    if sd == 0:
        return n, None
    from math import erfc
    z = (w_plus - mu) / sd
    return n, min(1.0, erfc(abs(z) / 2 ** 0.5))


def compare(task: str, dclm: str, other: str, other_name: str) -> dict | None:
    A, B = row_scores(dclm), row_scores(other)
    shared = sorted(set(A) & set(B))
    gt_mismatch = [i for i in shared if S.normalize_text(A[i][0]) != S.normalize_text(B[i][0])]
    usable = [i for i in shared if i not in set(gt_mismatch)]
    if len(usable) < 20:
        print(f"  {task:14s} vs {other_name:22s} REFUSED: only {len(usable)} rows align "
              f"(shared ids {len(shared)}, ground-truth mismatches {len(gt_mismatch)}) -- "
              f"not the same rows, so no paired test is valid")
        return None
    binary = task in CHOICE_TASKS or task.startswith("lambada")
    rec = {"task": task, "vs": other_name, "n_paired": len(usable),
           "shared_ids": len(shared), "gt_mismatches": len(gt_mismatch),
           "dclm_mean": sum(A[i][1] for i in usable) / len(usable) * 100,
           "other_mean": sum(B[i][1] for i in usable) / len(usable) * 100}
    if binary:
        both = d_only = o_only = neither = 0
        for i in usable:
            a, b = A[i][1] >= 0.5, B[i][1] >= 0.5
            both += a and b
            neither += (not a) and (not b)
            d_only += a and not b
            o_only += b and not a
        p = exact_mcnemar(d_only, o_only)
        rec.update(test="exact McNemar", both=both, neither=neither,
                   dclm_only=d_only, other_only=o_only, p=p)
        win = "DCLM" if d_only > o_only else ("arm" if o_only > d_only else "tie")
        sig = "SIGNIFICANT" if p < 0.05 else "n.s."
        print(f"  {task:14s} vs {other_name:22s} n={len(usable):3d} "
              f"{rec['dclm_mean']:5.1f} vs {rec['other_mean']:5.1f}  "
              f"discordant {d_only}/{o_only}  p={p:.4f}  {sig} ({win})")
    else:
        diffs = [A[i][1] - B[i][1] for i in usable]
        nz, p = wilcoxon(diffs)
        d_only = sum(1 for i in usable if A[i][2] >= 0.5 and B[i][2] < 0.5)
        o_only = sum(1 for i in usable if B[i][2] >= 0.5 and A[i][2] < 0.5)
        p_em = exact_mcnemar(d_only, o_only)
        rec.update(test="Wilcoxon signed-rank on per-row F1", n_nonzero=nz, p=p,
                   mean_diff=sum(diffs) / len(diffs) * 100,
                   em_dclm_only=d_only, em_other_only=o_only, p_exact_match=p_em)
        ps = "n/a (<6 non-zero diffs)" if p is None else f"{p:.4g}"
        sig = "" if p is None else ("SIGNIFICANT" if p < 0.05 else "n.s.")
        print(f"  {task:14s} vs {other_name:22s} n={len(usable):3d} "
              f"F1 {rec['dclm_mean']:5.1f} vs {rec['other_mean']:5.1f}  "
              f"paired diff {rec['mean_diff']:+5.1f}  Wilcoxon p={ps} {sig}; "
              f"EM McNemar {d_only}/{o_only} p={p_em:.4g}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    # Arm directory names differ between the two capability roots: the 8 main columns live in
    # results/benchmarks/<arm>/ while the 6 re-generated option columns live in
    # results/benchmarks/fix2choices/baseline-<arm>-<tag>/. Both spellings are tried for each
    # arm so a missing directory is reported rather than silently skipping half the table.
    ap.add_argument("--arms", default="champ-dynamic,global-dit-50k")
    ap.add_argument("--fix2-arms", default="baseline-champ-dynamic-30k,baseline-global-dit-50k")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    dclm_dirs = [REPO / "results/benchmarks/baseline-dclm-1b",
                 REPO / "results/benchmarks/fix2choices/baseline-dclm-1b"]
    assert all(d.exists() for d in dclm_dirs), "DCLM capability outputs missing"
    print("=" * 100)
    print("DCLM-1B vs our arms -- PAIRED on identical rows (never two marginal accuracies)")
    print("=" * 100)
    recs = []
    pairs = [(REPO / "results/benchmarks/baseline-dclm-1b",
              REPO / "results/benchmarks", a.arms.split(","))]
    if a.fix2_arms:
        pairs.append((REPO / "results/benchmarks/fix2choices/baseline-dclm-1b",
                      REPO / "results/benchmarks/fix2choices", a.fix2_arms.split(",")))
    for dd, root, arms in pairs:
        for arm in arms:
            od = root / arm
            print(f"\n### {dd.name} ({'6 option columns' if 'fix2' in str(dd) else '8 main columns'}) "
                  f"vs {od.relative_to(REPO / 'results/benchmarks')}")
            if not od.exists():
                print(f"  MISSING {od} -- no comparison made (not reported as a tie)")
                continue
            dfiles, ofiles = S.collect(str(dd)), S.collect(str(od))
            only_here = sorted(set(dfiles) - set(ofiles))
            if only_here:
                print(f"  (DCLM has no counterpart for: {', '.join(only_here)})")
            for t in sorted(set(dfiles) & set(ofiles)):
                r = compare(t, dfiles[t], ofiles[t], arm)
                if r:
                    recs.append(r)
    print("\n" + "=" * 100)
    print("Any cell above with p >= 0.05 is NOT a demonstrated difference, whatever the two "
          "marginal accuracies look like.")
    print("=" * 100)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(recs, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

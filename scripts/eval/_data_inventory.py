#!/usr/bin/env python3
"""Which measured results are NOT in the paper yet?

The question "what data is still missing from the paper" needs a denominator that is
enumerated rather than asserted, so this walks BOTH directions:
  (a) every result dir that looks complete -> is its model named in main.tex?
  (b) every model named in a table row -> does it have a complete result dir?
Direction (b) is the one that catches a row printed from a number with no file behind it,
which has already happened in this paper (RACE 59.0 in tab:ablations, sourced nowhere).

A dir counts as complete only if it has 14 nonempty task files; a partial dir is listed
separately, because "has files" and "has all its cells" are different claims.
"""
import os
import re
import sys

BENCH = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/"
         "results/benchmarks")
TEX = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/"
       "paper_ICLR27/lalaland/main.tex")
TASKS14 = ["mmlu", "obqa", "siqa", "story_cloze", "hellaswag", "piqa", "arc_easy",
           "arc_challenge", "winogrande", "sciq", "commonsenseqa", "lambada", "race",
           "squad"]


def complete(d):
    """(nonempty task files, of 14)."""
    n = 0
    for t in TASKS14:
        p = os.path.join(d, t + ".jsonl")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            n += 1
    return n


def main():
    tex = open(TEX).read()
    # every table row label: start of line up to the first &
    rows = set()
    for m in re.finditer(r"^([A-Za-z\\][^&\n]{0,40}?)\s*&", tex, re.M):
        lab = m.group(1).strip()
        if lab and not lab.startswith(("\\multicolumn", "\\midrule", "\\cmidrule",
                                       "Model", "System", "Task", "Writer", "Variant")):
            rows.add(lab)

    dirs = sorted(d for d in os.listdir(BENCH)
                  if os.path.isdir(os.path.join(BENCH, d)))
    print("=== (a) COMPLETE result dirs (14/14) and whether the paper cites them")
    cited = uncited = 0
    for d in dirs:
        full = os.path.join(BENCH, d)
        n = complete(full)
        if n != 14:
            continue
        # a dir is "cited" if its slug's distinctive words appear in the tex
        key = d.replace("baseline-", "").replace("-", " ")
        hit = any(w in tex.lower() for w in [d.lower()])
        print(f"  {'CITED  ' if hit else 'ABSENT '} {d}")
        cited += hit
        uncited += not hit
    print(f"  -> {cited} cited by dir name, {uncited} not named in the tex")

    print("\n=== bosfix/ subdirs (the BOS and protocol reruns)")
    bf = os.path.join(BENCH, "bosfix")
    if os.path.isdir(bf):
        for d in sorted(os.listdir(bf)):
            full = os.path.join(bf, d)
            if os.path.isdir(full):
                n = complete(full)
                flag = "COMPLETE" if n == 14 else f"PARTIAL {n}/14"
                intex = "bosfix" in tex
                print(f"  {flag:14s} {d}   (tex mentions 'bosfix': {intex})")

    print("\n=== (b) printed table rows, and whether a complete dir exists for each")
    # map a row label to the dirs whose name shares its distinctive token
    missing = []
    for lab in sorted(rows):
        if lab.startswith("\\method") or not re.search(r"[A-Za-z]", lab):
            continue
        toks = [t.lower() for t in re.split(r"[^A-Za-z0-9.]+", lab) if len(t) > 2]
        if not toks:
            continue
        cands = [d for d in dirs
                 if all(t.replace(".", "") in d.lower().replace(".", "").replace("-", "")
                        or t in d.lower() for t in toks[:1])]
        best = max((complete(os.path.join(BENCH, c)) for c in cands), default=0)
        if best < 14:
            missing.append((lab, best, len(cands)))
    for lab, best, nc in missing:
        print(f"  {lab:34s} best dir has {best:2d}/14 tasks  ({nc} candidate dirs)")
    print(f"  -> {len(missing)} printed row labels with no 14/14 dir matched by name")
    print("\n  NOTE: name matching is a weak predicate -- a row can be correctly sourced "
          "from a\n  dir whose slug does not contain its label. Treat direction (b) as a "
          "list to check,\n  not as a list of defects.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

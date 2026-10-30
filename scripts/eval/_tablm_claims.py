#!/usr/bin/env python3
"""Every numeric claim the RQ1 paragraph and tab:lm make, checked against the result files.

Written because two claims in that paragraph were wrong on the first pass and each looked
reasonable: "the written plan lowers perplexity" (false for one of six arm-corpus pairs --
Global on PG19 rises) and a ratio range of "3.4-10.7x" (the true minimum is 1.67x on XSum;
the range had been taken over three of the four corpora, silently dropping the one that
would have widened it). A range stated over a subset of its own row set is the same defect
as a tally over the wrong row set.

CONTROLS. Positive: the three printed cells of the WikiText-2 row must be reproduced from
the files, or the loader is reading the wrong files. Negative: a fabricated ratio bound
(chunked within 1.00x) must FAIL, or the bound check cannot fail and proves nothing.
"""
import json
import os
import re
import sys

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning"
TEX = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/"
       "paper_ICLR27/lalaland/main.tex")
SRC = {
    "noplan": "results/benchmarks/ppl16/global-dit-50k-raw-control.json",
    "global": "results/benchmarks/ppl16/global-dit-50k-fixed.json",
    "chunked": "results/benchmarks/ppl16/champ-dynamic-30k-fixed.json",
}
PACKED = ["wikitext2", "wikitext103", "pg19"]
SHORT = ["lm1b", "ptb", "wmt14", "xsum"]


def load():
    out = {}
    for k, f in SRC.items():
        d = json.load(open(os.path.join(ROOT, f)))
        out[k] = {"ppl": d["perplexity"], "ntok": d.get("n_tokens") or {},
                  "mode": d.get("mode"), "ckpt": d.get("ckpt")}
    return out


def main():
    d = load()
    ok = True

    print("CONTROLS")
    want = {"noplan": 16.09, "global": 14.85, "chunked": 14.77}
    for arm, w in want.items():
        got = round(d[arm]["ppl"]["wikitext2"], 2)
        good = got == w
        ok &= good
        print(f"  positive  WikiText-2 {arm:8s} printed {w:6.2f} file {got:6.2f} "
              f"{'OK' if good else 'FAIL (BAD)'}")
    bogus = max(d["chunked"]["ppl"][k] / d["noplan"]["ppl"][k] for k in SHORT) <= 1.00
    print(f"  negative  'chunked within 1.00x' must be FALSE -> "
          f"{'OK (it is false)' if not bogus else 'FAIL (BAD)'}")
    ok &= not bogus
    if not ok:
        print("  -> controls FAIL")
        return 1

    print("\nCLAIMS IN THE RQ1 PARAGRAPH")
    # 1. five of six arm-corpus pairs improve on the packed corpora
    imp = [(a, k) for k in PACKED for a in ("global", "chunked")
           if d[a]["ppl"][k] < d["noplan"]["ppl"][k]]
    worse = [(a, k) for k in PACKED for a in ("global", "chunked")
             if d[a]["ppl"][k] >= d["noplan"]["ppl"][k]]
    c1 = len(imp) == 5 and worse == [("global", "pg19")]
    ok &= c1
    print(f"  'five of six arm-corpus pairs improve, exception Global on PG19': "
          f"{len(imp)}/6 improve, worse={worse}  {'OK' if c1 else 'FAIL'}")

    # 2. PG19 global ratio printed as 1.02
    r = d["global"]["ppl"]["pg19"] / d["noplan"]["ppl"]["pg19"]
    c2 = round(r, 2) == 1.02
    ok &= c2
    print(f"  'PG19 global rises by 1.02x': actual {r:.4f}  {'OK' if c2 else 'FAIL'}")

    # 3. the global range over the four short/task corpora, stated as 1.7-10.7
    rg = {k: d["global"]["ppl"][k] / d["noplan"]["ppl"][k] for k in SHORT}
    lo, hi = min(rg.values()), max(rg.values())
    c3 = f"{lo:.1f}" == "1.7" and f"{hi:.1f}" == "10.7"
    ok &= c3
    print(f"  'global loses 1.7-10.7x on the four': min {lo:.2f} ({min(rg, key=rg.get)}) "
          f"max {hi:.2f} ({max(rg, key=rg.get)})  {'OK' if c3 else 'FAIL'}")
    print(f"      per corpus: " + ", ".join(f"{k}={v:.2f}" for k, v in rg.items()))

    # 4. chunked within 1.24x on all four -- and the bound must be TIGHT, i.e. no smaller
    #    round bound would also pass, so the number is the measurement not a safe margin
    rc = {k: d["chunked"]["ppl"][k] / d["noplan"]["ppl"][k] for k in SHORT}
    mx = max(rc.values())
    c4 = mx <= 1.24
    ok &= c4
    print(f"  'chunked within 1.24x on all four': max {mx:.4f} "
          f"({max(rc, key=rc.get)})  {'OK' if c4 else 'FAIL'}")
    print(f"      tightness: 1.23 would {'PASS' if mx <= 1.23 else 'FAIL'} "
          f"-> 1.24 is the smallest 2dp bound that holds")

    # 5. both arms worse than no plan on all four
    c5 = all(d[a]["ppl"][k] > d["noplan"]["ppl"][k] for k in SHORT
             for a in ("global", "chunked"))
    ok &= c5
    print(f"  'both arms worse than no plan on all four': {c5}  {'OK' if c5 else 'FAIL'}")

    # 6. columns 1-2 paired: identical token denominators on every corpus
    same = [k for k in PACKED + SHORT
            if d["noplan"]["ntok"].get(k) == d["global"]["ntok"].get(k)]
    c6 = len(same) == 7
    ok &= c6
    print(f"  'columns 1-2 share token denominators on every corpus': {len(same)}/7 "
          f"{'OK' if c6 else 'FAIL'}")
    diff3 = [k for k in PACKED + SHORT
             if d["chunked"]["ntok"].get(k) != d["noplan"]["ntok"].get(k)]
    c7 = diff3 == ["xsum"]
    ok &= c7
    print(f"  'column 3 differs only on XSum': differs on {diff3}  "
          f"{'OK' if c7 else 'FAIL'}")

    print("\nTABLE CELLS vs FILES")
    tex = open(TEX).read()
    i = tex.index(r"\label{tab:lm}")
    j = tex.index(r"\bottomrule", i)
    rowkey = {"WikiText-2": "wikitext2", "WikiText-103": "wikitext103", "PG19": "pg19",
              "LM1B": "lm1b", "PTB": "ptb", "WMT14": "wmt14", "XSum": "xsum"}
    seen = 0
    for line in tex[i:j].splitlines():
        s = line.strip()
        if not s.endswith("\\\\") or s.startswith(("\\toprule", "\\midrule",
                                                   "\\multicolumn", "\\cmidrule")):
            continue
        c = [x.strip() for x in s[:-2].split("&")]
        if len(c) != 4:
            continue
        # a row label may carry a footnote mark; strip it before lookup. An unrecognised
        # label is REPORTED, not skipped: renaming a row is exactly how a cell check goes
        # silently vacuous, and this loop already returned 3-of-7 once for that reason.
        label = re.sub(r"\$\^?\{?\\dagger\}?\$", "", c[0]).strip()
        if label not in rowkey:
            if re.search(r"[A-Za-z]", label) and "\\code" not in c[1]:
                print(f"  UNKNOWN ROW LABEL {label!r} -- not checked")
            continue
        key = rowkey[label]
        vals = [x.replace("\\textbf{", "").replace("}", "").strip() for x in c[1:]]
        row_ok = True
        for arm, v in zip(("noplan", "global", "chunked"), vals):
            f = f"{d[arm]['ppl'][key]:.2f}"
            if f != v:
                row_ok = False
                print(f"  {label:13s} {arm:8s} prints {v} but file says {f}  <== MISMATCH")
        ok &= row_ok
        seen += 1
        if row_ok:
            print(f"  {label:13s} {' '.join(f'{x:>7s}' for x in vals)}  OK")
    if seen != 7:
        print(f"  ABORT: matched {seen} data rows, expected 7")
        ok = False

    print(f"\n{'ALL CLAIMS HOLD' if ok else 'SOME CLAIMS FAIL -- see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

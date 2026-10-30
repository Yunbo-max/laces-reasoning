"""tab:models tally, recomputed after adding OLMo-2-1B.

Three defects in this checker's own history, each of which produced a plausible wrong
number, so all three guards stay:

 1. WINDOW. An early run tallied 21 rows for a 20-row table because the window ran past
    \\end{table} into tab:capability, whose second column is also "<x>B &". -> row count is
    asserted, not assumed.
 2. PRECISION. Classifying from the Measured cell (3 decimals) instead of the tensor count
    flips a bucket: RWKV-6 prints 3.100B, whose floor at 1dp is 3.1 (=> NEITHER), but the
    count is 3,099,863,040 = 3.0999B, whose floor is 3.0 (=> TRUNC, which is what the paper
    claims). -> measured values come from the full-precision ledger.
 3. LABEL STRIPPING. `re.sub(r"\\$\\^?\\{?\\\\dagger\\}?\\$|B$", "", "2.0B$^{\\dagger}$")`
    leaves "2.0B", because `B$` anchors at end-of-string and the B is not last until the
    dagger is already gone -- one pass cannot see that. Those two rows then fell into a
    "no numeric label" bucket and vanished from the denominator, which is precisely the two
    rows the prose's "match under neither rule" is about. -> strip in two passes and assert
    every baseline row yields a numeric label.

CONTROLS. Positive: a row known to round (Gemma 2.614 -> 2.6) must land in ROUND, and a row
known to truncate (Phi-2 2.780 -> 2.7) in TRUNC, or the classifier is inert. Negative: a
fabricated row whose label matches under neither rule must land in NEITHER, and one with a
non-numeric label must land in NOLABEL -- otherwise the classifier cannot distinguish the
cases it is being trusted to count.
"""
import json
import math
import re
import sys

TEX = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/"
       "paper_ICLR27/lalaland/main.tex")
LEDGER = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/"
          "results/paper_ledger/model_params.json")

params = {k: v["params"] if isinstance(v, dict) and "params" in v else v.get("total")
          for k, v in json.load(open(LEDGER)).items()}


def strip_label(nominal):
    """The bare number in a Nominal cell, or None. Two passes: footnote marks first, then
    the unit, because the unit is only final once the mark is gone."""
    s = re.sub(r"\$\^?\{?\\dagger\}?\$", "", nominal).strip()
    s = re.sub(r"B\s*$", "", s).strip()
    return s if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", s) else None


def classify(nom, measured_b):
    """ROUND / TRUNC / NEITHER at the precision the CELL prints."""
    dp = len(nom.split(".")[1]) if "." in nom else 0
    if f"{measured_b:.{dp}f}" == nom:
        return "ROUND"
    if f"{math.floor(measured_b * 10 ** dp) / 10 ** dp:.{dp}f}" == nom:
        return "TRUNC"
    return "NEITHER"


def controls():
    ok = True
    print("CONTROLS")
    cases = [("2.6B", 2.614341888, "ROUND"), ("2.7B", 2.779683840, "TRUNC"),
             ("2.0B$^{\\dagger}$", 2.331748454, "NEITHER"), ("1B", 1.484916736, "ROUND"),
             ("3.0B", 3.099863040, "TRUNC")]
    for cell, meas, want in cases:
        nom = strip_label(cell)
        got = classify(nom, meas) if nom else "NOLABEL"
        good = got == want
        ok &= good
        print(f"  {cell:22s} {meas:<12.6f} want {want:8s} got {got:8s} "
              f"{'OK' if good else 'FAIL (BAD)'}")
    for cell in ["n/a", "--"]:
        good = strip_label(cell) is None
        ok &= good
        print(f"  {cell:22s} {'':12s} want NOLABEL  got "
              f"{'NOLABEL' if good else strip_label(cell):8s} "
              f"{'OK' if good else 'FAIL (BAD)'}")
    print(f"  -> controls {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    if not controls():
        return 1
    tex = open(TEX).read()
    i = tex.index(r"\label{tab:models}")
    j = tex.index(r"\bottomrule", i)
    rows = []
    for line in tex[i:j].splitlines():
        s = line.strip()
        if not s.endswith("\\\\") or s.startswith(("\\multicolumn", "\\midrule",
                                                   "\\toprule", "Checkpoint")):
            continue
        c = [x.strip() for x in s[:-2].split("&")]
        if len(c) == 5:
            rows.append(c)
    labels = [r[0] for r in rows]
    dups = sorted({l for l in labels if labels.count(l) > 1})
    print(f"rows={len(rows)}  duplicate labels={dups or 'none'}")
    if len(rows) != 20 or dups:
        print("ABORT: expected 20 rows (18 baselines + 2 arms), no duplicates")
        return 1

    buckets = {"ROUND": [], "TRUNC": [], "NEITHER": [], "NOLABEL": []}
    unledgered = []
    for label, nominal, meas_cell, _f, _t in rows:
        key = label if label in params else None
        if key is None:
            for k in params:
                if k.replace("\\method{} ", "") == label or k == label:
                    key = k
                    break
        if key is None:
            unledgered.append(label)
            continue
        meas_b = params[key] / 1e9
        # the Measured cell must itself be the ledger count at 3dp, or the table and the
        # ledger disagree and neither tally means anything
        if f"{meas_b:.3f}B" != meas_cell:
            print(f"ABORT: {label} prints {meas_cell} but the ledger says "
                  f"{meas_b:.3f}B")
            return 1
        nom = strip_label(nominal)
        if nom is None:
            buckets["NOLABEL"].append((label, nominal, meas_b))
        else:
            buckets[classify(nom, meas_b)].append((label, nom, meas_b))

    if unledgered:
        print(f"ABORT: no ledger entry for {unledgered}")
        return 1
    tot = sum(len(v) for v in buckets.values())
    if tot != 20:
        print(f"ABORT: buckets hold {tot} rows, not 20")
        return 1

    for k in ("ROUND", "TRUNC", "NEITHER", "NOLABEL"):
        print(f"{k:8s} {len(buckets[k]):2d}  " +
              ", ".join(f"{a}({b}B vs {c:.4f})" for a, b, c in buckets[k]))
    n_lab = tot - len(buckets["NOLABEL"])
    print(f"\nnumeric nominal labels: {n_lab}   round={len(buckets['ROUND'])} "
          f"truncate={len(buckets['TRUNC'])} neither={len(buckets['NEITHER'])}")
    print("\nPROSE SAYS: eighteen nominal labels / thirteen round / three truncate "
          "/ two neither")
    want = (18, 13, 3, 2)
    got = (n_lab, len(buckets["ROUND"]), len(buckets["TRUNC"]), len(buckets["NEITHER"]))
    print(f"MEASURED  : {got[0]} / {got[1]} / {got[2]} / {got[3]}   "
          f"=> prose {'HOLDS' if got == want else 'MUST BE UPDATED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

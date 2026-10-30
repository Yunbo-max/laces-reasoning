#!/usr/bin/env python3
"""Verify every printed cell of tab:resources against the result file it came from.

WHY THIS FILE EXISTS. The published version of this table printed a Chunked throughput
(5.2 ms/tok) that was `decode_ms_tok_bs8 / 16` while its eight neighbours printed
`decode_ms_tok_bs1` -- a different quantity in the same column -- and a Peak GB of 7.1
against a file value of 7.05. Neither is visible by reading the table. Both are caught by
re-deriving each cell from the file and comparing.

WHY THE PREDICATE IS NOT A TOLERANCE. A first version compared with abs(got-want) < 0.051,
which PASSED the negative control that restores the old 7.1 against the file's 7.05,
because |7.1 - 7.05| = 0.050 sits inside the tolerance. The tolerance was wider than the
one digit in dispute, so the check could not fail on the error it exists to catch. Cells
are now compared by the exact string the file's value formats to at the printed number of
decimals, so a 0.05 disagreement is a mismatch.

7.05 IS PRINTED WITH TWO DECIMALS ON PURPOSE. It is an exact tie at one decimal: Python's
round() gives 7.0 (banker's rounding on the binary value, which is just below 7.05) and
half-up gives 7.1. Printing either asserts a tie-break; printing 7.05 reproduces the file.
The other eight rows round unambiguously to one decimal.

Run: verify_tab_resources.py [--tex <main.tex>] [--results <dir>] [--self-test]
Exit 0 only if every cell matches. --self-test runs five negative controls and one positive
control and exits non-zero unless all five negatives FAIL and the positive PASSES.
"""
import argparse
import json
import os
import re
import sys

TEX_DEFAULT = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/"
               "paper_ICLR27/lalaland/main.tex")
RES_DEFAULT = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/"
               "laces-reasoning/results/benchmarks/efficiency")

# row-name substring -> result file stem. The two \method{} rows resolve to the WARM files:
# the non-warm files of the same name are the pre-warm-up era and are not what is printed.
FILES = {
    "Qwen2.5-3B": "qwen2.5-3b",
    "Llama-3.2-3B": "llama-3.2-3b",
    "Phi-2": "phi-2",
    "Gemma-2-2B": "gemma-2-2b",
    "Mamba 2.8B": "mamba-2.8b",
    "RWKV-6 World2.1": "rwkv6-world",
    "RWKV-7 World3": "rwkv7-world3",
    "Global": "global-dit-50k-warm",
    "Chunked": "champ-dynamic-30k-warm",
}
ROW_STARTS = ("Qwen", "Llama", "Phi", "Gemma", "Mamba", "RWKV", "\\method")
GEN_LEN = 512


def table_block(tex):
    i = tex.index("\\label{tab:resources}")
    return tex[i:tex.index("\\end{table*}", i)]


def cell_number(c):
    """The numeric literal a cell prints, and how many decimals it prints it with."""
    c = re.sub(r"\\(textbf|mathbf|underline)\{", "", c).replace("}", "").replace("$", "")
    m = re.search(r"-?\d+\.?\d*", c)
    if not m:
        return None, None
    lit = m.group()
    dp = len(lit.split(".")[1]) if "." in lit else 0
    return float(lit), dp


def fmt(v, dp):
    return None if v is None else f"{v:.{dp}f}"


def expected(d):
    """Re-derive the four data columns from the file, as the caption defines them."""
    b1 = d["decode_ms_tok_bs1"]
    plan_s = d["plan_ms"] / 1000.0 if "plan_ms" in d else None
    return {
        "Peak GB": d["peak_mem_gb"],
        "token/s": 1000.0 / b1,
        "Plan (s)": plan_s,
        "Total (s)": (plan_s or 0.0) + GEN_LEN * b1 / 1000.0,
    }


def check(block, resdir, verbose=True):
    cols = ["Peak GB", "token/s", "Plan (s)", "Total (s)"]
    bad = n = rows = 0
    if verbose:
        print(f"{'row':22s} {'column':10s} {'printed':>8s} {'from file':>10s}  verdict")
    for line in block.splitlines():
        ls = line.strip()
        if not ls.startswith(ROW_STARTS):
            continue
        rows += 1
        cells = [c.strip() for c in ls.rstrip("\\").split("&")]
        key = next((k for k in FILES if k in cells[0]), None)
        if key is None:
            print(f"   UNMATCHED ROW: {cells[0][:60]!r} -- selector missed a printed row")
            bad += 1
            continue
        if len(cells) != len(cols) + 1:
            print(f"   {key}: {len(cells)} cells, expected {len(cols)+1}")
            bad += 1
            continue
        d = json.load(open(os.path.join(resdir, FILES[key] + ".json")))
        exp = expected(d)
        for col, raw in zip(cols, cells[1:]):
            n += 1
            got, dp = cell_number(raw)
            want = exp[col]
            if got is None and want is None:
                v = "OK (--)"
            elif got is None or want is None:
                v = "MISMATCH"
            else:
                # compare as printed: the file's value formatted to the cell's own precision
                v = "OK" if fmt(got, dp) == fmt(want, dp) else "MISMATCH"
            bad += v == "MISMATCH"
            if verbose:
                w = "--" if want is None else f"{want:.4g}"
                print(f"{key:22s} {col:10s} {str(got if got is not None else '--'):>8s} "
                      f"{w:>10s}  {v}")
    if verbose:
        print(f"\nrows checked: {rows}/{len(FILES)}   cells checked: {n}   mismatches: {bad}")
    if rows != len(FILES):
        print(f"   DENOMINATOR FAIL: matched {rows} rows, expected {len(FILES)} -- "
              f"a check that skips rows cannot fail on them")
        bad += 1
    if n == 0:
        print("   DENOMINATOR FAIL: 0 cells checked")
        bad += 1
    return bad


def self_test(block, resdir):
    NEG = [
        ("N1 token/s 15.5 -> 15.6", lambda b: b.replace("& 15.5 &", "& 15.6 &")),
        ("N2 Peak 7.05 -> 7.1 (the old printed value)",
         lambda b: b.replace("(2.95B) & 7.05", "(2.95B) & 7.1")),
        ("N3 Plan 0.89 -> -- (drop a present value)",
         lambda b: b.replace("& 0.89 & 38.0", "& -- & 38.0")),
        ("N4 Total 37.5 -> 5.2 (the old bs8/16 figure)",
         lambda b: b.replace("& 37.5", "& 5.2")),
        ("N5 row renamed so the selector misses it",
         lambda b: b.replace("Mamba 2.8B (2.77B)", "Mamba-2.8B (2.77B)")),
        ("N6 Peak GB read from the wrong batch (bs1 file key)",
         lambda b: b.replace("& 8.56 &", "& 8.4 &").replace("(2.77B) & 10.2", "(2.77B) & 6.14")),
    ]
    # An acceptable ALTERNATIVE rendering, not an error: 7.05 is an exact tie at one decimal,
    # and `round()` resolves it to 7.0, so a cell printing 7.0 still equals the file value at
    # the precision it prints. It is a positive control precisely because an earlier version
    # of this self-test listed it as a negative one and then reported the checker as holed
    # when the checker was right.
    POS = [
        ("P1 the table as committed", lambda b: b),
        ("P2 Peak 7.05 -> 7.0 (the tie resolved by round(), still faithful)",
         lambda b: b.replace("(2.95B) & 7.05", "(2.95B) & 7.0")),
    ]
    ok = True
    print("negative controls -- each MUST report at least one mismatch:")
    for label, mut in NEG:
        m = mut(block)
        if m == block:
            print(f"  {label:58s} VACUOUS: the mutation changed nothing")
            ok = False
            continue
        bad = check(m, resdir, verbose=False)
        print(f"  {label:58s} mismatches={bad} -> {'FAIL (good)' if bad else 'PASS (BAD)'}")
        ok &= bad > 0
    print("positive controls -- each MUST report zero mismatches:")
    for label, mut in POS:
        m = mut(block)
        bad = check(m, resdir, verbose=False)
        print(f"  {label:58s} mismatches={bad} -> "
              f"{'PASS (good)' if bad == 0 else 'FAIL (BAD)'}")
        ok &= bad == 0
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=TEX_DEFAULT)
    ap.add_argument("--results", default=RES_DEFAULT)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    block = table_block(open(a.tex).read())
    if a.self_test:
        sys.exit(self_test(block, a.results))
    sys.exit(1 if check(block, a.results) else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect every DCLM-1B number, refusing any cell the model's 2048-token window cannot see.

THE RULE THIS SCRIPT ENFORCES. A truncated long-context cell must be reported as "not measurable
within this model's window", never as a score. The precedent is phi-2: with `--max_input_tokens
32768` against its 2048 window, 7 cells collapsed to 0.0; after clamping, its RULER aggregate went
0.0000 -> 0.4103 and one cell went 0.000 -> 0.990 -- a 0.990 "earned" while seeing 44% of the
document. Both the pre-clamp 0.0 and the post-clamp 0.990 are artefacts of where the truncation
landed relative to the needle, so neither is a measurement of the model.

DCLM-1B is the extreme case: its trained context is 2048 (`config.json: seq_len`, and it declares
NEITHER `max_position_embeddings` NOR `n_positions`), while the briefed grid starts at 4096. So
this script is expected to refuse most of the long-context grid and to print a short list of
genuinely in-window numbers.

The refusal decision is made from the RESULT FILES' own per-row fields, not from any CLI argument:
`config.max_input_tokens` is the raw request (e.g. 32768), NOT the effective clamp, so auditing it
would report 32768 for a run that actually saw 1536 tokens. Per-cell `seen_fraction`,
`rows_truncated` and `mean_input_tokens_seen` are the only trustworthy evidence, and the harness
writes them for every cell.

Usage:
  python3 scripts/eval/report_dclm1b.py                     # human-readable
  python3 scripts/eval/report_dclm1b.py --json out.json     # machine-readable, for the ledger
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222")
REPO = ROOT / "research/laces-reasoning"
REV = "f43c272dab1b06bc5fdd6bbb5819f8a30ef2833b"

LONGCTX = REPO / "results/benchmarks/longctx"
INWIN = REPO / "results/benchmarks/longctx_inwindow"
PPL = REPO / "results/benchmarks/ppl16/dclm-1b.json"
GM_MAIN = REPO / "results/benchmarks/baseline-dclm-1b"
GM_FIX2 = REPO / "results/benchmarks/fix2choices/baseline-dclm-1b"

LONGCTX_FILES = [
    (INWIN / "dclm-1b__babilong_inwindow.json", "babilong (in-window 0k/1k/2k)"),
    (LONGCTX / "dclm-1b__babilong.json", "babilong (briefed 4k/8k/16k)"),
    (LONGCTX / "dclm-1b__ruler.json", "ruler"),
    (LONGCTX / "dclm-1b__longbench.json", "longbench"),
    (LONGCTX / "dclm-1b__infinitebench.json", "infinitebench"),
    (LONGCTX / "dclm-1b__memeval.json", "memeval"),
]

# LongMemEval's 50-row slice: 22 of 50 questions are effectively two-way, giving a random floor of
# 0.220 that no CI over n=50 can exclude, on top of scorer defects already documented in this repo
# (naive containment credits a reversed-order answer). Any memeval number at or below this is
# uninformative, and that is a property of the slice, not of DCLM.
MEMEVAL_FLOOR = 0.220


def cell_verdict(agg: dict) -> tuple[bool, str]:
    """Reportable only if the model saw the whole prompt on EVERY row of the cell."""
    seen = agg.get("seen_fraction")
    trunc = agg.get("rows_truncated")
    if trunc is None and seen is None:
        return False, "no truncation audit fields in this file (pre-audit harness run?)"
    if trunc:
        return False, (f"{trunc}/{agg.get('n')} rows truncated, "
                       f"seen_fraction={seen:.4f}" if seen is not None
                       else f"{trunc} rows truncated")
    if seen is not None and seen < 1.0:
        return False, f"seen_fraction={seen:.4f} < 1.0"
    return True, "fully in-window"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out: dict = {"model": "TRI-ML/DCLM-1B", "revision": REV, "positional_window": 2048,
                 "window_source": "config.json:seq_len",
                 "reportable": {}, "refused": {}, "missing_files": []}
    print("=" * 86)
    print("DCLM-1B (TRI-ML/DCLM-1B, third-party Toyota Research reproduction) results")
    print(f"revision {REV}   positional window 2048 (config.json:seq_len)")
    print("=" * 86)

    # ---- long context -------------------------------------------------------------------
    print("\n--- LONG CONTEXT (cell-by-cell window audit) ---")
    for path, label in LONGCTX_FILES:
        if not path.exists():
            print(f"[{label}] MISSING {path}")
            out["missing_files"].append(str(path))
            continue
        doc = json.loads(path.read_text())
        aggs = doc.get("aggregates", {})
        decode = doc.get("decode_path", "?")
        print(f"\n[{label}]  {path}")
        print(f"  decode_path={decode}  decode_is_trustworthy={doc.get('decode_is_trustworthy')}")
        for cell, agg in sorted(aggs.items()):
            ok, why = cell_verdict(agg)
            score = agg.get("score")
            key = f"{label}/{cell}"
            if ok:
                note = ""
                if "memeval" in label and score is not None and score <= MEMEVAL_FLOOR:
                    note = (f"  <-- AT OR BELOW the {MEMEVAL_FLOOR} random floor of this "
                            f"50-row slice: UNINFORMATIVE")
                print(f"  REPORTABLE  {cell:34s} score={score}  n={agg.get('n')}{note}")
                out["reportable"][key] = {"score": score, "n": agg.get("n"),
                                          "source_file": str(path), "evidence": agg}
            else:
                print(f"  REFUSED     {cell:34s} not measurable within this model's window "
                      f"({why}); withheld score would have been {score}")
                out["refused"][key] = {"reason": why, "withheld_score": score,
                                       "n": agg.get("n"), "source_file": str(path),
                                       "evidence": agg}

    # ---- perplexity ---------------------------------------------------------------------
    print("\n--- PERPLEXITY (256-token chunks, far inside 2048: fully in-window) ---")
    if PPL.exists():
        doc = json.loads(PPL.read_text())
        out["ppl"] = {"source_file": str(PPL), "perplexity": doc.get("perplexity"),
                      "bits_per_byte": doc.get("bits_per_byte"),
                      "n_tokens": doc.get("n_tokens")}
        for corpus, ppl in (doc.get("perplexity") or {}).items():
            bpb = (doc.get("bits_per_byte") or {}).get(corpus)
            print(f"  {corpus:12s} PPL={ppl}  bpb={bpb}  "
                  f"n_tokens={(doc.get('n_tokens') or {}).get(corpus)}")
        p = doc.get("perplexity") or {}
        if p.get("wikitext2") == p.get("wikitext103") and p.get("wikitext2") is not None:
            print("  NOTE wikitext2 == wikitext103 is EXPECTED, not a bug: the two preprocessed "
                  "dirs share their first 10631 chunks and --max_files 100 reads the same 100. "
                  "Every published ppl16 row shows the same tie.")
    else:
        print(f"  MISSING {PPL}")
        out["missing_files"].append(str(PPL))

    # ---- short-context capability -------------------------------------------------------
    print("\n--- SHORT-CONTEXT CAPABILITY (generate-then-match, all in-window) ---")
    sys.path.insert(0, str(REPO / "scripts/eval"))
    try:
        from score_gen_match import collect, score_file
    except Exception as exc:                                        # pragma: no cover
        print(f"  cannot import score_gen_match ({exc})")
        collect = score_file = None
    if collect is not None:
        out["capability"] = {}
        for d, tasks in ((GM_MAIN, ["mmlu", "obqa", "race", "squad", "story_cloze",
                                    "hellaswag", "lambada", "siqa"]),
                         (GM_FIX2, ["arc_easy", "arc_challenge", "winogrande", "sciq",
                                    "commonsenseqa", "piqa"])):
            if not d.exists():
                print(f"  MISSING {d}")
                out["missing_files"].append(str(d))
                continue
            files = collect(str(d))
            for t in tasks:
                if t not in files:
                    print(f"  {t:16s} (not produced yet)")
                    continue
                _task, acc, _x, n = score_file(files[t])
                # A file that exists but has 0 scorable rows is a run still in flight (the
                # harness appends as it goes), and score_file returns nan for it. Printing
                # "acc=   nan" next to real numbers invites transcription of a non-number, and
                # printing 0.0 would be worse. Withhold it and say why.
                if not n or acc != acc:      # acc != acc  <=> acc is NaN
                    print(f"  {t:16s} INCOMPLETE (n={n}) -- file exists but no scorable rows "
                          f"yet; no number reported")
                    out.setdefault("incomplete", {})[t] = {"n": n,
                                                           "source_file": str(files[t])}
                    continue
                print(f"  {t:16s} acc={acc:6.1f}  n={n}")
                out["capability"][t] = {"acc": acc, "n": n, "source_file": str(files[t])}

    print("\n" + "=" * 86)
    print(f"reportable long-context cells: {len(out['reportable'])}   "
          f"refused: {len(out['refused'])}")
    print("Refused cells must be printed as 'not measurable within this model's window' -- "
          "never as a score, and never as a 0.")
    print("=" * 86)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

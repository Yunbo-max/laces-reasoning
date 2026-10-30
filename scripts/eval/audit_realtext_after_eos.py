#!/usr/bin/env python3
"""Re-derive the section-46 "real text after EOS" table from current bytes.

WHY THIS FILE EXISTS: the section-46 table in logs/qz/DECODE_FIX_JOBS.md was produced by an
inline heredoc probe that was never saved. That makes its counts unreproducible -- the
"unbuildable check is no check" defect. Five of the six canonical files it audited have since
been rewritten (mtimes 09:19-12:31 on 2026-09-21), so the table needed re-deriving anyway.

TWO DEFECTS THIS SCRIPT IS BUILT TO AVOID:

1. NESTING LEVEL. `stop_ids` is a TOP-LEVEL key, not `config["stop_ids"]`. A probe reading
   `config` prints ABSENT for all 15 cells -- a uniform negative that reads like a clean
   finding. Asserted below, not assumed.

2. VACUOUS PREDICATE. A counter that never fires looks exactly like a clean corpus. So
   --self-test injects a synthetic row with real text after a stop id and requires the
   counter to fire on it (TP) and to stay silent on a row without one (FP).

The stop set is READ FROM EACH FILE, never hand-typed: hand-typed checker parameters are the
single most repeated defect in this investigation.
"""
import argparse
import json
import os
import sys

SUITES = ("longbench", "ruler", "babilong", "infinitebench", "memeval")
MODELS = ("rwkv7-world3", "champ-dynamic-30k", "global-dit-50k")


def classify(ids, stop_set, text_of_tail):
    """Return one of 'none' | 'pure_pad' | 'realtext'.

    'realtext' = a stop id occurs and is followed by at least one NON-stop id whose decoded
    form is not whitespace. 'pure_pad' = stop id followed only by stop/padding ids.
    """
    if not stop_set or not ids:
        return "none"
    first = None
    for i, t in enumerate(ids):
        if t in stop_set:
            first = i
            break
    if first is None:
        return "none"
    tail = [t for t in ids[first + 1:]]
    if not tail:
        return "pure_pad"
    if all(t in stop_set for t in tail):
        return "pure_pad"
    if text_of_tail is not None and not text_of_tail.strip():
        return "pure_pad"
    return "realtext"


def reference_stop_set(d):
    """The stop set a FIXED cell declares, read off disk -- never hand-typed.

    Needed because the interesting question for a cell with `stop_ids` ABSENT is counterfactual:
    would the correct stop set have fired here? Answering it with the declared set is vacuous
    (there is none), which is why the plain audit prints 0 for those cells.
    """
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, name)) as fh:
                doc = json.load(fh)
        except Exception:
            continue
        if isinstance(doc, dict) and doc.get("stop_ids"):
            return set(doc["stop_ids"]), name
    return set(), None


def audit_file(path, reference=None):
    with open(path) as fh:
        doc = json.load(fh)
    # DEFECT 1 GUARD: prove where stop_ids lives rather than trusting either level.
    top_has = "stop_ids" in doc
    cfg = doc.get("config", {}) or {}
    cfg_has = "stop_ids" in cfg
    stop = doc.get("stop_ids") if top_has else cfg.get("stop_ids")
    stop_set = set(stop) if stop else set()
    applied = "declared"
    if not stop_set and reference:
        stop_set = set(reference)
        applied = "reference"
    rows = doc.get("rows") or []
    counts = {"none": 0, "pure_pad": 0, "realtext": 0}
    n_with_stop = 0
    for r in rows:
        ids = r.get("prediction_token_ids") or []
        if stop_set and any(t in stop_set for t in ids):
            n_with_stop += 1
        counts[classify(ids, stop_set, None)] += 1
    return {
        "path": path,
        "rows": len(rows),
        "stop_ids": stop if stop else "ABSENT",
        "stop_level": "top" if top_has else ("config" if cfg_has else "-"),
        "decode_path": doc.get("decode_path"),
        "applied": applied,
        "with_eos": n_with_stop,
        "pure_pad": counts["pure_pad"],
        "realtext": counts["realtext"],
    }


def self_test():
    """Negative + positive control. A checker with TP=0 reports 0 defects on a broken file."""
    stop = {0, 2, 65530}
    tp = classify([5, 7, 2, 9, 11], stop, None)
    fp = classify([5, 7, 9, 11], stop, None)
    pad = classify([5, 7, 2, 2, 0], stop, None)
    empty = classify([5, 7, 9], set(), None)
    ok = True
    for label, got, want in (
        ("TP realtext-after-stop", tp, "realtext"),
        ("FP no-stop-present", fp, "none"),
        ("pure-pad tail", pad, "pure_pad"),
        ("no stop set declared", empty, "none"),
    ):
        flag = "OK " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{flag}] {label:26s} got={got!r} want={want!r}")
    print(f"  self-test: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/benchmarks/longctx")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--counterfactual", action="store_true",
                    help="apply a FIXED cell's stop set to cells that declare none. This is the "
                         "test §46 actually ran: for an ABSENT cell the declared-set count is 0 "
                         "by construction, which is NOT-APPLICABLE, not PASS.")
    args = ap.parse_args()

    print("=== SELF-TEST (a counter that never fires is not evidence of a clean corpus) ===")
    if not self_test():
        print("self-test FAILED -- counts below are not trustworthy")
        return 2
    if args.self_test:
        return 0

    print()
    ref, ref_src = (set(), None)
    if args.counterfactual:
        ref, ref_src = reference_stop_set(args.dir)
        if not ref:
            print("NO reference stop set found on disk -- cannot run counterfactual mode")
            return 2
        print(f"=== COUNTERFACTUAL MODE: stop set {sorted(ref)} read from {ref_src} ===")
    print(f"=== §46 TABLE RE-DERIVED from {args.dir} ===")
    hdr = (f"{'cell':36s} {'rows':>5s} {'w/eos':>6s} {'pure':>5s} {'REAL':>5s} "
           f"{'set':>9s} {'lvl':>4s} {'decode':14s} stop_ids")
    print(hdr)
    tot_real = 0
    tot_rows_affected = 0
    n_cells_affected = 0
    for m in MODELS:
        for s in SUITES:
            p = os.path.join(args.dir, f"{m}__{s}.json")
            if not os.path.exists(p):
                continue
            r = audit_file(p, reference=ref if args.counterfactual else None)
            tot_real += r["realtext"]
            if r["realtext"]:
                tot_rows_affected += r["realtext"]
                n_cells_affected += 1
            print(f"{m + '__' + s:36s} {r['rows']:5d} {r['with_eos']:6d} "
                  f"{r['pure_pad']:5d} {r['realtext']:5d} {r['applied']:>9s} "
                  f"{r['stop_level']:>4s} {str(r['decode_path']):14s} {r['stop_ids']}")
    print()
    print(f"TOTAL rows with real text after a stop id: {tot_real}  "
          f"(across {n_cells_affected} cells)")
    if not args.counterfactual:
        print("NOTE: cells with stop_ids ABSENT cannot exhibit the defect by construction -- for "
              "them this is NOT-APPLICABLE, not PASS. Re-run with --counterfactual for §46's test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

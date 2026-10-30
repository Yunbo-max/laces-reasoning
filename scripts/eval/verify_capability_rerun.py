#!/usr/bin/env python3
"""Verify a capability re-run landed, and report old vs new for every cell.

This is the part that has to be able to FAIL. Three controls, each printing its own
denominator, because a check whose denominator is 0 cannot fail:

  C1 PROVENANCE  run_summary.json records restore_bos exactly as intended.
  C2 LANDED      on rows whose prompt TEXT is byte-identical old vs new, prompt_tokens is
                 exactly +1 when BOS was restored (exactly 0 when it was not). A delta set
                 of {0} under --expect-bos means the flag never reached the tokenizer.
  C3 EFFECT      predictions actually changed on some rows. 0/N changed with a +1 token
                 delta would mean the extra token is inert, which is itself a finding.

WHY C2 PAIRS ON PROMPT TEXT AND NOT ON A FIXED TASK LIST. An earlier version used a
hard-coded (sciq, piqa, mmlu) as its control tasks. That is only sound when the old cell
already carries the option-inlined prompt. Measured: build_prompt -> inline_choices rewrites
exactly six tasks (piqa arc_easy arc_challenge winogrande sciq commonsenseqa) and leaves the
other eight byte-identical to the stored plain cells. So for a model WITH a fix2choices dir
the delta on sciq/piqa is purely BOS, but for a model WITHOUT one (rwkv-6, olmo, nemotron:
0 files) old_path falls back to the plain dir and the delta becomes "option block + BOS",
which made C2 exit 1 on a run where the BOS had in fact landed correctly -- a false failure
that would have been read as "the fix did not work". Selecting rows by prompt-text identity
makes the control self-validating instead of depending on which dirs happen to exist, and the
count of prompt-CHANGED rows is printed separately so a protocol change can never hide
inside a BOS check.

The old row is NOT one directory. The published table resolves six option-inlined tasks
(piqa arc_easy arc_challenge winogrande sciq commonsenseqa) from fix2choices/ and the rest
from the plain baseline dir; `old_path` mirrors that, fix2choices first. Pointing at either
dir alone does not reproduce the printed row. A fresh full-14-task run of the current script
therefore reproduces the published mixed protocol natively, which is also why one run
backfills the six fix2 cells for a model that never had them.

Usage:
  verify_capability_rerun.py --new <dir> --old <dir>[,<dir>...] [--expect-bos|--expect-no-bos]
"""
import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import score_gen_match as S  # noqa: E402

TASKS = ("mmlu obqa siqa story_cloze hellaswag piqa arc_easy arc_challenge "
         "winogrande sciq commonsenseqa lambada race squad").split()
# The table's Avg drops these three; print both means so a row change can never be
# confused with a change in which columns the row averages over.
AVG_TASKS = [t for t in TASKS if t not in ("hellaswag", "arc_challenge", "squad")]


def pct(x):
    return 100 * x if x <= 1 else x


def load_rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True, help="comma-separated, resolved in order")
    ap.add_argument("--expect-bos", dest="expect", action="store_true", default=None)
    ap.add_argument("--expect-no-bos", dest="expect", action="store_false")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    old_dirs = [d for d in a.old.split(",") if d]

    def old_path(t):
        for d in old_dirs:
            p = f"{d}/{t}.jsonl"
            if os.path.exists(p):
                return p
        return None

    print(f"=== {a.label or a.new} ===")

    # ---- C1 PROVENANCE -------------------------------------------------------
    s = json.load(open(f"{a.new}/run_summary.json"))
    got = s.get("restore_bos")
    print(f"C1 PROVENANCE restore_bos={got} expected={a.expect} "
          f"rep_pen={s.get('repetition_penalty')} temp={s.get('temperature')} "
          f"max_samples={s.get('max_samples')}")
    if a.expect is not None and bool(got) != bool(a.expect):
        print("   C1 FAIL: the flag recorded in run_summary is not the one intended")
        sys.exit(1)

    # ---- C2 LANDED + C3 EFFECT ----------------------------------------------
    # Pair on prompt TEXT, not on a task whitelist: only rows whose prompt is byte-identical
    # old vs new isolate the BOS. Rows whose prompt text changed are counted and reported
    # separately -- they are the protocol (option-inlining) axis, not the BOS axis.
    want = 1 if a.expect else 0
    checked = 0
    tot_same = tot_ch_prompt = 0
    for t in TASKS:
        po, pn = old_path(t), f"{a.new}/{t}.jsonl"
        if not (po and os.path.exists(pn)):
            continue
        old = load_rows(po)
        new = load_rows(pn)
        n = min(len(old), len(new))
        # both carry id for most tasks; where they do, refuse to pair if ids disagree
        bad = [i for i in range(n) if "id" in old[i] and "id" in new[i]
               and old[i]["id"] != new[i]["id"]]
        if bad:
            print(f"   C2 FAIL {t}: line-order pairing broken at row {bad[0]} "
                  f"(old id={old[bad[0]]['id']} new id={new[bad[0]]['id']})")
            sys.exit(1)
        same = [i for i in range(n) if old[i].get("prompt") == new[i].get("prompt")]
        tot_same += len(same)
        tot_ch_prompt += n - len(same)
        if not same:
            print(f"   C2 SKIP {t:<14} n={n:<4} 0 rows with identical prompt text "
                  f"(all {n} rewritten by the option-inliner) -- BOS not isolable here")
            continue
        d = sorted({new[i]["prompt_tokens"] - old[i]["prompt_tokens"] for i in same})
        ch = sum(1 for i in same if new[i].get("generate") != old[i].get("generate"))
        ok = (d == [want])
        print(f"C2 LANDED {t:<14} n_same_prompt={len(same):<4} (prompt changed {n - len(same)})"
              f"  token delta set={d} (want [{want}]) -> {'OK' if ok else 'FAIL'}")
        print(f"C3 EFFECT {t:<14} predictions changed {ch}/{len(same)}"
              + ("   <-- 0 changed: extra token is inert" if a.expect and ch == 0 else ""))
        checked += 1
        if not ok:
            print("   C2 FAIL: the intervention did not land as specified")
            sys.exit(1)
    print(f"C2/C3 denominators: {checked} task(s) checkable, {tot_same} identical-prompt rows, "
          f"{tot_ch_prompt} rows whose prompt text also changed (protocol axis)")
    if checked == 0:
        print("   C2/C3 FAIL: 0 tasks could be checked -- a check with denominator 0 "
              "cannot fail, so this is not a pass")
        sys.exit(1)
    if tot_same == 0:
        print("   C2/C3 FAIL: 0 identical-prompt rows -- nothing isolates the BOS")
        sys.exit(1)

    # ---- scores --------------------------------------------------------------
    print(f"\n{'task':16s} {'old':>7s} {'new':>7s} {'delta':>7s}  {'in Avg':>6s}")
    o_all, n_all = {}, {}
    for t in TASKS:
        po, pn = old_path(t), f"{a.new}/{t}.jsonl"
        if not os.path.exists(pn):
            continue
        _, an, _, _ = S.score_file(pn)
        vn = pct(an)
        tag = "yes" if t in AVG_TASKS else "--"
        if po:
            _, ao, _, _ = S.score_file(po)
            vo = pct(ao)
            o_all[t], n_all[t] = vo, vn
            print(f"{t:16s} {vo:7.2f} {vn:7.2f} {vn - vo:+7.2f}  {tag:>6s}")
        else:
            print(f'{t:16s} {"--":>7s} {vn:7.2f} {"n/a":>7s}  {tag:>6s}   (no old cell)')

    for label, sel in (("AVG(11, as printed)", AVG_TASKS), ("MEAN(all 14)", TASKS)):
        ks = [t for t in sel if t in o_all]
        if not ks:
            continue
        mo = statistics.mean(o_all[t] for t in ks)
        mn = statistics.mean(n_all[t] for t in ks)
        # paired sign test over tasks, two-sided exact
        w = sum(1 for t in ks if n_all[t] > o_all[t])
        ties = sum(1 for t in ks if n_all[t] == o_all[t])
        n = len(ks) - ties
        if n:
            lo = sum(math.comb(n, k) for k in range(w, n + 1)) / 2 ** n
            hi = sum(math.comb(n, k) for k in range(0, w + 1)) / 2 ** n
            p = min(1.0, 2 * min(lo, hi))
        else:
            p = 1.0
        print(f"{label:20s} {mo:7.2f} {mn:7.2f} {mn - mo:+7.2f}   "
              f"(n_tasks={len(ks)}, new>old {w}/{n}, sign p={p:.4f}"
              + (f", ties={ties}" if ties else "") + ")")


if __name__ == "__main__":
    main()

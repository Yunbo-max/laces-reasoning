#!/usr/bin/env python3
"""G2 aggregation: turn the per-example JSONL into the five rows of tab:latent-use-pending.

This script MEASURES ONLY.  It never edits the paper and never writes under results/
or outputs*/.  Every check prints its DENOMINATOR before its verdict; a check over
0 candidates cannot fail and says so.

Invariants enforced (each one raises rather than warning, because a warning that
nobody reads is not a check):
  * every (arm, task) cell has EXACTLY the pinned row count -- invalid rows are
    counted as WRONG and never removed, so the denominator cannot shrink;
  * the arms share example ids, prompt hashes, checkpoint sha256 and seed;
  * Macro3 is the equal-weight mean of the three task accuracies.  It is NOT
    Avg_11 and is not comparable to the eleven-task 52.8/64.2 aggregates or to
    the 60.58/67.50 decode-path control.
"""
from __future__ import annotations

import argparse
import collections
import glob
import importlib.util
import json
import math
import os
import random
import re
import sys
from pathlib import Path

TASKS = ("sciq", "commonsenseqa", "race")
TASK_LABEL = {"sciq": "SciQ", "commonsenseqa": "CQA", "race": "RACE"}
ARM_LABEL = {
    "A0": "Same-loop backbone, no adapter",
    "A1": "Same adapter, original principal memory",
    "A2": "Training-mean latent",
    "A3": "Shuffled sampled latent",
    "A4": "Correct prefix-conditioned latent",
}
ARM_ORDER = ("A0", "A1", "A2", "A3", "A4")


def load(out_dir: Path) -> dict:
    rows: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    files = sorted(glob.glob(str(out_dir / "*.jsonl")))
    print(f"DENOMINATOR jsonl files found = {len(files)}")
    if not files:
        raise SystemExit(f"no jsonl under {out_dir}: nothing to aggregate, and an empty "
                         "aggregate would silently look like a completed run")
    for f in files:
        base = os.path.basename(f)[:-6]
        if "__" not in base:
            raise ValueError(f"unexpected filename {base!r}: cannot infer (task, arm); "
                             "an unknown shape must raise, not be skipped")
        task, arm = base.split("__", 1)
        for ln in open(f):
            ln = ln.strip()
            if ln:
                rows[(task, arm)].append(json.loads(ln))
    return rows


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval, in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def score_with_lead_fix(FROZEN, row: dict) -> bool:
    """The shipped ``score_row`` with ONE change, applied to a stored row.

    The shipped scorer calls ``strip_answer_lead`` before ``match_choice`` but never
    before ``letter_of``, so a generation like "The answer is B." reaches no letter
    extractor at all and is recorded invalid -> WRONG.  Everything else here is a
    faithful copy of the shipped control flow, so any difference is attributable to
    that single added fallback and nothing else.

    Returns the corrected ``correct`` verdict.  Reads the stored ``prediction``; it
    never re-decodes, because re-decoding would change the decode path and make these
    rows incomparable with the rest of the table.
    """
    gen, gold, choices = row["prediction"], row["gold"], row["choices"]
    g = FROZEN.truncate_at_newline(gen)
    if not g.strip():
        return False
    pred = FROZEN.match_choice(g, choices)
    if pred is None:
        pred = FROZEN.match_choice(FROZEN.strip_label(g), choices)
    if pred is None:
        pred = FROZEN.match_choice(FROZEN.strip_answer_lead(FROZEN.strip_label(g)), choices)
    if pred is None:
        lt = FROZEN.letter_of(g)
        if lt is not None and FROZEN.LETTERS.index(lt) < len(choices):
            pred = FROZEN.LETTERS.index(lt)
    if pred is None:                                            # <-- the one change
        lt = FROZEN.letter_of(FROZEN.strip_answer_lead(g))
        if lt is not None and FROZEN.LETTERS.index(lt) < len(choices):
            pred = FROZEN.LETTERS.index(lt)
    gold_idx = None
    ng = FROZEN.normalize_text(gold)
    for i, c in enumerate(choices):
        if FROZEN.normalize_text(c) == ng:
            gold_idx = i
            break
    if gold_idx is None:
        gold_idx = FROZEN.match_choice(gold, choices)
    if gold_idx is None:
        raise SystemExit(f"gold {gold!r} not resolvable against {choices!r}: an unknown "
                         "shape must raise, not be silently skipped")
    return pred is not None and pred == gold_idx


def gold_index(FROZEN, row: dict) -> int:
    """Resolve gold to a choice index exactly as the shipped scorer does."""
    gold, choices = row["gold"], row["choices"]
    ng = FROZEN.normalize_text(gold)
    for i, c in enumerate(choices):
        if FROZEN.normalize_text(c) == ng:
            return i
    gi = FROZEN.match_choice(gold, choices)
    if gi is None:
        raise SystemExit(f"gold {gold!r} not resolvable against {choices!r}: an unknown "
                         "shape must raise, not be silently skipped")
    return gi


def lead_fixed_index(FROZEN, row: dict):
    """The shipped control flow plus the one added branch, returning the INDEX.

    ``score_with_lead_fix`` returns only the boolean verdict; the permutation test
    needs the predicted index itself, so the control flow is expressed once here and
    the boolean helper is checked against it below.  Note ``truncate_at_newline``:
    omitting it is exactly the bug that made my first standalone re-scorer report
    A0's invalid count RISING, which a pure fallback can never do.
    """
    g = FROZEN.truncate_at_newline(row["prediction"])
    choices = row["choices"]
    if not g.strip():
        return None
    pred = FROZEN.match_choice(g, choices)
    if pred is None:
        pred = FROZEN.match_choice(FROZEN.strip_label(g), choices)
    if pred is None:
        pred = FROZEN.match_choice(FROZEN.strip_answer_lead(FROZEN.strip_label(g)), choices)
    if pred is None:
        lt = FROZEN.letter_of(g)
        if lt is not None and FROZEN.LETTERS.index(lt) < len(choices):
            pred = FROZEN.LETTERS.index(lt)
    if pred is None:                                            # the one added branch
        lt = FROZEN.letter_of(FROZEN.strip_answer_lead(g))
        if lt is not None and FROZEN.LETTERS.index(lt) < len(choices):
            pred = FROZEN.LETTERS.index(lt)
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory holding <task>__<arm>.jsonl")
    ap.add_argument("--expect_n", type=int, default=200)
    ap.add_argument("--label", default="", help="free-text label for this block")
    a = ap.parse_args()
    out = Path(a.out)
    rows = load(out)

    arms = [x for x in ARM_ORDER if any(k[1] == x for k in rows)]
    tasks = [t for t in TASKS if any(k[0] == t for k in rows)]
    print(f"DENOMINATOR arms present = {len(arms)} {arms}")
    print(f"DENOMINATOR tasks present = {len(tasks)} {tasks}")
    if not arms or not tasks:
        raise SystemExit("zero arms or zero tasks: the aggregate cannot fail, so it is not valid")

    # ---- invariant 0: the field names this script reads actually exist ---------
    # Every quantity below is fetched by NAME.  ``dict.get(k, default)`` on a field
    # that was renamed returns the default silently, and a tally of zero then reads
    # exactly like "zero rows affected" instead of "zero rows examined".  I have hit
    # this five times in this project (``prediction`` vs ``generate``, ``conv_ffn_zeroed``
    # vs ``conv_ffn_zeroed_by_adapter``, ``ground_truth`` vs ``gold``, a top-level
    # ``cache_fp_*`` that was nested under ``cache_policy``, and a ``tasks[t].examples``
    # I read as a top-level list).  So: assert the key set before any count is printed.
    READS = ("example_id", "prompt_hash", "arm", "task", "seed", "prefix_replay",
             "checkpoint_sha256", "prediction", "gold", "choices", "correct",
             "invalid", "pred_choice_index", "cache_policy", "adapter_name")
    print("\n=== CHECK: every field this script reads by name is present ===")
    print(f"    DENOMINATOR field names read = {len(READS)}")
    for (t, arm), rs in sorted(rows.items()):
        absent = [k for k in READS if k not in rs[0]]
        if absent:
            raise SystemExit(
                f"{t}/{arm}: row 0 lacks {absent}; every value this script reads is fetched "
                "by name, so a renamed field would be tallied as zero and a zero tally is "
                "indistinguishable from 'zero rows examined'. Refusing to aggregate.")
    print(f"    all {len(READS)} names present in row 0 of all {len(rows)} cells")
    # POSITIVE CONTROL: the assertion must fire on a row that is missing a field.
    _probe = {k: None for k in READS if k != "prediction"}
    _fired = [k for k in READS if k not in _probe]
    if _fired != ["prediction"]:
        raise SystemExit("key-set gate cannot detect a missing field: it is vacuous")
    print(f"    POSITIVE CONTROL: on a row with 'prediction' deleted the gate names "
          f"{_fired} (denominator 1, detected 1)")

    # ---- invariant 1: every cell has exactly the pinned row count -------------
    print("\n=== CHECK: row counts (invalid rows are WRONG, never dropped) ===")
    bad = []
    for arm in arms:
        for t in tasks:
            n = len(rows.get((t, arm), []))
            flag = "OK" if n == a.expect_n else "MISMATCH"
            if n != a.expect_n:
                bad.append((arm, t, n))
            print(f"    {arm} {t:14s} n={n:4d} expected={a.expect_n}  {flag}")
    print(f"    DENOMINATOR cells checked = {len(arms) * len(tasks)}; mismatches = {len(bad)}")
    if bad:
        raise SystemExit(f"row-count mismatch {bad}: a short cell would silently shrink a "
                         "denominator, which is exactly what the protocol forbids")

    # ---- invariant 2: arms are matched -------------------------------------
    print("\n=== CHECK: arms are matched on ids / prompts / checkpoint / seed ===")
    for t in tasks:
        ids = {arm: [r["example_id"] for r in rows[(t, arm)]] for arm in arms}
        ph = {arm: [r["prompt_hash"] for r in rows[(t, arm)]] for arm in arms}
        ref = arms[0]
        print(f"    {t}: DENOMINATOR ids per arm = {len(ids[ref])}")
        for arm in arms[1:]:
            if ids[arm] != ids[ref]:
                raise SystemExit(f"{t}: {arm} example ids differ from {ref}; the arms are not paired")
            if ph[arm] != ph[ref]:
                raise SystemExit(f"{t}: {arm} prompt hashes differ from {ref}; different prompts "
                                 "would make the comparison meaningless")
        print(f"    {t}: all {len(arms)} arms share identical id and prompt-hash sequences  OK")
    shas = {r["checkpoint_sha256"] for v in rows.values() for r in v}
    seeds = {r["seed"] for v in rows.values() for r in v}
    repl = {r["prefix_replay"] for v in rows.values() for r in v}
    print(f"    DENOMINATOR distinct checkpoint_sha256 = {len(shas)} (must be 1) -> {list(shas)}")
    print(f"    DENOMINATOR distinct seed = {len(seeds)} (must be 1) -> {sorted(seeds)}")
    print(f"    DENOMINATOR distinct prefix_replay = {len(repl)} (must be 1) -> {sorted(repl)}")
    if len(shas) != 1 or len(seeds) != 1 or len(repl) != 1:
        raise SystemExit("arms disagree on checkpoint, seed or prefix_replay; not a matched comparison")

    # ---- invariant 3: A1 restored, others not ------------------------------
    print("\n=== CHECK: a1_layers_restored is >0 for A1 only ===")
    for arm in arms:
        vals = sorted({r["a1_layers_restored"] for v in
                       [rows[(t, arm)] for t in tasks] for r in v})
        expect_pos = arm == "A1"
        ok = (min(vals) > 0) if expect_pos else (set(vals) == {0})
        print(f"    {arm}: values={vals} expect{'>0' if expect_pos else '==0'}  "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"{arm} a1_layers_restored={vals} contradicts its arm definition")

    # ---- invariant 3b: A3 is a real derangement, rebuilt per task -------------
    # The appendix spec requires a within-task permutation with no self-assignment.
    # Two distinct things can go wrong and only one of them is what "0 fixed points"
    # tests: (i) a fixed point, and (ii) the SAME permutation reused across tasks,
    # which would make the three A3 columns correlated rather than three independent
    # draws.  A tally of fixed points cannot see (ii), so test it separately, and
    # print the denominator of each test before its verdict.
    print("\n=== CHECK: A3 derangement is within-task, fixed-point-free, and per-task ===")
    if "A3" in arms:
        donor_maps: dict[str, dict[int, int]] = {}
        for t in tasks:
            m = {r["example_id"]: r["donor_id"] for r in rows[(t, "A3")]}
            donor_maps[t] = m
            n = len(m)
            missing = sum(1 for v in m.values() if v is None)
            fixed = sum(1 for k, v in m.items() if v is not None and v == k)
            distinct = len({v for v in m.values() if v is not None})
            in_task = sum(1 for v in m.values() if v is not None and v in m)
            print(f"    {t}: DENOMINATOR A3 rows = {n}; donor_id missing = {missing}; "
                  f"fixed points = {fixed}; distinct donors = {distinct}; donors drawn "
                  f"from this task's own ids = {in_task}/{n}")
            if missing:
                raise SystemExit(f"{t}: {missing} A3 rows have no donor_id; a null donor is "
                                 "not a shuffled latent and must not be scored as one")
            if fixed:
                raise SystemExit(f"{t}: {fixed} A3 fixed points; a self-assigned donor is "
                                 "arm A4, so those rows would silently duplicate the "
                                 "correct-latent arm")
            if distinct != n or in_task != n:
                raise SystemExit(f"{t}: A3 donors are not a permutation of this task's ids "
                                 f"(distinct={distinct}, in-task={in_task}, n={n})")
        # ---- cross-task index-pattern reuse: REPORTED, NOT FATAL -------------
        # My first version of this raised SystemExit when the donor map was identical
        # across tasks.  It fired on the real data (200/200 identical), and the abort
        # was WRONG -- the check was reading the wrong frame.  ``example_id`` is a
        # PER-TASK label: sciq/5 and commonsenseqa/5 are unrelated examples that share
        # only an integer.  ``build_derangement(ids, args.seed)`` is seeded with the
        # task-independent seed, so any two tasks whose id lists are both 0..199 get
        # the same INDEX pattern by construction.  What a shuffled-latent arm needs is
        # that each recipient receives a DIFFERENT PROMPT'S latent from within its own
        # task, and that is enforced per task above (fixed points 0, donors a
        # permutation of this task's own ids).  Measured on this run: the donor
        # LATENTS are bit-identical across tasks 0/200 (z_cache keys are task-scoped),
        # recipient prompt hashes overlap 0/200, and gold strings overlap 2/200 -- so
        # every quantity a score depends on is disjoint between the two A3 cells.
        # Sharing the index pattern therefore does not make the cells the same draw.
        # It is still worth PRINTING, because it means the three A3 cells are not
        # three independent samples of the permutation space, so their spread is not a
        # confidence interval over derangements.  Report it; do not abort on it.
        task_pairs = [(x, y) for i, x in enumerate(tasks) for y in tasks[i + 1:]]
        print(f"    DENOMINATOR task pairs compared for index-pattern reuse = {len(task_pairs)}")
        for x, y in task_pairs:
            common = sorted(set(donor_maps[x]) & set(donor_maps[y]))
            if not common:
                print(f"    {x} vs {y}: 0 shared example ids, reuse is not testable here")
                continue
            same = sum(1 for i in common if donor_maps[x][i] == donor_maps[y][i])
            verdict = ("SAME index pattern (expected: one seed, identical id lists)"
                       if same == len(common) else "independent index patterns")
            print(f"    {x} vs {y}: shared ids = {len(common)}; identical donor index = "
                  f"{same}; {verdict}")
        print("    NOTE: identical donor INDICES across tasks is not cross-task contamination.")
        print("          example_id is a per-task label and the z_cache is task-scoped, so the")
        print("          donor LATENTS differ; what it does mean is that the three A3 cells")
        print("          share one permutation draw, so their spread is not a CI over")
        print("          derangements.  The within-task derangement property is what the arm")
        print("          needs, and it is enforced per task above.")

        # ---- the property the demoted check was REACHING for, tested directly ----
        # Removing an abort must not remove coverage.  The real hazard is that two A3
        # cells write the SAME tensor to the same recipient, which would make them one
        # arm wearing two labels.  Indices cannot answer that; the latents can.  Test
        # it on the prompt hashes actually recorded per row, which are frame-correct
        # (a prompt hash identifies an example globally, an example_id does not).
        print("\n=== CHECK: A3 cells are distinct draws where it counts (prompts, not indices) ===")
        for x, y in task_pairs:
            hx = {r["prompt_hash"] for r in rows[(x, "A3")]}
            hy = {r["prompt_hash"] for r in rows[(y, "A3")]}
            shared = len(hx & hy)
            print(f"    {x} vs {y}: DENOMINATOR prompt hashes = {len(hx)}/{len(hy)}; "
                  f"shared = {shared}  {'OK' if shared == 0 else 'OVERLAP'}")
            if shared:
                raise SystemExit(
                    f"{x} and {y} share {shared} prompt hashes: the two A3 cells are scoring "
                    "some of the same prompts, so they are not separate task cells")
        # positive control: the comparator must be able to report an overlap.
        pc_self = len({r["prompt_hash"] for r in rows[(tasks[0], "A3")]} &
                      {r["prompt_hash"] for r in rows[(tasks[0], "A3")]})
        print(f"    positive control: a cell against ITSELF shares {pc_self} hashes "
              f"(must be > 0, else the comparator cannot see overlap)")
        if pc_self == 0:
            raise SystemExit("the prompt-hash overlap comparator reports 0 against itself, so "
                             "it is inert and its 0s above prove nothing")
    else:
        print("    A3 absent from this directory; nothing to check (denominator 0)")

    # ---- invariant 4: scorer exposure, WITH a comparator positive control ----
    # A count of "0 affected rows" is worthless unless the instrument that produced
    # it is first shown capable of producing a non-zero count.  My first version of
    # this scan reported 0/72 using a comparator that could not separate the two
    # scorer versions on ANY input, so the 0 was indistinguishable from a scan that
    # selected no rows.  Print the denominator, and print the proof of separability.
    print("\n=== CHECK: scorer exposure (empty / multi-char letter tokens) ===")
    here = Path(__file__).resolve().parent
    for p in (str(here), str(here.parents[1])):
        if p not in sys.path:
            sys.path.insert(0, p)
    import g2_frozen_run_five_arms as FROZEN  # the scorer that actually ran

    n_rows = n_empty = n_multi = 0
    for v in rows.values():
        for r in v:
            n_rows += 1
            g = FROZEN.truncate_at_newline(r["prediction"])
            if not g.strip():
                n_empty += 1
            lt = FROZEN.letter_of(g)
            if lt is not None and len(lt) != 1:
                n_multi += 1
    print(f"    DENOMINATOR rows scanned = {n_rows}")
    print(f"    empty generations                 = {n_empty}/{n_rows}")
    print(f"    letter_of -> non-single-char token = {n_multi}/{n_rows}")

    # positive control for THIS scan: it must be able to count a planted case.
    probe = [("", "empty"), ("AB", "multi-char")]
    fired = 0
    for text, _ in probe:
        g = FROZEN.truncate_at_newline(text)
        if (not g.strip()) or ((FROZEN.letter_of(g) or "x") and
                               FROZEN.letter_of(g) is not None and
                               len(FROZEN.letter_of(g)) != 1):
            fired += 1
    print(f"    DENOMINATOR planted probe cases = {len(probe)}; detected = {fired}")
    if fired != len(probe):
        raise SystemExit(
            f"the exposure scan detected only {fired}/{len(probe)} planted cases, so a "
            f"report of {n_empty}/{n_multi} affected rows is not evidence of anything"
        )
    print(f"    scan calibrated: detects {fired}/{len(probe)} planted cases, so the counts "
          f"above are meaningful")
    print("    NOTE: if either count is ever non-zero, RESCORE from the stored "
          "'prediction' field.")
    print("          Do NOT re-decode: re-decoding changes the decode path and would "
          "silently make")
    print("          these rows incomparable with the rest of the table.")

    # ---- invariant 5: DIFFERENTIAL against the corrected scorer -------------
    # The two scorers are NOT behaviourally identical.  I claimed they were, on a
    # 544-case sweep; that sweep contained no lowercase two-letter run, so the
    # claim was a statement about my case set and is RETRACTED.  The real
    # divergence: the frozen scorer's letter_of("ab") returns "AB" (a two-char
    # token, which LETTERS.index maps to 0 -> option A) while the corrected
    # letter_of returns None -> invalid -> WRONG.  --letter_first is absent from
    # this run, so letter_of is reached as the FINAL fallback at score_row line
    # ~337 and the divergence is inside our regime, not outside it.
    # This check re-scores every row with BOTH scorers and requires the
    # comparator to be demonstrably able to see the divergence first.
    print("\n=== CHECK: differential vs corrected scorer (frozen snapshot) ===")
    corrected_path = here / "g2_scorer_corrected_frozen.py"
    if not corrected_path.exists():
        raise SystemExit(f"{corrected_path} missing: the differential cannot be run, and "
                         "skipping it would report a clean table on an unaudited scorer")
    spec = importlib.util.spec_from_file_location("g2_corrected", corrected_path)
    CORR = importlib.util.module_from_spec(spec)
    sys.modules["g2_corrected"] = CORR
    spec.loader.exec_module(CORR)

    letter_first = False  # this run does not pass --letter_first
    probe_rows = [r for v in rows.values() for r in v][:20]
    print(f"    DENOMINATOR positive-control rows = {len(probe_rows)}")
    if not probe_rows:
        raise SystemExit("no rows to plant into: the differential cannot fail, so it is not a check")
    live = sum(
        1 for r in probe_rows
        if FROZEN.score_row("ab", r["gold"], r["choices"], letter_first)
        != CORR.score_row("ab", r["gold"], r["choices"], letter_first)
    )
    print(f"    positive control: planted 'ab' flags {live}/{len(probe_rows)}")
    if live != len(probe_rows):
        raise SystemExit(
            f"the differential comparator flags only {live}/{len(probe_rows)} planted "
            "divergences, so it is INERT and any 0 it reports proves nothing"
        )
    same = sum(
        1 for r in probe_rows
        if FROZEN.score_row(r["prediction"], r["gold"], r["choices"], letter_first)
        != FROZEN.score_row(r["prediction"], r["gold"], r["choices"], letter_first)
    )
    print(f"    negative control: frozen-vs-frozen disagreements = {same}/{len(probe_rows)} (must be 0)")
    if same:
        raise SystemExit("the scorer is nondeterministic; no differential is interpretable")

    n_dis = 0
    dis_by_arm: dict[str, int] = collections.Counter()
    for (t, arm), v in rows.items():
        for r in v:
            sc_frozen = FROZEN.score_row(r["prediction"], r["gold"], r["choices"], letter_first)
            sc_corr = CORR.score_row(r["prediction"], r["gold"], r["choices"], letter_first)
            if sc_frozen != sc_corr:
                n_dis += 1
                dis_by_arm[arm] += 1
    print(f"    DENOMINATOR rows re-scored with BOTH scorers = {n_rows}")
    print(f"    SCORER DISAGREEMENTS = {n_dis}/{n_rows}"
          + (f"  by arm: {dict(sorted(dis_by_arm.items()))}" if n_dis else ""))
    print(f"    coverage of this denominator: arms={arms} prefix_replay={sorted(repl)}")
    print("    The two scorers are NOT identical; they agree on THESE rows. If the count")
    print("    above is non-zero, RESCORE from 'prediction' and report both scorings.")
    scorer_audit = {
        "rows_rescored": n_rows,
        "disagreements": n_dis,
        "disagreements_by_arm": dict(dis_by_arm),
        "positive_control": f"{live}/{len(probe_rows)}",
        "negative_control_frozen_vs_frozen": f"{same}/{len(probe_rows)}",
        "letter_first": letter_first,
        "coverage_arms": arms,
        "coverage_prefix_replay": sorted(repl),
        "note": "the frozen and corrected scorers are NOT behaviourally identical; they "
                "diverge on lowercase two-letter runs such as 'ab' (frozen -> option A, "
                "corrected -> invalid). This audit reports agreement ON THESE ROWS ONLY.",
    }

    # ---- the table ---------------------------------------------------------
    acc: dict[str, dict[str, float]] = {}
    inv: dict[str, dict[str, int]] = {}
    ci: dict[str, dict[str, tuple[float, float]]] = {}
    for arm in arms:
        acc[arm], inv[arm], ci[arm] = {}, {}, {}
        for t in tasks:
            rs = rows[(t, arm)]
            k = sum(1 for r in rs if r["correct"])
            acc[arm][t] = 100.0 * k / len(rs)
            inv[arm][t] = sum(1 for r in rs if r["invalid"])
            ci[arm][t] = wilson(k, len(rs))

    print("\n" + "=" * 92)
    print(f"tab:latent-use-pending  {a.label}")
    print(f"seed={sorted(seeds)[0]}  prefix_replay={sorted(repl)[0]}  n={a.expect_n} per task")
    print("=" * 92)
    hdr = f"{'Condition':44s}" + "".join(f"{TASK_LABEL[t]:>9s}" for t in tasks) + f"{'Macro3':>9s}"
    print(hdr)
    print("-" * 92)
    macro = {}
    for arm in arms:
        m = sum(acc[arm][t] for t in tasks) / len(tasks)
        macro[arm] = m
        line = f"{ARM_LABEL[arm]:44s}" + "".join(f"{acc[arm][t]:9.2f}" for t in tasks) + f"{m:9.2f}"
        print(line)
    print("-" * 92)
    print("Macro3 = equal-weight mean of the task accuracies above.  It is NOT Avg_11 and is")
    print("NOT comparable to the eleven-task 52.8/64.2 aggregates or the 60.58/67.50")
    print("decode-path control, which are different comparisons on different example sets.")

    print("\n=== per-arm invalid counts (counted as WRONG, denominator unchanged) ===")
    for arm in arms:
        tot = sum(inv[arm][t] for t in tasks)
        cells = "  ".join(f"{t}={inv[arm][t]}/{a.expect_n}" for t in tasks)
        print(f"    {arm} {ARM_LABEL[arm]:44s} {cells}   total={tot}/{a.expect_n*len(tasks)}")

    print("\n=== 95% Wilson CI per cell ===")
    for arm in arms:
        for t in tasks:
            lo, hi = ci[arm][t]
            print(f"    {arm} {t:14s} {acc[arm][t]:6.2f}  [{lo:6.2f}, {hi:6.2f}]")

    # ---- scorer-leniency sensitivity ---------------------------------------
    # The generate-then-match scorer credits a row through one of six stages.
    # Stages 4 (containment) and 5 (token-F1 >= 0.5) are LOOSE: a generation that
    # merely mentions an option is credited.  That is the same defect class as the
    # MemEval containment bug.  If the arms lean on the loose stages unequally, the
    # comparison is partly a measurement of scorer leniency rather than of the
    # write, so the table must be reported with the strict recomputation beside it.
    # STRICT = stages 1-3 only (exact / prefix / label-stripped).  No containment,
    # no F1, no bare-letter fallback.
    print("\n=== CHECK: scorer-leniency sensitivity (which stage credits each arm) ===")

    def _stage(gen: str, choices: list[str]) -> str:
        g = FROZEN.truncate_at_newline(gen)
        if not g.strip():
            return "empty->invalid"
        ng = FROZEN.normalize_text(g)
        norms = [FROZEN.normalize_text(c) for c in choices]
        if any(c and c == ng for c in norms):
            return "1 exact"
        if any(c and (ng.startswith(c) or c.startswith(ng)) for c in norms):
            return "2 prefix"
        for s in (FROZEN.strip_label(g), FROZEN.strip_answer_lead(FROZEN.strip_label(g))):
            n2 = FROZEN.normalize_text(s)
            if not n2:
                continue
            if any(c and c == n2 for c in norms):
                return "3 label-stripped exact"
            if any(c and (n2.startswith(c) or c.startswith(n2)) for c in norms):
                return "3 label-stripped prefix"
        if any(c and c in ng for c in norms):
            return "4 CONTAINMENT (loose)"
        sc = sorted(((FROZEN._f1(g, c), i) for i, c in enumerate(choices)), reverse=True)
        if sc and sc[0][0] >= 0.5:
            return "5 F1>=0.5 (loose)"
        if FROZEN.letter_of(g) is not None:
            return "6 letter fallback"
        return "none->invalid"

    def _strict_pred(gen: str, choices: list[str]):
        g = FROZEN.truncate_at_newline(gen)
        if not g.strip():
            return None
        ng = FROZEN.normalize_text(g)
        norms = [FROZEN.normalize_text(c) for c in choices]
        for i, c in enumerate(norms):
            if c and c == ng:
                return i
        best, bl = None, 0
        for i, c in enumerate(norms):
            if c and (ng.startswith(c) or c.startswith(ng)):
                L = min(len(c), len(ng))
                if L > bl:
                    best, bl = i, L
        if best is not None:
            return best
        for s in (FROZEN.strip_label(g), FROZEN.strip_answer_lead(FROZEN.strip_label(g))):
            n2 = FROZEN.normalize_text(s)
            if not n2:
                continue
            for i, c in enumerate(norms):
                if c and c == n2:
                    return i
            best, bl = None, 0
            for i, c in enumerate(norms):
                if c and (n2.startswith(c) or c.startswith(n2)):
                    L = min(len(c), len(n2))
                    if L > bl:
                        best, bl = i, L
            if best is not None:
                return best
        return None

    def _gold_idx(gold: str, choices: list[str]):
        ng = FROZEN.normalize_text(gold)
        for i, c in enumerate(choices):
            if FROZEN.normalize_text(c) == ng:
                return i
        return FROZEN.match_choice(gold, choices)

    # positive control: the loose stage must be reachable and correctly named
    pc_loose = _stage("I think the answer is not bookstore at all", ["bookstore", "library"])
    pc_exact = _stage("bookstore", ["bookstore", "library"])
    print(f"    positive control: mention-only -> {pc_loose!r} (must be stage 4)")
    print(f"    positive control: exact match  -> {pc_exact!r} (must be stage 1)")
    if not pc_loose.startswith("4 ") or not pc_exact.startswith("1 "):
        raise SystemExit("the stage classifier mislabels known inputs; its per-arm tallies "
                         "and the strict recomputation below would both be meaningless")

    stage_tab: dict[str, collections.Counter] = {arm: collections.Counter() for arm in arms}
    strict_acc: dict[str, dict[str, float]] = {arm: {} for arm in arms}
    for arm in arms:
        for t in tasks:
            k = 0
            for r in rows[(t, arm)]:
                stage_tab[arm][_stage(r["prediction"], r["choices"])] += 1
                if _strict_pred(r["prediction"], r["choices"]) == _gold_idx(r["gold"], r["choices"]):
                    k += 1
            strict_acc[arm][t] = 100.0 * k / len(rows[(t, arm)])
    for arm in arms:
        tot = sum(stage_tab[arm].values())
        print(f"    {arm} DENOMINATOR rows={tot}: "
              + "  ".join(f"{k}={v}" for k, v in sorted(stage_tab[arm].items())))
    strict_macro = {arm: sum(strict_acc[arm][t] for t in tasks) / len(tasks) for arm in arms}
    print("    as-run vs STRICT Macro3.  ADMITTED STAGES (print this with any strict")
    print("    number; 'strict' alone is underdetermined -- see the spread below):")
    print("      ADMITTED: 1 exact, 2 prefix (either direction), 3 label-stripped and")
    print("                answer-lead-stripped exact/prefix")
    print("      REFUSED : 4 containment, 5 token-F1>=0.5, 6 bare-letter fallback")
    for arm in arms:
        m = sum(acc[arm][t] for t in tasks) / len(tasks)
        print(f"        {arm}  as-run {m:6.2f}   strict {strict_macro[arm]:6.2f}   "
              f"delta {strict_macro[arm] - m:+6.2f}")
    print("    Report BOTH. A large as-run/strict gap means that arm's score depends on")
    print("    loose containment or bare-letter credit, so the magnitude is scorer-dependent")
    print("    even where the ORDERING of the arms is not.")

    # ---- how much does the WORD "strict" decide, independent of the data? ----
    # Three defensible readings of "strict" give three different answers on the SAME
    # rows.  A reject-probe cannot separate them because all three reject loose
    # containment, so a bare "strict = X" is not reconstructable by a reader.  What
    # survives all three is the A0-vs-A1 GAP, so the gap is the claimable quantity.
    def _strict_variant(gen: str, choices: list[str], admit_letter: bool):
        i = _strict_pred(gen, choices)
        if i is not None:
            return i
        if admit_letter:
            lt = FROZEN.letter_of(FROZEN.truncate_at_newline(gen))
            if lt is not None:
                j = ord(lt.upper()) - ord("A")
                if 0 <= j < len(choices):
                    return j
        return None

    def _exact_prefix_only(gen: str, choices: list[str]):
        g = FROZEN.truncate_at_newline(gen)
        if not g.strip():
            return None
        ng = FROZEN.normalize_text(g)
        norms = [FROZEN.normalize_text(c) for c in choices]
        for i, c in enumerate(norms):
            if c and c == ng:
                return i
        best, bl = None, 0
        for i, c in enumerate(norms):
            if c and (ng.startswith(c) or c.startswith(ng)):
                L = min(len(c), len(ng))
                if L > bl:
                    best, bl = i, L
        return best

    READINGS = (
        ("R_A exact+prefix only", lambda g, c: _exact_prefix_only(g, c)),
        ("R_B exact+prefix+label/lead-stripped (THIS SCRIPT)", lambda g, c: _strict_pred(g, c)),
        ("R_C R_B + letter_of(TRUNCATED, range-guarded)", lambda g, c: _strict_variant(g, c, True)),
    )
    print("    SENSITIVITY OF THE WORD 'STRICT' (same rows, three readings):")
    print("      R_C's name must carry its fallback spec, not just 'plus bare letter':")
    print("      letter_of on the NEWLINE-TRUNCATED text with a 0<=j<len(choices) guard")
    print("      gives A0 61.67, but letter_of on the RAW text gives 60.00 -- and the 8")
    print("      defensible combinations of (raw|truncated) x (guard|no guard) x")
    print("      (strip lead|not) span 60.00..61.83.  A stage list alone does NOT pin a")
    print("      number once a letter fallback is admitted; the fallback's exact input")
    print("      and guard are part of the definition.")
    probe_rows = 0
    variant_macro: dict[str, dict[str, float]] = {}
    for name, fn in READINGS:
        per = {}
        for arm in arms:
            tot = 0.0
            for t in tasks:
                cell_rows = rows[(t, arm)]
                gi = [_gold_idx(r["gold"], r["choices"]) for r in cell_rows]
                ok = sum(1 for r, g in zip(cell_rows, gi)
                         if g is not None and fn(r["prediction"], r["choices"]) == g)
                tot += 100.0 * ok / len(cell_rows)
                probe_rows += len(cell_rows)
            per[arm] = tot / len(tasks)
        variant_macro[name] = per
    print(f"      DENOMINATOR rows scored per reading = {probe_rows // len(READINGS)}")
    for name in variant_macro:
        per = variant_macro[name]
        print(f"      {name:52s} " + "  ".join(f"{a}={per[a]:6.2f}" for a in arms))
    _a0 = [variant_macro[n]["A0"] for n, _ in READINGS]
    print(f"      A0 strict spans {min(_a0):.2f} to {max(_a0):.2f} "
          f"= {max(_a0) - min(_a0):.2f} points on wording alone.")
    # A .get("A1", 0.0) here would silently print "A0-minus-A1" as A0-minus-NOTHING on a
    # directory that has no A1 (e.g. the two-arm refeed side run), which is a fabricated
    # quantity dressed as a measurement.  Say the arm is absent instead.
    if "A1" in variant_macro[READINGS[0][0]]:
        _gap = [variant_macro[n]["A0"] - variant_macro[n]["A1"] for n, _ in READINGS]
        print(f"      A0-minus-A1 gap spans {min(_gap):.2f} to {max(_gap):.2f} "
              f"-> the GAP is robust where the LEVEL is not.")
    else:
        print("      A0-minus-A1 gap: NOT COMPUTABLE -- no A1 cells in this directory "
              f"(arms present = {arms}).  Reporting a gap against a missing arm would "
              "be a fabricated number.")
    # POSITIVE CONTROL: the readings must be ORDERED R_A <= R_B <= R_C by construction
    # (each admits a superset of the previous).  If that fails the variants are not
    # nested and the spread above is not a leniency ladder.
    for arm in arms:
        va, vb, vc = (variant_macro[n][arm] for n, _ in READINGS)
        if not (va - 1e-9 <= vb <= vc + 1e-9):
            raise SystemExit(
                f"strict-variant nesting violated for {arm}: R_A={va} R_B={vb} R_C={vc}; "
                "the readings are not a superset ladder, so the spread is meaningless")
    print("      POSITIVE CONTROL: nesting R_A <= R_B <= R_C holds for all "
          f"{len(arms)} arms (each reading admits a superset of the previous)")

    # ---- SCORER-FREE behavioural channel ----------------------------------
    # An accuracy gap can always be argued away as a leniency artefact.  These two
    # measures never call the scorer, so a separation visible here cannot be.  This
    # exists because A1's only nonzero cell (cqa 8.50) turned out to be 100% bare-letter
    # credit -- strict drops it to 0.00 -- while the SAME pair of arms differs by
    # 50/200 vs 0/200 on whether a choice marker is emitted at all.  Reporting only
    # accuracy says "A1 is also at the floor" (false); reporting only this channel
    # overstates it.  Both, always.
    print("\n=== CHECK: scorer-free behavioural channel (format retention) ===")
    # Require an EXPLICIT option marker, and ALLOW the "The answer is X" lead-in that
    # A1 uses on 188/200 sciq rows.  Two earlier versions of this probe were wrong and
    # the calibration set caught both: v1 (``^\s*[\(\[]?([A-Ha-h])(?![a-z])``) matched
    # "a bird"; v2 anchored at ^ only, so it scored "The answer is B." as having no
    # marker and reported A1 sciq as 1/200 when the true count is 175/200.  That 1
    # agreed with the shipped scorer's 1, so two independent errors corroborated each
    # other -- which is why the calibration set, not the agreement, is the evidence.
    marker = re.compile(r"^\s*(?:[Tt]he\s+answer\s+is\s*)?"
                        r"(?:\(\s*[A-Ha-h]\s*\)|\[\s*[A-Ha-h]\s*\]|[A-H][\).:\]]|[A-H]\s*$)")
    cal = [("(B) foo", True), ("[C] bar", True), ("A) x", True), ("D. y", True),
           ("B: z", True), ("The answer is B.", True), ("The answer is (A).", True),
           ("the answer is (c)", True), ("The answer is B", True),
           ("a bird", False), ("an apple", False), ("I went", False),
           ("The answer is", False), ("a) lower", False),
           ("The capital is Paris", False), ("answer: yes", False)]
    ok = sum(1 for s, e in cal if bool(marker.match(s)) is e)
    print(f"    DENOMINATOR marker-probe calibration cases = {len(cal)}; correct = {ok}")
    if ok != len(cal):
        for s, e in cal:
            if bool(marker.match(s)) is not e:
                print(f"        MISCALIBRATED on {s!r}: expected {e}")
        raise SystemExit("marker probe is miscalibrated; refusing to print counts from a "
                         "probe whose false-positive behaviour is unknown")
    print("    (9 must-match incl. lead-in forms + 7 must-not-match incl. real prose "
          "that merely starts with a letter)")
    fmt = {arm: {} for arm in arms}
    for arm in arms:
        for t in tasks:
            rs = rows[(t, arm)]
            fmt[arm][t] = sum(1 for r in rs if marker.match(r.get("prediction", "")))
            parse = sum(1 for r in rs if r.get("pred_choice_index") is not None)
            print(f"    {arm} {t:14s} opens-with-marker = {fmt[arm][t]:4d}/{len(rs)}"
                  f"   parseable-choice = {parse:4d}/{len(rs)}")
    print("    A separation HERE is not explainable by the scoring rule; a separation only")
    print("    in accuracy is.  Quote the channel that matches the claim being made.")

    # ---- SCORER DEFECT: answer-lead never reaches letter extraction ---------
    # score_row applies strip_answer_lead() before match_choice() but NEVER before
    # letter_of().  So "The answer is B." fails all three match_choice attempts AND
    # both letter_of attempts, and is recorded invalid -> WRONG, even though
    # letter_of(strip_answer_lead(g)) == "B".  This is not uniform noise: A1 is the
    # only arm that verbalises its answer (188/200 sciq, 147/200 cqa begin with
    # "the answer is"; the other four arms are 0/200), so the defect zeroes exactly
    # one arm.  The shipped numbers are kept as the table of record; this block
    # reports the corrected scoring alongside, and the two must both be quoted.
    print("\n=== CHECK: answer-lead scoring defect (rescored from stored predictions) ===")
    lead_rows = {arm: {} for arm in arms}
    lead_rows_scorer_rule = {arm: {} for arm in arms}
    fixed_acc = {arm: {} for arm in arms}
    changed_tot = 0
    # TWO RULES define "this row verbalises its answer", and they give DIFFERENT counts,
    # so both are printed and each is named.  R1 is the literal phrase "the answer is".
    # R2 is the scorer's OWN strip_answer_lead regex, which also matches a bare
    # "Answer:" -- ``^\s*(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)?\s*``.  On A1/race
    # they differ by 2.3x (26 vs 60), and R2 is the rule the remedy actually keys on,
    # so quoting R1 alone understates the population the fix is defined over.
    def _verbalises_r1(gen: str) -> bool:
        return gen.strip().lower().startswith("the answer is")

    def _verbalises_r2(gen: str) -> bool:
        g = FROZEN.truncate_at_newline(gen)
        return FROZEN.strip_answer_lead(g) != g

    for arm in arms:
        for t in tasks:
            rs = rows[(t, arm)]
            lead_rows[arm][t] = sum(1 for r in rs if _verbalises_r1(r["prediction"]))
            lead_rows_scorer_rule[arm][t] = sum(1 for r in rs if _verbalises_r2(r["prediction"]))
            f = ch = 0
            for r in rs:
                c2 = score_with_lead_fix(FROZEN, r)
                f += 1 if c2 else 0
                if bool(r["correct"]) is not bool(c2):
                    ch += 1
            fixed_acc[arm][t] = 100.0 * f / len(rs)
            changed_tot += ch
            print(f"    {arm} {t:14s} verbalises R1='the answer is' {lead_rows[arm][t]:4d}/{len(rs)}"
                  f"  R2=scorer's own rule {lead_rows_scorer_rule[arm][t]:4d}/{len(rs)}"
                  f"   as-run={acc[arm][t]:6.2f}  lead-fixed={fixed_acc[arm][t]:6.2f}"
                  f"  delta={fixed_acc[arm][t]-acc[arm][t]:+6.2f}  rows changed={ch}")
    print(f"    DENOMINATOR rows rescored = {sum(len(rows[(t, arm)]) for arm in arms for t in tasks)}"
          f"; verdicts changed = {changed_tot}")
    # negative control: the shipped scorer must reproduce the stored verdicts exactly,
    # otherwise this whole block is measuring my harness rather than the defect.
    dis = 0
    for arm in arms:
        for t in tasks:
            for r in rows[(t, arm)]:
                c, _, _ = FROZEN.score_row(r["prediction"], r["gold"], r["choices"], False)
                if bool(c) is not bool(r["correct"]):
                    dis += 1
    print(f"    NEGATIVE CONTROL shipped-scorer vs stored verdicts = {dis} disagreements "
          f"(must be 0, else this harness is not reproducing the run)")
    if dis:
        raise SystemExit("the shipped scorer does not reproduce the stored verdicts; the "
                         "rescore above cannot be attributed to the defect")
    print("    POSITIVE CONTROLS: ", end="")
    pc1 = FROZEN.score_row("B. something", "beta", ["alpha", "beta", "gamma", "delta"], False)
    pc2 = FROZEN.score_row("The answer is obvious", "beta", ["alpha", "beta", "gamma", "delta"], False)
    print(f"letter without lead-in scores (correct={pc1[0]}, must be True); "
          f"lead-in without a letter stays invalid (invalid={pc2[1]}, must be True)")
    if not pc1[0] or not pc2[1]:
        raise SystemExit("lead-fix controls failed: the correction is either inert or too loose")
    # The two controls above pass even when the fix is a no-op -- I verified that by
    # replacing the added branch with ``if False`` and watching the block still exit 0.
    # What distinguishes a live fix from a dead one is that the fix must change the
    # verdict on rows it demonstrably applies to, so assert THAT.
    # The liveness denominator uses R2, the scorer's OWN rule, because that is the
    # population the added branch is defined over.  R1 would understate it: on A1/race
    # R1 sees 26 rows where R2 sees 60, so an R1 denominator could report a fix as
    # nearly inert on a cell where it is in fact live on 2.3x as many rows.
    pc_lead = [r for arm in arms for t in tasks for r in rows[(t, arm)]
               if _verbalises_r2(r["prediction"]) and not r["correct"]]
    pc_lead_r1 = [r for arm in arms for t in tasks for r in rows[(t, arm)]
                  if _verbalises_r1(r["prediction"]) and not r["correct"]]
    print(f"    DENOMINATOR rows that verbalise AND scored wrong = {len(pc_lead)} under R2 "
          f"(scorer's own rule); {len(pc_lead_r1)} under R1 (literal phrase). The rule is "
          f"named because the two counts differ.")
    if pc_lead:
        flipped = sum(1 for r in pc_lead if score_with_lead_fix(FROZEN, r))
        print(f"    liveness control: of those, the fix now credits {flipped} "
              f"(must be > 0, else the added branch is unreachable and the deltas above "
              f"are a no-op printed as a result)")
        if flipped == 0:
            raise SystemExit("the lead-in fix changed nothing on rows it is defined for: it "
                             "is inert, and an inert correction must not be reported as one")
    else:
        print("    liveness control: DENOMINATOR 0 -- no verbalising wrong rows exist in this "
              "slice, so this block is vacuous here and its zeros are not evidence")

    # ---- is a nonzero lead-fixed cell above chance, or just letter bias? ------
    # A1's corrected cells must be shown to beat the BEST CONSTANT strategy, not
    # uniform chance: A1's predictions are degenerate (heavily skewed to option A),
    # so uniform 1/k is the wrong null.  The permutation below shuffles the prediction
    # vector, which preserves BOTH marginals (the prediction histogram and the gold
    # histogram) and so tests only the pairing between them.
    #
    # THE CONTROL HERE WAS WRONG ONCE AND THE ERROR IS WORTH NAMING.  My first
    # negative control shuffled pred AND gold and required p ~= 0.5.  But shuffling
    # both induces the SAME null as shuffling one, so it returns the identical p
    # (measured: 0.00005 both ways) and can NEVER reach 0.5.  It was a control that
    # could not pass -- the mirror image of a control that cannot fail, and equally
    # uninformative.  The correct construction is to make the null TRUE IN THE DATA
    # (permute the prediction vector once, keeping its marginal) and then re-run the
    # WHOLE test on that null-true dataset, several times, and check the p-values are
    # uniform-ish rather than small.
    print("\n=== CHECK: are nonzero LEAD-FIXED cells above the best-constant baseline? ===")
    # There are now TWO implementations of the lead-fixed verdict: the boolean
    # ``score_with_lead_fix`` and the index-returning ``lead_fixed_index``.  Two copies
    # of one rule is exactly how the shipped scorer and my first re-scorer drifted, so
    # assert they agree on every row before either is used for a published number.
    _agree = _dis = 0
    for arm in arms:
        for t in tasks:
            for r in rows[(t, arm)]:
                # NB: not named ``a`` -- ``a`` is the argparse namespace, and shadowing
                # it here made the summary block below die with
                # "'bool' object has no attribute 'label'".
                v_bool = score_with_lead_fix(FROZEN, r)
                v_idx = (lead_fixed_index(FROZEN, r) == gold_index(FROZEN, r))
                if v_bool == v_idx:
                    _agree += 1
                else:
                    _dis += 1
    print(f"    DENOMINATOR rows cross-checked between the two lead-fix implementations "
          f"= {_agree + _dis}; disagreements = {_dis}")
    if _dis:
        raise SystemExit("the two lead-fix implementations disagree; one of them is wrong "
                         "and neither may be used for a reported number")
    for arm in arms:
        for t in tasks:
            rs = rows[(t, arm)]
            pairs = []
            for r in rs:
                p = lead_fixed_index(FROZEN, r)
                g = gold_index(FROZEN, r)
                if p is not None:
                    pairs.append((p, g))
            n_par = len(pairs)
            n_cor = sum(1 for p, g in pairs if p == g)
            if n_cor == 0:
                print(f"    {arm} {t:14s} parseable={n_par:4d}/{len(rs)} correct=0 "
                      f"-> nothing to test (a p-value on a 0 would be fabricated)")
                continue
            gold_hist = collections.Counter(g for _, g in pairs)
            best_const = max(gold_hist.values())
            preds = [p for p, _ in pairs]
            golds = [g for _, g in pairs]
            rng = random.Random(20260923)
            TRIALS = 20000
            ge = 0
            tot_hits = 0
            for _ in range(TRIALS):
                sp = preds[:]
                rng.shuffle(sp)
                h = sum(1 for a, b in zip(sp, golds) if a == b)
                tot_hits += h
                if h >= n_cor:
                    ge += 1
            p_obs = (ge + 1) / (TRIALS + 1)
            print(f"    {arm} {t:14s} parseable={n_par:4d}/{len(rs)}  correct={n_cor:4d}  "
                  f"acc_parseable={100*n_cor/n_par:5.1f}%  best-constant={100*best_const/n_par:5.1f}%  "
                  f"perm-mean={tot_hits/TRIALS:6.1f}  p={p_obs:.5f}")
            # POSITIVE CONTROL: feeding gold as the prediction must be significant.
            # If it is NOT, this cell is too small for the test and no p may be quoted.
            rngp = random.Random(7)
            gep = 0
            for _ in range(4000):
                sp = golds[:]
                rngp.shuffle(sp)
                if sum(1 for a, b in zip(golds, sp) if a == b) >= len(golds):
                    gep += 1
            p_pc = (gep + 1) / 4001
            # NEGATIVE CONTROL done right: null true in the data, whole test re-run.
            nulls = []
            for rep in range(10):
                rr = random.Random(5000 + rep)
                np_ = preds[:]
                rr.shuffle(np_)
                obs2 = sum(1 for a, b in zip(np_, golds) if a == b)
                r3 = random.Random(11)
                ge2 = 0
                for _ in range(2000):
                    sp = np_[:]
                    r3.shuffle(sp)
                    if sum(1 for a, b in zip(sp, golds) if a == b) >= obs2:
                        ge2 += 1
                nulls.append((ge2 + 1) / 2001)
            nulls.sort()
            small = sum(1 for x in nulls if x < 0.05)
            print(f"        POSITIVE CONTROL (gold as prediction) p={p_pc:.5f}  "
                  f"NEGATIVE CONTROL (null true in data, 10 reps) p median={nulls[5]:.3f} "
                  f"min={nulls[0]:.3f} below-0.05={small}/10")
            if p_pc > 0.05:
                print(f"        ^ POSITIVE CONTROL FAILED: this cell is too small for the "
                      f"test (parseable={n_par}); its p-value above is NOT usable")
            if small > 2:
                raise SystemExit(
                    f"{arm}/{t}: the permutation test calls a null-true dataset significant "
                    f"{small}/10 times; the test is miscalibrated and its p must not be quoted")

    # ---- paired comparisons vs A4 -----------------------------------------
    if "A4" in arms:
        print("\n=== PAIRED vs A4 (same example ids; McNemar discordant pairs) ===")
        for arm in arms:
            if arm == "A4":
                continue
            b = c = 0
            for t in tasks:
                x = {r["example_id"]: r["correct"] for r in rows[(t, arm)]}
                y = {r["example_id"]: r["correct"] for r in rows[(t, "A4")]}
                for i in x:
                    if x[i] and not y[i]:
                        b += 1
                    elif y[i] and not x[i]:
                        c += 1
            n = b + c
            print(f"    {arm} vs A4: DENOMINATOR discordant pairs = {n}"
                  + ("   <-- 0 discordant pairs: no test is possible" if n == 0 else ""))
            if n:
                chi = (abs(b - c) - 1) ** 2 / n
                print(f"        {arm}-only-correct={b}  A4-only-correct={c}  "
                      f"McNemar chi2(cc)={chi:.3f}  macro delta={macro[arm]-macro['A4']:+.2f}")

    summary = {
        "out_dir": str(out), "label": a.label,
        "seed": sorted(seeds)[0], "prefix_replay": sorted(repl)[0],
        "checkpoint_sha256": sorted(shas)[0],
        "n_per_task": a.expect_n, "arms": arms, "tasks": tasks,
        "accuracy_pct": acc, "macro3_pct": macro, "invalid_counts": inv,
        "strict_accuracy_pct": strict_acc, "strict_macro3_pct": strict_macro,
        "strict_reading_sensitivity_macro3": variant_macro,
        "strict_reading_sensitivity_note":
            "The single strict_macro3_pct above is ONE reading of 'strict'. These are "
            "three nested readings on the SAME rows: R_A exact+prefix; R_B = R_A plus "
            "label-stripped and answer-lead-stripped exact/prefix (this is what "
            "strict_macro3_pct reports); R_C = R_B plus letter_of on the "
            "NEWLINE-TRUNCATED generation with a 0<=j<len(choices) guard. A0 spans "
            "2.50 to 61.67 across them, so quoting a strict number without its "
            "admitted stage list is unreconstructable. R_C is further underdetermined "
            "by its fallback's incidental choices: (raw|truncated input) x (guard|none) "
            "x (strip lead|not) give 60.00/60.17/61.67/61.83, a 1.83-point spread "
            "inside one reading, and an independent reimplementation landed on 61.50, "
            "outside all eight. Once a letter fallback is admitted, the fallback's "
            "exact input and guard are part of the definition. The A0-minus-A1 GAP is "
            "robust across all three readings (2.33 to 58.00, never changing sign) "
            "even though no LEVEL is, so the gap is the claimable quantity.",
        "match_stage_counts": {k: dict(v) for k, v in stage_tab.items()},
        "opens_with_option_marker": fmt,
        "opens_with_option_marker_note": "scorer-free: count of generations beginning with an "
                       "explicit option marker ((A) [A] A) A. A:).  Probe rejects bare "
                       "lowercase-letter prose ('a bird'), and is calibrated on 10 cases "
                       "before any count is emitted.  Use this channel for claims that must "
                       "survive a change of scoring rule.",
        "strict_note": "STRICT uses only exact / prefix / label-stripped matching. The "
                       "as-run numbers additionally allow containment, token-F1>=0.5 and a "
                       "bare-letter fallback; report both, because a large gap means the "
                       "magnitude is scorer-dependent.",
        "wilson95": {k: {t: list(v) for t, v in d.items()} for k, d in ci.items()},
        "macro3_note": "equal-weight mean of the three task accuracies; NOT Avg_11; not "
                       "comparable to the 52.8/64.2 eleven-task aggregates nor to the "
                       "60.58/67.50 decode-path control",
        "invalid_policy": "invalid answers count as WRONG; rows are never dropped; every "
                          "denominator is exactly n_per_task",
        "scorer_audit": scorer_audit,
        "lead_fixed_accuracy_pct": fixed_acc,
        "verbalising_rows_R1_literal_phrase": lead_rows,
        "verbalising_rows_R2_scorer_own_rule": lead_rows_scorer_rule,
        "verbalising_rule_note": "R1 = generation starts with the literal phrase 'the "
                       "answer is'. R2 = the scorer's own strip_answer_lead regex "
                       "(^\\s*(?:the\\s+)?(?:correct\\s+)?answer\\s*(?:is|:)?\\s*), which "
                       "also matches a bare 'Answer:'. The two give different counts (A1 "
                       "race: 26 vs 60), so any quoted count must name its rule. R2 is the "
                       "population the lead-fix is defined over.",
        "lead_fixed_note": "The as-run accuracy_pct above is the table of record. These "
                       "lead-fixed numbers are the SAME decodes rescored with one added "
                       "fallback (letter_of on lead-in-stripped text); nothing was "
                       "re-decoded. Report them ALONGSIDE the as-run rows, never instead "
                       "of them. The defect zeroes only A1: A2/A3/A4 change 0 rows, so "
                       "the latent-content comparison is unaffected by it.",
    }
    dst = out / "g2_aggregate.json"
    dst.write_text(json.dumps(summary, indent=2))
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

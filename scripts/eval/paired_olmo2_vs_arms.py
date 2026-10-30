#!/usr/bin/env python3
"""Paired comparison of OLMo-2-0425-1B against our arms and peers, on IDENTICAL rows.

WHY NOT TWO MARGINAL NUMBERS. This repo has already paid for that mistake: a "three wins"
claim assembled from marginal accuracies collapsed once the same rows were paired, and an
arm that looked 7-for-7 dominant turned out to be winning only on rows carrying a scoring
artefact. Two non-overlapping confidence intervals are not a paired test, and two overlapping
ones do not establish equivalence. Everything reported here is paired on row id.

ROW IDENTITY IS VERIFIED, NOT ASSUMED. Equal n is not equal rows -- and this project has a
recorded case of two arms matching to four digits while 22% of their predictions differed.
Rows are joined on `id`, and the join is only accepted when the ground truth stored with that
id agrees between the two files. Ids whose gold disagrees are counted and dropped, and if the
surviving overlap falls below --min-overlap the cell is REFUSED rather than reported.

TEST CHOICE FOLLOWS THE METRIC, WHICH DIFFERS BY CELL. The brief prescribed McNemar
everywhere; that is wrong for the partial-credit cells, and another agent in this project
demonstrated that binarising a continuous cell can flip the answer. So:

  * binary per-row score (RULER niah_single_1 / niah_multikey_1 / qa_1, bAbILong, and the 13
    choice-style capability tasks) -> exact two-sided McNemar, binomial on the discordant
    pairs. Deliberately exact and not the chi-square approximation: discordant counts in
    these cells are frequently under 25, where the approximation is not trustworthy.
  * continuous per-row score (RULER vt = mean over 5 golds, cwe = mean over 10 golds,
    squad token-F1) -> Wilcoxon signed-rank over per-row differences with zero differences
    dropped, plus the paired mean difference. A McNemar on a thresholded version of the same
    cell is printed alongside ONLY as a secondary line, flagged as such, never as the result.

The script verifies empirically that a cell it believes is binary really is binary
(every per-row score in {0,1}) and refuses to run McNemar on it otherwise, so a harness
change that turns a binary cell continuous cannot silently produce the wrong test.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
LONGCTX = REPO / "results/benchmarks/longctx"
SHORTCTX = REPO / "results/benchmarks"

# Short-context tasks whose per-row score is partial credit, so McNemar is the wrong test.
# squad is token-F1; lambada's secondary is EM but its primary here is a 0/1 head match.
SHORTCTX_CONTINUOUS = ("squad",)

# The six tasks whose prompts were FIXED to inline the `choices` field. run_olmo2_baseline.sh
# emits the fixed prompt for these (inline_choices now lives inside build_prompt), so a peer
# must be read from results/benchmarks/fix2choices/baseline-<peer>/ when that exists -- its
# plain baseline-<peer>/ dir holds PRE-FIX prompts, which is a different protocol.
#
# This is not cosmetic routing. The gold key folds in the prompt tail, so a pre-fix peer gave
# "0 rows pair with matching gold" and the cell was REFUSED. That refusal was correct, but it
# was refusing comparisons that ARE available from the fixed dir. Had I keyed on the bare
# answer string instead, those rows would have paired silently and the test would have
# reported a prompt-protocol difference as a capability difference -- on the six tasks where
# the fix is worth 20-40 points to some models.
FIXED_PROMPT_TASKS = ("piqa", "arc_easy", "arc_challenge", "winogrande", "sciq",
                      "commonsenseqa")


def peer_shortctx_path(peer: str, task: str):
    """Locate the peer file for `task` generated under MY prompt protocol, or say why not."""
    fixed = SHORTCTX / "fix2choices" / f"baseline-{peer}" / f"{task}.jsonl"
    base = SHORTCTX / f"baseline-{peer}" / f"{task}.jsonl"
    if task in FIXED_PROMPT_TASKS:
        if fixed.exists():
            return fixed, "fix2choices"
        if base.exists():
            return None, ("peer has only PRE-FIX prompts for this task (no fix2choices dir); "
                          "not protocol-comparable")
        return None, "absent"
    if base.exists():
        return base, "baseline"
    return None, "absent"

# RULER cells whose per-row score is a mean over several golds -> continuous partial credit.
# vt has 5 golds, cwe has 10; run_open_context_bench scores them with string_match_all (mean),
# while qa_1/qa_2 use string_match_part (max over golds) which is binary.
CONTINUOUS_RULER = ("vt", "cwe", "fwe")


def exact_mcnemar(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar: binomial(n=discordant, p=0.5). Returns 1.0 when no discordance."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilcoxon(diffs: list[float]) -> tuple[int, float | None]:
    """Two-sided Wilcoxon signed-rank over non-zero differences.

    Exact enumeration for n<=20 (where the normal approximation is poor and these cells often
    land), normal approximation with tie correction above that. Returns (n_nonzero, p).
    """
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return 0, None
    mags = sorted((abs(d), i) for i, d in enumerate(nz))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and mags[j + 1][0] == mags[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0            # average rank over the tie block (1-based)
        for k in range(i, j + 1):
            ranks[mags[k][1]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nz) if d < 0)
    w = min(w_plus, w_minus)
    if n <= 20:
        # exact: enumerate every sign assignment over the observed ranks
        total = 1 << n
        cnt = 0
        for mask in range(total):
            s = sum(ranks[b] for b in range(n) if mask >> b & 1)
            if s <= w:
                cnt += 1
        return n, min(1.0, 2.0 * cnt / total)
    mean = n * (n + 1) / 4.0
    # tie-corrected variance
    from collections import Counter
    tie_term = sum(t ** 3 - t for t in Counter(ranks).values())
    var = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var <= 0:
        return n, None
    z = (w - mean) / var ** 0.5
    # two-sided normal tail without scipy
    from math import erfc
    return n, min(1.0, erfc(abs(z) / 2 ** 0.5))


def longctx_rows(path: Path, dataset: str) -> dict[int, tuple[str, float, int]]:
    """{id: (gold_key, score, input_tokens)} for one dataset inside a longctx result file."""
    d = json.loads(path.read_text())
    out: dict[int, tuple[str, float, int]] = {}
    for r in d.get("rows", []):
        if r.get("dataset") != dataset:
            continue
        gold = json.dumps(r.get("answers"), sort_keys=True)
        out[r["id"]] = (gold, float(r.get("score") or 0.0), int(r.get("input_tokens") or 0))
    return out


def seen_ok(path: Path, dataset: str) -> tuple[bool, str]:
    """Decide whether a cell is measurable for THIS file, from the per-row `input_tokens`.

    NOT from the file's top-level `audit` block: that block records the *arm / checkpoint*
    (step, writer_type, backbone_path), not per-cell window coverage, so keying into it by
    dataset name silently yields {} for every file and every cell would be waved through. I
    made exactly that mistake and it passed 27 comparisons before I checked.

    The reliable signal is `input_tokens` against the cell's nominal length, which is encoded
    in the dataset name (`4096_vt` -> 4096). A model whose window is below the cell length
    shows up as a *constant* `input_tokens` equal to its own budget on every row -- e.g.
    pythia/phi-2/dclm all sit at exactly 1920 for both the 4096 and the 16384 cells. Such a
    cell measures truncation, not long-context capability, and must be refused.

    KEPT_FRAC is 0.85 rather than 1.0 on purpose: RULER prompts tokenise to slightly different
    lengths per tokenizer, so an in-window model legitimately lands a little under nominal
    (0.87-1.0 across the arms here), while a clamped model lands at 0.47 or 0.12. The gap
    between those two regimes is wide, so the threshold is not delicately placed.
    """
    import re as _re
    d = json.loads(path.read_text())
    toks = [int(r.get("input_tokens") or 0) for r in d.get("rows", [])
            if r.get("dataset") == dataset]
    if not toks:
        return False, "no rows for this dataset"
    m = _re.match(r"^(\d+)[_/]", dataset)
    if not m:
        # RULER names the length first (`4096_vt`); bAbILong names it LAST with a `k` suffix
        # (`qa1_16k`, `qa2_4k`). Only matching the RULER form silently waved through every
        # bAbILong cell -- including 16k cells on a 2048-window model, which is how this
        # bug first hid from me.
        m = _re.search(r"[_/](\d+)k$", dataset, _re.I)
        nominal = int(m.group(1)) * 1024 if m else None
    else:
        nominal = int(m.group(1))
    if nominal is None:
        return True, f"no nominal length in dataset name; max_input_tokens={max(toks)}"
    mx, n = max(toks), len(toks)
    frac = mx / nominal
    # A single constant value across every row is the fingerprint of a hard clamp.
    constant = len(set(toks)) == 1
    if frac >= 0.85:
        return True, f"max_input_tokens={mx}/{nominal} ({frac:.2f}) over {n} rows"
    return False, (f"max_input_tokens={mx}/{nominal} ({frac:.2f}"
                   f"{', constant on every row' if constant else ''}) -- the model never saw "
                   f"the full context, so this measures truncation, not capability")


def compare(name: str, A: dict, B: dict, a_label: str, b_label: str,
            force_continuous: bool, min_overlap: int) -> dict | None:
    ids = sorted(set(A) & set(B))
    gold_clash = [i for i in ids if A[i][0] != B[i][0]]
    usable = [i for i in ids if A[i][0] == B[i][0]]
    rec: dict = {
        "cell": name, "a": a_label, "b": b_label,
        "n_a": len(A), "n_b": len(B), "n_overlap": len(ids),
        "n_gold_clash": len(gold_clash), "n_usable": len(usable),
    }
    if len(usable) < min_overlap:
        rec["verdict"] = (f"REFUSED: only {len(usable)} rows pair with matching gold "
                          f"(need {min_overlap})")
        return rec
    a = [A[i][1] for i in usable]
    b = [B[i][1] for i in usable]
    rec["a_mean"] = sum(a) / len(a) * 100
    rec["b_mean"] = sum(b) / len(b) * 100
    rec["paired_mean_diff"] = rec["a_mean"] - rec["b_mean"]
    # a_mean is the mean ON THE PAIRED SUBSET, which is NOT the headline aggregate whenever the
    # peer evaluated fewer rows (mamba-2.8b has 50 of the 100 bAbILong rows, where OLMo-2 scores
    # 0.220 rather than its headline 0.190). Flag it so the two are never transcribed as if
    # interchangeable.
    if len(usable) < max(len(A), len(B)):
        rec["subset_warning"] = (
            f"means are over the {len(usable)} paired rows only, not all "
            f"{max(len(A), len(B))}; do not quote a_mean as the headline score")
    # Equal aggregates do NOT imply equal runs: OLMo-2 and qwen2.5-1.5b both score 58.0 on
    # bAbILong qa1_4k while disagreeing on 36 rows (18-18) with 0 identical predictions. Record
    # the concordance so a p=1 is never mistaken for "the models behaved the same".
    rec["n_concordant"] = sum(1 for x, y in zip(a, b) if x == y)
    rec["n_discordant_rows"] = len(usable) - rec["n_concordant"]

    is_binary = all(x in (0.0, 1.0) for x in a + b)
    rec["scores_are_binary"] = is_binary
    if force_continuous or not is_binary:
        nz, p = wilcoxon([x - y for x, y in zip(a, b)])
        rec.update(test="wilcoxon_signed_rank", n_nonzero=nz, p=p)
        if not force_continuous:
            rec["note"] = "treated as continuous because per-row scores are not all in {0,1}"
        # secondary only, clearly marked: thresholding changes the question being asked
        ao = sum(1 for x, y in zip(a, b) if x > y)
        bo = sum(1 for x, y in zip(a, b) if y > x)
        rec["secondary_mcnemar_on_thresholded"] = {
            "a_better_rows": ao, "b_better_rows": bo, "p": exact_mcnemar(ao, bo),
            "caveat": "NOT the headline test: thresholding a partial-credit cell asks a "
                      "different question and has flipped an answer in this project",
        }
    else:
        ao = sum(1 for x, y in zip(a, b) if x == 1.0 and y == 0.0)
        bo = sum(1 for x, y in zip(a, b) if x == 0.0 and y == 1.0)
        rec.update(test="exact_mcnemar", a_only=ao, b_only=bo,
                   n_discordant=ao + bo, p=exact_mcnemar(ao, bo))
    return rec


def shortctx_rows(path: Path) -> dict:
    """{id: (gold_key, per_row_score, 0)} for one generate-then-match .jsonl.

    The per-row score is recomputed HERE rather than read from a summary, because a paired
    test needs the per-row 0/1 and the only published artefact is the cell aggregate.
    Recomputing also guarantees both sides of every comparison are graded by the identical
    code path.

    IT MUST MIRROR score_gen_match.score_file EXACTLY, and a self-check enforces that
    (--verify-recompute). My first version silently diverged in two ways, and the check is
    the only reason I know:

      * it applied the bare-option-label rule unconditionally. That rule is `LETTER_FIRST`,
        an opt-in (SCORE_LETTER_FIRST=1) that is OFF by default precisely so previously
        reported numbers reproduce. Forcing it on moved 5 OLMo-2 cells (sciq +9.0, arc_easy
        +5.5) and 20 peer cells, i.e. it would have manufactured a scoring-protocol
        difference and then tested it as if it were a capability difference.
      * it scored lambada with token-F1. score_file special-cases lambada: the gold is the
        final word and must appear in the first 3 tokens of the generation. Using F1 gave
        13.7 where the repo publishes 68.5 -- a 55-point artefact.

    So the rule here is `S.LETTER_FIRST`, read from the module, not a local choice.
    """
    import score_gen_match as S
    task = path.stem
    out: dict = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("id")
            if rid is None:
                continue
            gen = S.truncate(r.get("generate", "") or "")
            gt = str(r.get("ground_truth", ""))
            choices = r.get("choices") or []
            # gold key = question tail + expected answer STRING (not its index): option order
            # could differ between dirs and an index match would then pair two different rows.
            gold_key = json.dumps([str(r.get("prompt", ""))[-160:], gt], sort_keys=True)
            if choices:
                pred = None
                if S.LETTER_FIRST:
                    bare = S.letter_of(gen)
                    if bare is not None and (len(S.strip_label(gen)) == 0
                                             or len(gen.strip()) <= 2) \
                            and S.LETTERS.index(bare) < len(choices):
                        pred = S.LETTERS.index(bare)
                if pred is None:
                    pred = S.match_choice(gen, choices)
                if pred is None:
                    pred = S.match_choice(S.strip_label(gen), choices)
                if pred is None:
                    pred = S.match_choice(S.strip_answer_lead(S.strip_label(gen)), choices)
                if pred is None:
                    lt = S.letter_of(gen)
                    if lt is not None and S.LETTERS.index(lt) < len(choices):
                        pred = S.LETTERS.index(lt)
                gold = None
                ng = S.normalize_text(gt)
                for i, c in enumerate(choices):
                    if S.normalize_text(c) == ng:
                        gold = i
                        break
                if gold is None:
                    gold = S.match_choice(gt, choices)
                score = float(pred is not None and gold is not None and pred == gold)
            elif task.startswith("lambada"):
                # score_file's own rule: gold is the final word, must land in the first 3 tokens
                first = S.normalize_text(gen).split()
                head = " ".join(first[:3])
                ngt = S.normalize_text(gt)
                score = 1.0 if (ngt and ngt in head) else 0.0
            else:
                score = S.f1(gen, gt)
            out[rid] = (gold_key, score, 0)
    return out


def verify_recompute(dirs: list[str]) -> int:
    """Fatal gate: my per-row recompute must reproduce score_file's published aggregate.

    Run before any p-value is printed. If these disagree, the paired test is being run on
    scores the repo never published, and any "significant" result is an artefact of my own
    scorer divergence rather than of the models.
    """
    import score_gen_match as S
    bad = []
    n_checked = 0
    bases = []
    for d in dirs:
        bases.append(SHORTCTX / f"baseline-{d}")
        f2 = SHORTCTX / "fix2choices" / f"baseline-{d}"
        if f2.is_dir():
            bases.append(f2)      # the fixed dirs are read too, so they must be gated too
    for base in bases:
        for jf in sorted(base.glob("*.jsonl")):
            rows = shortctx_rows(jf)
            if not rows:
                continue
            mine = sum(v[1] for v in rows.values()) / len(rows) * 100
            task, pub, _, n = S.score_file(str(jf))
            n_checked += 1
            if abs(mine - pub) > 0.005:
                bad.append((d, task, mine, pub))
    print(f"[verify-recompute] LETTER_FIRST={S.LETTER_FIRST}  files={n_checked}  "
          f"mismatches={len(bad)}")
    for d, t, m, p in bad:
        print(f"  MISMATCH {d}/{t}: recompute={m:.2f} published={p:.2f}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="olmo-2-1b")
    ap.add_argument("--peers", default="pythia-2.8b,dclm-1b,champ-dynamic-30k,global-dit-50k,"
                                       "qwen2.5-1.5b,gla-1.3b,phi-2")
    ap.add_argument("--min-overlap", type=int, default=50)
    ap.add_argument("--json", default=None)
    ap.add_argument("--suites", default="babilong,ruler",
                    help="comma list; 'shortctx' runs the generate-then-match suite")
    ap.add_argument("--short-dirs", default="champ-dynamic-30k,global-dit-50k,133b-chunked,"
                                           "133b-global,qwen25-15b,pythia-28b,dclm-1b,"
                                           "nemotron-4b,phi-2,gla-13b,mamba-2.8b",
                    help="directory suffixes under results/benchmarks/baseline-* for shortctx")
    ap.add_argument("--verify-recompute", action="store_true",
                    help="only run the scorer-agreement gate and exit nonzero on mismatch")
    args = ap.parse_args()

    recs: list[dict] = []
    suites = [s for s in args.suites.split(",") if s]
    sys.path.insert(0, str(REPO / "scripts/eval"))

    if args.verify_recompute:
        return verify_recompute([args.slug] + args.short_dirs.split(","))

    if "shortctx" in suites:
        # the gate is not optional: a divergent scorer invalidates every p-value below
        if verify_recompute([args.slug] + args.short_dirs.split(",")) != 0:
            print("FATAL: per-row recompute disagrees with score_file; refusing to report "
                  "paired tests on scores the repo never published")
            return 2
        mine_dir = SHORTCTX / f"baseline-{args.slug}"
        for jf in sorted(mine_dir.glob("*.jsonl")):
            task = jf.stem
            A = shortctx_rows(jf)
            if len(A) < 200:
                print(f"[skip] shortctx/{task}: only {len(A)} rows, still running")
                continue
            cont = task in SHORTCTX_CONTINUOUS
            for peer in args.short_dirs.split(","):
                pf, why = peer_shortctx_path(peer, task)
                if pf is None:
                    if why != "absent":
                        print(f"  shortctx/{task:18s} vs {peer:18s} REFUSED: {why}")
                        recs.append({"cell": f"shortctx/{task}", "b": peer,
                                     "verdict": f"REFUSED: {why}"})
                    continue
                B = shortctx_rows(pf)
                if not B:
                    continue
                r = compare(f"shortctx/{task}", A, B, args.slug, peer, cont, args.min_overlap)
                if r:
                    r["peer_source"] = str(pf.relative_to(SHORTCTX))
                    recs.append(r)
                    if "verdict" in r:
                        print(f"  {r['cell']:26s} vs {peer:18s} {r['verdict']}")
                    else:
                        star = "*" if (r.get("p") is not None and r["p"] < 0.05) else " "
                        print(f"  {r['cell']:26s} vs {peer:18s} "
                              f"{r['a_mean']:5.1f} vs {r['b_mean']:5.1f}  "
                              f"{r['test']:22s} p={r['p']:.4g}{star} "
                              f"(n={r['n_usable']}, gold_clash={r['n_gold_clash']})")

    for suite in [s for s in suites if s != "shortctx"]:
        mine = LONGCTX / f"{args.slug}__{suite}.json"
        if not mine.exists():
            print(f"[skip] {mine} not written yet")
            continue
        d = json.loads(mine.read_text())
        for dataset in sorted((d.get("aggregates") or {})):
            ok, why = seen_ok(mine, dataset)
            if not ok:
                print(f"[REFUSED] {suite}/{dataset}: {why}")
                recs.append({"cell": f"{suite}/{dataset}", "verdict": f"REFUSED: {why}"})
                continue
            A = longctx_rows(mine, dataset)
            cont = suite == "ruler" and any(dataset.endswith(c) for c in CONTINUOUS_RULER)
            for peer in args.peers.split(","):
                pf = LONGCTX / f"{peer}__{suite}.json"
                if not pf.exists():
                    continue
                pok, pwhy = seen_ok(pf, dataset)
                if not pok:
                    print(f"[skip] {suite}/{dataset} vs {peer}: peer cell refused ({pwhy})")
                    continue
                B = longctx_rows(pf, dataset)
                if not B:
                    continue
                r = compare(f"{suite}/{dataset}", A, B, args.slug, peer, cont,
                            args.min_overlap)
                if r:
                    recs.append(r)
                    if "verdict" in r:
                        print(f"  {r['cell']:28s} vs {peer:18s} {r['verdict']}")
                    else:
                        star = "*" if (r.get("p") is not None and r["p"] < 0.05) else " "
                        print(f"  {r['cell']:28s} vs {peer:18s} "
                              f"{r['a_mean']:5.1f} vs {r['b_mean']:5.1f}  "
                              f"{r['test']:22s} p={r['p']:.4g}{star} "
                              f"(n={r['n_usable']}, gold_clash={r['n_gold_clash']})")
    if args.json:
        Path(args.json).write_text(json.dumps(recs, indent=2) + "\n")
        print(f"[wrote] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

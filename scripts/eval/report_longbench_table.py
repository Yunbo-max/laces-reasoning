#!/usr/bin/env python3
"""Per-model x per-dataset LongBench table, with the truncation accounting next to it.

Prints two blocks:
  1. scores (LongBench convention: 100 x metric), old run vs fixed run
  2. the truncation audit: each model's effective input budget and how much of the prompt it
     actually saw -- so a reader can tell a capability result from an artefact of our cap.
"""
from __future__ import annotations
import json, math, random, statistics
from collections import Counter
from pathlib import Path

ROOT = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
OLD = ROOT / "results/benchmarks/longctx"
NEW = ROOT / "results/benchmarks/longctx_fix"
DATASETS = ["hotpotqa", "2wikimqa", "musique", "narrativeqa", "multifieldqa_en",
            "passage_retrieval_en", "gov_report", "qmsum", "multi_news"]
MODELS = ["qwen2.5-3b", "qwen2.5-1.5b", "llama-3.2-3b", "llama-3.2-3b-bos", "phi-2",
          "gemma-2-2b", "gemma-2-2b-bos",
          "gla-1.3b", "gla-1.3b-loop", "gla-1.3b-bos-loop",
          "rwkv7-world3", "rwkv7-world3-loop", "mamba-2.8b",
          "global-dit-50k", "champ-dynamic-30k"]
OURS = {"global-dit-50k", "champ-dynamic-30k"}
# The one row per model that is actually publishable, and why the others are not. Printed as a
# separate block because three different defects each invalidate a different subset of cells,
# and a reader picking rows off the big table by name would otherwise silently mix
# with-BOS against without-BOS and hf-decode against loop-decode.
PUBLISHABLE = {
    "qwen2.5-3b":        ("qwen2.5-3b",        "no BOS, no fla decode bug: unaffected"),
    "qwen2.5-1.5b":      ("qwen2.5-1.5b",      "no BOS, no fla decode bug: unaffected"),
    "phi-2":             ("phi-2",             "no BOS; window-bound floor, not a measurement"),
    "llama-3.2-3b":      ("llama-3.2-3b-bos",  "needs BOS (id 128000)"),
    "gemma-2-2b":        ("gemma-2-2b-bos",    "needs BOS (id 2); also window-bound floor"),
    "gla-1.3b":          ("gla-1.3b-bos-loop", "needs BOS (id 1) + loop decode"),
    "rwkv7-world3":      ("rwkv7-world3-loop", "hf decode is 2/20 vs loop: hf row invalid"),
    # Not a truncation artefact and not a decode defect (deg%=0 in both arms): mamba scores
    # BETTER on less input. Paired on the same 50 rows, same official 32-token gen cap, budget
    # is the only difference: 8.18 @1024 vs 2.83 @32768, delta +5.35 [+2.53, +8.31], 18/4,
    # sign p=0.0043. So its main row measures a sub-4k degradation, not long-context skill.
    #
    # CONSEQUENCE FOR OUR ARMS, and the reason this must never be scored as a win: on the 171
    # rows mamba did finish (recovered from logs/qz/lbfix-mamba-2.8b-longbench.log; the log's
    # `score=` is per-row, verified against qwen2.5-3b/hotpotqa where log mean == JSON mean ==
    # 32.37), ours looks like a clear winner -- global +2.91 [+1.19,+4.62], champ +2.83
    # [+1.11,+4.60], both CIs excluding 0. Re-paired on the SAME rows at mamba's BEST budget
    # (1024) the sign FLIPS: global -1.45 [-4.70,+1.71], champ -2.14 [-5.13,+0.63], i.e.
    # unresolved and pointing the other way. The apparent win was OUR truncation choice, not
    # mamba's limitation. Never report a mamba comparison drawn at our 32768 cap.
    "mamba-2.8b":        ("mamba-2.8b",        "REAL length collapse (8.2 @1024 vs 2.8 @32768, n=50 paired)"),
    "global-dit-50k":    ("global-dit-50k",    "ours"),
    "champ-dynamic-30k": ("champ-dynamic-30k", "ours"),
}
# Official LongBench dataset2maxlen -- the generation cap, not a bug. A model that never
# emits EOS burns the whole cap and dilutes token-overlap F1 against a 1-3 word gold, so the
# stop rate has to be printed next to the score or the table is not comparable across models.
#
# Which cells never emit EOS is a MEASUREMENT, with no causal story attached. Measure it as
# "rows ending BELOW their own dataset's max_gen", not as a distinct-length count: MAXLEN takes
# only 4 values (32/64/128/512) across the 9 datasets, so a fully pinned cell shows 4 distinct
# lengths, not 9, and a distinct-length count mostly measures how many budgets the suite has.
# (An earlier version of this comment said "450/450 with 9 distinct lengths (= one per dataset)".
# Both halves were wrong: the count is 4, and it is per BUDGET, not per dataset.) Measured on the
# cells this report actually SELECTS (re-measured 2026-09-21 after the EOS fix landed; the earlier
# numbers here were taken from the pre-fix cells and are kept below only as the "before" column):
# qwen2.5-3b 393/450, qwen2.5-1.5b 305/450, llama-3.2-3b-bos 161/450, phi-2 156/450,
# rwkv7-world3 141/450, gemma-2-2b-bos 88/450, global-dit-50k (ours) 79/450,
# champ-dynamic-30k (ours) 32/450, gla-1.3b-bos-loop 0/450.
# The EOS fix is what moved rwkv7 0 -> 141 and global-dit 0 -> 79; those two were the cells whose
# stop id was declared but unresolved. gla remains the ONE genuinely never-stopping cell, and that
# is not the same defect: gla declares eos=2 consistently in all three config files, so its stop
# condition was live and it simply never emits it (0 of its rows contain id 2 AND 0 end below
# budget -- both halves needed, since break-before-append makes the first alone worthless).
# Do NOT attribute this to architecture or to BOS -- the 2x2 refutes both. Architecture: an
# attention model (gemma, no BOS) was never-EOS while a recurrent one (champ) stops on 32 rows.
# BOS: not sufficient (qwen2.5-3b, phi-2 have no BOS and stop) and not necessary
# (gla-1.3b-bos-loop has prepend_bos=True and is 0/450). BOS does shift it on gemma --
# runner-matched inside longctx_fix, 450/450 -> 362/450 and macro 6.21 -> 14.21.
#
# DO NOT infer "the model never emitted id X" from "id X appears in 0 stored rows". greedy_loop
# and _stream both `break` BEFORE `ids.append(token)` (verified at run_open_context_bench.py
# 1188-1190 and 1366-1368, laces_latent_refiner.py 281-283), so an id that FIRED leaves no
# trace. "0 rows contain X" is equally consistent with "never emitted" and "always fired".
# The sound form of the argument uses the below-budget count plus the run-time stop set: the
# published runs passed eos_id=tokenizer.eos_token_id alone = {65530}, and the only early exit
# is `if token in stop: break`, so champ's 32 short rows (30 distinct lengths, 3..458, all
# ending on sentence punctuation, and 0 of its 418 full rows containing 65530) can ONLY be
# champ emitting 65530. 65530 is therefore champ's one WORKING stop token -- dropping it in
# favour of generation_config's 0 would move champ's published score, which is why
# resolve_stop_ids takes the UNION rather than the first source that answers.
MAXLEN = {"hotpotqa": 32, "2wikimqa": 32, "musique": 32, "passage_retrieval_en": 32,
          "narrativeqa": 128, "multifieldqa_en": 64,
          "gov_report": 512, "qmsum": 512, "multi_news": 512}


def load(d: Path, name: str):
    p = d / f"{name}__longbench.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:            # still being written
        return None


# The directories cell selection may read, as an explicit ALLOWLIST rather than a glob.
#
# 2026-09-21: a peer's repo-wide sweep found 4 LIVE pre-fix alias files for these two models, and
# my report had accounted for only 2. The other two -- `longctx_decodefix/rwkv7-world3` (md5
# b3b4d7ce, macro 9.55) and `longctx_globalarm/global-dit-50k` (md5 ff3d353c, 9.55) -- are outside
# this allowlist, so they cannot reach the table. That is LUCK, not design: I chose these three
# directories before knowing the other two existed. Keeping the allowlist explicit is what
# converts the luck into a property, and `audit_unread_cells` below reports what it excludes so
# the next such file is visible instead of merely harmless.
#
# Do not replace this with a glob over `longctx*`. `longctx_fix` and `longctx_decodefix` both
# read like "the fixed directory" and both contain pre-fix files; `longctx_decodefix` is the
# DEFAULT --stage of scripts/eval/publish_gated_loop_cell.py, so it fills up with whatever was
# most recently staged. Directory names are not provenance.
CANDIDATE_DIRS = (OLD, NEW, ROOT / "results/benchmarks/longctx_eosfix")


def stop_fires(doc) -> bool:
    """Does the declared stop set actually END any row? A BEHAVIOURAL check, not a field read.

    Peer's publishing gate (2026-09-21) passed a pre-fix file with GATE PASSED on every other
    check -- rows 450/450, 9 cells, n=50, cap 32768, scorer control 450/450, "no real text after a
    stop id" (vacuously true, there are no stop ids). All of those are DOWNSTREAM of the defect:
    they pass on a broken file precisely because the stop condition was the thing that was broken.

    The same reasoning attacks `cell_tier`. `stop_ids` is a provenance FIELD, and a provenance
    field can be present while the behaviour it implies is absent -- a file could be re-tagged, or
    written by a run whose resolver produced an inert set. So tier 2 requires the set to have had
    an observable effect: at least one row ending below its own dataset's `max_gen`.

    This is the sound form of the argument, and it is the only form available, because
    break-before-append means a stop id that FIRED leaves no trace in the stored ids -- you cannot
    verify firing by searching for the id. Counting short rows is the observable consequence.
    """
    if not doc.get("stop_ids"):
        return False
    for r in doc.get("rows", []):
        cap = MAXLEN.get(r.get("dataset"))
        if cap is not None and len(r.get("prediction_token_ids") or []) < cap:
            return True
    return False


def cell_tier(doc) -> int:
    """How corrected a cell is, read off the FILE rather than its path.

    2 = declares the resolved stop set AND that set demonstrably fires (see `stop_fires`)
    1 = written after the budget fix (`effective_input_budget` exists)
    0 = neither, i.e. a genuine pre-fix run

    All of these are properties of the file, so they survive another agent rewriting the
    directory, which mtime and directory name do not.

    Tier 2 is deliberately NOT just "`stop_ids` is present". A declared-but-inert stop set is
    exactly the original defect wearing the fix's provenance fields, and it would outrank a
    genuinely corrected cell. Measured on the three cells this report selects: rwkv7 141/450,
    global-dit 79/450, champ 32/450 rows end below budget, so no current cell is demoted by this.
    """
    cfg = doc.get("config", {}) or {}
    if stop_fires(doc):
        return 2
    if "effective_input_budget" in cfg:
        return 1
    return 0


def load_best_cell(name: str, aliases: tuple = ()):
    """The most-corrected cell for a model, chosen by PROVENANCE rather than by path.

    2026-09-21: the publishable rwkv7 row was pinned to the literal cell name
    `rwkv7-world3-loop` in `longctx_fix`. When the EOS fix was published it went to the CANONICAL
    path (`longctx/rwkv7-world3`, md5 d5c909d0, macro 0.1197, 141/450 rows stopping,
    `stop_ids=[0,2,65530]`) and the alias file was left untouched (md5 fcbb4e98, macro 0.0955,
    0/450 stopping, `stop_ids=None`). So the report went on printing 9.5 for a cell that had been
    superseded by 11.97, and re-running it produced a byte-identical table -- the staleness was
    invisible precisely because nothing errored. Same failure mode `load_prefix` was written for,
    one directory over.

    Selection is by TIER, taken over every (directory, candidate) pair. My first version of this
    function returned the first directory that answered, which is wrong in both directions and I
    caught it only by testing it: `OLD` comes first, so phi-2 would have REGRESSED from the fixed
    16.01 back to the broken-budget 2.80 -- reintroducing, in the fix for one stale-file bug, the
    exact bug the whole re-run existed to remove.

    An alias carries a REQUIRED CONFIG, not just a name, because the same cell name means
    different measurements in different directories: `longctx_fix/rwkv7-world3` is the hf-decode
    row (macro 2.91) that the publishable list exists to exclude, while `longctx/rwkv7-world3` is
    loop decode (11.97). Matching on the name alone would have silently swapped in the invalid
    one. A candidate is usable only if every key in its require-dict matches its own `config`.
    """
    best, best_tier = (None, None), -1
    for directory in CANDIDATE_DIRS:
        for candidate, require in ((name, {}),) + tuple(aliases):
            doc = load(directory, candidate)
            if doc is None:
                continue
            cfg = doc.get("config", {}) or {}
            if any(cfg.get(k) != v for k, v in require.items()):
                continue
            tier = cell_tier(doc)
            if tier > best_tier:
                best, best_tier = (doc, f"{directory.name}/{candidate}"), tier
    return best


# Aliases are gated on config, never on name alone -- see load_best_cell.
PUBLISHABLE_ALIASES = {
    # `longctx/rwkv7-world3` and `longctx_eosfix/rwkv7-world3` are byte-identical (md5 d5c909d0)
    # and are the SAME configuration as the pinned `longctx_fix/rwkv7-world3-loop`: decode='loop',
    # and no BOS (`prepend_bos=False` there vs the key being absent in the pinned file, which is
    # the same tokenization for rwkv7 -- its tokenizer prepends nothing).
    "rwkv7-world3": (("rwkv7-world3", {"decode": "loop"}),),
}


def audit_unread_cells() -> list:
    """Every LongBench cell for a publishable model that selection could NOT see, repo-wide.

    The allowlist protects the table, but silently: a stray higher-tier cell in a directory I did
    not list would simply never be read, which is the same invisibility that caused the original
    trap (the report printed 9.5 and nothing errored). So the excluded set is reported.

    Returns (path, tier, macro, relation) for anything outranking or tying what was selected.
    """
    allowed = {d.resolve() for d in CANDIDATE_DIRS}
    cells = {c for c, _w in PUBLISHABLE.values()}
    for al in PUBLISHABLE_ALIASES.values():
        cells.update(c for c, _r in al)
    out = []
    for p in sorted((ROOT / "results").rglob("*__longbench.json")):
        if p.parent.resolve() in allowed:
            continue
        name = p.stem.replace("__longbench", "")
        if name not in cells:
            continue
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        model = next((m for m, (c, _w) in PUBLISHABLE.items() if c == name), name)
        sel, sel_src = load_best_cell(PUBLISHABLE.get(model, (name, ""))[0],
                                      aliases=PUBLISHABLE_ALIASES.get(model, ()))
        t, st = cell_tier(doc), (cell_tier(sel) if sel else -1)
        rel = "OUTRANKS SELECTED" if t > st else ("same tier" if t == st else "lower tier")
        out.append((str(p.relative_to(ROOT)), t, macro_from_rows(doc), rel, sel_src))
    return out


def load_prefix(name: str):
    """The genuine PRE-fix run, identified by provenance rather than by directory.

    `OLD` is a live directory that other agents publish into: `longctx/rwkv7-world3` was
    overwritten at 10:03 with the post-fix loop re-run (budget 32768, `decode_path='loop'`),
    so reading `OLD` by name silently printed a FIXED number in the "OLD RUN (broken budget)"
    column and would have fed a post-fix prompt length into the truncation audit's fallback.

    A genuine pre-fix file has no budget bookkeeping at all -- `effective_input_budget` did not
    exist when it was written. That is a property of the file, so it survives someone else
    rewriting the directory; the directory name and the mtime do not.

    KNOWN LIMIT, and it is the reason this returns None rather than guessing: this test separates
    pre-fix from post-fix, but it CANNOT separate unpatched-hf from patched-hf. Both predate the
    budget key, so both look "pre-fix" here. For the fla models (gla, rwkv7) the fla
    recurrent-generate patch changes the sampled token stream while changing zero prompt tokens,
    and its state is recorded ONLY in the job log's `[load]` banner -- not in `config`, not
    anywhere in the JSON. So a residual difference between two cells that both pass this test is
    not yet attributable to noise; confirm the patch line in both logs first
    (`grep -c 'recurrent-generate patch applied' logs/qz/<job>.log`).
    """
    doc = load(OLD, name)
    if doc is None:
        return None
    return None if "effective_input_budget" in doc.get("config", {}) else doc


def degenerate_rate(doc) -> float:
    rows = doc.get("rows", [])
    if not rows:
        return float("nan")
    bad = sum(1 for r in rows if not (r.get("prediction") or "").strip())
    return 100.0 * bad / len(rows)


def degenerate_union_rate(doc) -> float:
    """Fraction of rows that are empty OR collapsed -- the UNION, not max() of the two rates.

    2026-09-21: this column was `max(empty%, collapse%)`, which silently understates whenever the
    two defects hit DIFFERENT rows. Caught on the landed EOS-fixed rwkv7 cell: 23 empty rows and
    46 collapsed rows with ZERO overlap, so max() reports 10.2% while the true unusable fraction
    is 69/450 = 15.3%. max() is only correct if one set contains the other, which nothing
    enforces. (I had predicted "degeneracy rises 46 -> 69" from the rescore preview and was
    right about the count but wrong about the mechanism -- I expected the newly exposed rows to
    be collapsed; they are empty, which is why the max() column did not move at all.)

    THIRD INSTANCE of the same divergence bug, found 2026-09-21 by checking whether this column
    moved when the predicate was widened -- it did not, because this function RE-IMPLEMENTED the
    predicate inline instead of calling `is_degenerate_row`. So the deg% column printed champ 8.0%
    while the exclusion tests twelve lines below used 24.2%. The first instance was a hand-written
    `<=2 distinct ids` exclusion test, the second was the VALIDITY AUDIT reading a different cell.
    All three had the same shape (a copy of the logic drifting from the original) and all three
    flattered whoever benefited. It now delegates, so there is exactly one definition.
    """
    rows = doc.get("rows", [])
    if not rows:
        return float("nan")
    return 100.0 * sum(1 for r in rows if is_degenerate_row(r)) / len(rows)


def local_repeat(seq, nmin: int = 4, nmax: int = 12, rep: int = 3) -> bool:
    """True if any n-gram repeats `rep` times CONSECUTIVELY anywhere in the sequence.

    ADDED 2026-09-21 because `phrase_loop` below, which gates on a repeat covering >=80% of the
    whole generation, was UNDER-REPORTING OUR OWN ARM BY 3x. It only sees whole-output loops.
    champ-dynamic's defect shape is different: a localized 3x repeat inside otherwise varied
    text, total coverage 0.23-0.76, which sails under an 80% threshold. Measured cost on the
    512-cap slice: my detector said champ 28/150, this one says 68/150, and my 28 is a strict
    SUBSET of the 68 (0 rows in mine and not in this). All-450 deg% goes 8.0% -> 24.2%.

    The gap is NOT symmetric between arms, which is why the omission mattered: gla 59 -> 60
    (+1) and rwkv7 19 -> 17 while champ 28 -> 68 (+40). A threshold tuned on gla's failure mode
    was nearly exact for gla and blind for us, so the detector I chose flattered our own arm
    without my noticing. Credit: peer agent af5680b, who showed no union CONTAINING a
    consecutive-repeat test could yield my champ=28 -- an impossibility argument about my
    number, which is what made it checkable rather than a competing measurement.

    TRUE POSITIVES VERIFIED, not assumed: the flagged high-scoring champ rows carry real 3x
    consecutive repeats -- gov_report score 0.365 repeats 'employees who are injured in the
    course of their employment.<?>Federal' 3x at word 77; multi_news 0.290 repeats 'using a new
    technique to remove the fat. The gang was' 3x. A degenerate loop can still score, so
    "it scored well" does not make it a false positive.

    FP-TESTED ON HUMAN TEXT, independently rerun: 0/145 on the gold `answers` of >=36 words,
    deduped, across all four arms. The denominator is asserted non-zero -- the peer's first
    attempt at this control tested 0/0 items because it guessed the wrong key, and a 0/0 control
    prints a clean 0.0% while proving nothing. Same family as "a control must bite first".

    Applied to token ids AND to whitespace-split text, since the two catch different rows.
    """
    if len(seq) < nmin * rep:
        return False
    for n in range(nmin, nmax + 1):
        for i in range(len(seq) - n * rep + 1):
            ph = seq[i:i + n]
            if all(seq[i + k * n:i + (k + 1) * n] == ph for k in range(1, rep)):
                return True
    return False


def phrase_loop(ids, nmin: int = 3, nmax: int = 12, cov: float = 0.8) -> bool:
    """True if one repeated n-gram covers >= `cov` of the generation (non-overlapping count).

    This exists because the token-level test below is the WRONG SHAPE for the defect that
    actually occurs here, and reporting only that test understated degeneracy on five cells.
    rwkv7's broken rows are not "The The The" -- they are

        'The answer is:\\nA:\\nThe answer is:\\nA:\\nThe answer is:\\nA:'

    which loops at the PHRASE level while the tokens inside the phrase stay diverse (uniq-frac
    median 0.094, yet `>=8 consecutive identical ids` fires on 0/450 rows). `collapse_rate`
    scored that file 0.0% when 76.9% of its rows are unusable.

    Counted non-overlapping (k*n / len) rather than by raw n-gram frequency: overlapping counts
    roughly double the apparent coverage on periodic text and inflate the rate.

    FALSE-POSITIVE TESTED BEFORE USE, which is the step that the four detectors in MEMORY.md all
    skipped: on rows known to be correct (score >= 0.9) across all 10 cells this fires on 1/93 =
    1.08%. A detector that has not been FP-tested on known-good rows must not have numbers
    reported from it -- it will silently condemn the cells most worth keeping.
    """
    if len(ids) < 2 * nmin:
        return False
    for n in range(nmin, min(nmax, len(ids) // 2) + 1):
        counts = Counter(tuple(ids[i:i + n]) for i in range(0, len(ids) - n + 1))
        _gram, k = counts.most_common(1)[0]
        if k >= 2 and (k * n) / len(ids) >= cov:
            return True
    return False


def collapse_rate(doc) -> float:
    """% of rows that are NON-EMPTY but degenerate. Delegates to `is_degenerate_row`.

    Two shapes, because a model can emit nothing usable in either. The token-level arm catches
    `<=2 distinct ids`; the phrase-level arm catches periodic text (see `phrase_loop`). Reporting
    only the first is what made me publish deg% = 0.2 for a cell that is 12% degenerate.

    DELEGATED 2026-09-21 -- this was the FOURTH instance of the divergence bug in this file, and
    the worst-placed one. It re-implemented the predicate as `(<=2 distinct ids) or phrase_loop`,
    so when `is_degenerate_row` gained `local_repeat` this column did not move: the VALIDITY
    AUDIT went on printing champ 8.0%, the number I had already RETRACTED, in the table whose own
    header says "read this before quoting any score", twenty lines from a deg% column reporting
    24.2% for the same cell. Same direction as every other instance: it understated OUR arm.

    Empty rows are excluded because `degenerate_rate` (empty%) reports them in the same table, so
    empty% + collapse% = deg%. That was already true by accident (an empty prediction has no ids,
    so neither old arm could fire on it); it is now true by construction.

    ⭐ The check that catches this family is not reading the code, it is asking after every change
    to a shared predicate: did EVERY column that claims to use it actually move? Three of the four
    instances here were found that way and this one was found by noticing a stale VALUE, not stale
    code -- which only worked because I had a retracted number memorised to collide with.
    """
    rows = doc.get("rows", [])
    if not rows:
        return float("nan")
    bad = sum(1 for r in rows
              if (r.get("prediction") or "").strip() and is_degenerate_row(r))
    return 100.0 * bad / len(rows)


def stop_rate(doc) -> float:
    """% of rows that stopped before the official per-dataset cap (i.e. actually emitted EOS)."""
    rows = doc.get("rows", [])
    scored = [r for r in rows if r.get("dataset") in MAXLEN]
    if not scored:
        return float("nan")
    stopped = sum(1 for r in scored
                  if len(r.get("prediction_token_ids") or []) < MAXLEN[r["dataset"]])
    return 100.0 * stopped / len(scored)


def boot_ci(values: list[float], reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for a mean, in LongBench's x100 units.

    Every cell here is 50 rows, so a two-significant-figure score implies far more precision
    than the data carries: a difference of a couple of points between two models can easily be
    the same 50 documents landing differently. Bootstrap rather than a normal SE because these
    per-row F1/ROUGE values are bounded in [0, 1] and heavily zero-inflated, so the sampling
    distribution of the mean is skewed and a symmetric interval would misstate it.
    """
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(reps))
    return (100 * means[int(0.025 * reps)], 100 * means[int(0.975 * reps) - 1])


# Two-sided 95% t critical values, keyed by degrees of freedom (k-1 for k clusters). Needed
# because the independence unit for every paired comparison here is the DATASET (9 of them),
# not the row (450), so the intervals are cluster-t and not bootstrap. scipy is not importable
# on this node (shared_pylibs only), hence the literal table.
T_CRIT = {1: 12.706, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469,
          7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281}


def boot_ci_paired(a: dict, b: dict, reps: int = 10000, seed: int = 0):
    """CI for the mean PER-ROW DIFFERENCE a-b, over shared row ids, CLUSTERED BY DATASET.

    Comparing two marginal CIs is the wrong test when both models scored the same rows, and it
    is needlessly conservative: between-row difficulty (some documents are hard for everything)
    is the dominant variance component in a 9-dataset pool, and pairing removes it. It also
    changes conclusions here -- qwen2.5-3b vs -1.5b have marginal CIs that abut at exactly 23.7,
    which licenses nothing, while the paired interval on the difference is [2.48, 7.96] and
    excludes zero.

    2026-09-21: this returned a percentile bootstrap over the 450 ROWS. That was wrong, and it
    was wrong in OUR FAVOUR. Rows inside one LongBench dataset share a document pool, a prompt
    template and a max_gen budget, so they are not independent draws; resampling them prices
    only within-dataset noise and ignores the much larger between-dataset spread. The unit of
    independence is the (dataset) cluster -- 9 of them -- not the row. Measured cost of the
    error: at the time, the only two "ours better" verdicts this report had ever produced were
    manufactured by it. global-dit vs gla-1.3b-bos-loop +2.06 [+0.90,+3.27] "BETTER" becomes
    [-1.75,+5.87] UNRESOLVED, and champ vs gla +1.52 [+0.45,+2.64] becomes [-2.14,+5.18]
    UNRESOLVED; both are positive on only 6 of 9 datasets, with per-dataset deltas spanning
    -3.77 to +10.20.
    Every "ours worse" verdict survives the correction unchanged. So the bug inflated our arms
    and nothing else -- the asymmetry is the reason this is a correction and not a refinement.

    NOT STILL TRUE, and the update is recorded rather than the sentence rewritten: honouring the
    RWKV-7 tokenizer's stop ids later moved global-dit vs gla to +4.49 [+0.78,+8.21], which
    survives clustering. So a clustered "ours better" verdict DOES now exist and the claim
    "every positive verdict was a unit artefact" would be false if re-asserted. What retires it
    is a different control, not the unit: the official RWKV-7 2.9B baseline earns the same
    margin from the same fix (see `control_vs_gla` and `gla_warning`).

    Now: cluster-t over dataset means (t_{k-1}, k = number of shared datasets). The row-level
    interval is still returned as ``row_lo/row_hi`` so the two can be printed side by side;
    the VERDICT must come from the clustered one. Both are reported in x100 units.

    Returns (n_shared, mean_diff, lo, hi, n_up, n_down, row_lo, row_hi, k, per_dataset).
    """
    keys = sorted(set(a) & set(b))
    if not keys:
        return (0, float("nan"), float("nan"), float("nan"), 0, 0,
                float("nan"), float("nan"), 0, {})
    diffs = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(reps))
    row_lo = 100 * means[int(0.025 * reps)]
    row_hi = 100 * means[int(0.975 * reps) - 1]

    by_ds: dict[str, list[float]] = {}
    for (dataset, _row_id), diff in zip(keys, diffs):
        by_ds.setdefault(dataset, []).append(diff)
    cluster_means = [statistics.mean(v) for v in by_ds.values()]
    k = len(cluster_means)
    point = 100 * statistics.mean(diffs)
    if k < 2:
        # One cluster carries no between-cluster information; refuse to invent an interval.
        lo = hi = float("nan")
    else:
        centre = statistics.mean(cluster_means)
        stderr = statistics.stdev(cluster_means) / math.sqrt(k)
        half = T_CRIT.get(k - 1, 2.306) * stderr
        lo, hi = 100 * (centre - half), 100 * (centre + half)
    per_ds = {d: 100 * statistics.mean(v) for d, v in sorted(by_ds.items())}
    return (n, point, lo, hi,
            sum(1 for d in diffs if d > 0), sum(1 for d in diffs if d < 0),
            row_lo, row_hi, k, per_ds)


def row_scores(doc) -> dict:
    """Per-row scores keyed by (dataset, id) so two models can be compared pairwise."""
    return {(r["dataset"], r["id"]): r["score"]
            for r in (doc or {}).get("rows", []) if "score" in r}


def is_degenerate_row(r) -> bool:
    """The row-level predicate behind `degenerate_union_rate`, exposed for exclusion tests.

    Factored out because I ran the gla exclusion with a hand-written `<=2 distinct ids` test and
    got 26 rows where this detector finds 110 -- the difference is `phrase_loop`, i.e. exactly the
    defect shape gla actually has. The weaker test made the margin look more robust to exclusion
    than it is. An exclusion test must reuse the SAME predicate the deg% column reports, or the
    two numbers in the report disagree and the looser one flatters whoever benefits.

    WIDENED 2026-09-21 with `local_repeat` on both ids and text. That addition raised champ's
    deg% from 8.0% to 24.2% and gla's from 24.4% to 30.2%, i.e. it cost OUR arm three times more
    than the baseline it is quoted against, which is the direction that says the narrower version
    was the biased one. Every number the report prints from this predicate moved with it; none of
    the SCORES did.
    """
    ids = r.get("prediction_token_ids") or []
    text = r.get("prediction") or ""
    return (not text.strip()
            or (len(ids) >= 4 and len(set(ids)) <= 2)
            or phrase_loop(ids)
            or local_repeat(ids)
            or local_repeat(text.split()))


def cluster_t(pairs):
    """(mean, lo, hi, n_pos, k) for [(dataset, diff)] -- cluster-t over datasets, x100 units."""
    by_ds: dict = {}
    for dataset, diff in pairs:
        by_ds.setdefault(dataset, []).append(diff)
    means = [statistics.mean(v) for _d, v in sorted(by_ds.items())]
    k = len(means)
    if k < 2:
        return (100 * statistics.mean(means) if means else float("nan"),
                float("nan"), float("nan"), sum(1 for x in means if x > 0), k)
    centre = statistics.mean(means)
    half = T_CRIT.get(k - 1, 2.306) * statistics.stdev(means) / math.sqrt(k)
    return (100 * centre, 100 * (centre - half), 100 * (centre + half),
            sum(1 for x in means if x > 0), k)


def control_vs_gla() -> float:
    """`rwkv7-world3 - gla` cluster-t mean: the attribution control, computed not quoted.

    Exists so the verdict label in the comparison table cannot drift out of step with the data,
    which is exactly the failure this report already had once (gla_warning printed hardcoded
    pre-EOS-fix numbers under a table showing post-fix ones, and contradicted itself for a day).
    A number that justifies a verdict has to be recomputed wherever the verdict is printed.
    """
    g, _gs = load_best_cell("gla-1.3b-bos-loop")
    r, _rs = load_best_cell("rwkv7-world3", aliases=PUBLISHABLE_ALIASES.get("rwkv7-world3", ()))
    if g is None or r is None:
        return float("nan")
    gsc, rsc = row_scores(g), row_scores(r)
    shared = sorted(set(gsc) & set(rsc))
    return cluster_t([(k[0], rsc[k] - gsc[k]) for k in shared])[0]


def spec_curve() -> None:
    """Specification curve over the DETECTOR's free parameters. Prints the band, not a point.

    ADDED 2026-09-21 because I published `champ - rwkv7 = -3.69 [-6.66,-0.73], excludes 0` as
    "significantly worse than the official baseline", and that CI-exclusion is an artefact of two
    free choices nobody voted on:

      (a) WHICH ARM'S ROWS GET DROPPED. -3.69 is the excl-rwkv7 column -- it removes only the
          BASELINE's degenerate rows. The symmetric column (drop rows degenerate in either arm)
          is -3.20 [-6.55,+0.15] and does NOT exclude 0. I already knew asymmetric exclusion
          favours whoever is excluded against; I wrote that rule for gla and then quoted the
          asymmetric number for rwkv7. The rule has to be applied to the comparison that hurts
          us as well as the one that helps us.
      (b) THE DETECTOR'S PARAMETERS. Sweeping nmin/nmax/rep and phrase_loop's coverage, the SIGN
          is invariant everywhere, but CI-exclusion is not: it holds in only part of the space.
          Peer af5680b swept a disjoint 24-config grid (no coverage threshold, nmax 9/13/17) and
          found exclusion in 8/24. Two independent implementations, same conclusion.
          ⚠️ BUT NEITHER THAT FRACTION NOR A CROSSOVER IS REPORTABLE -- see CORRECTION 9 in the
          body. k/N spans 0%..100% on identical data as the grid's nmax weighting changes, and
          exclusion is NOT monotone in nmax, so there is no single crossover to quote instead.

    ⭐ A CI-EXCLUSION THAT DEPENDS ON A FREE PARAMETER OF A PREPROCESSING DETECTOR IS NOT A
    FINDING ABOUT THE MODELS. What survives here is the SIGN (invariant in every config both of
    us tried) and the two NULLS. Those are the publishable objects.

    Note the direction: this correction makes our own arm look LESS bad (champ - rwkv7 stops being
    "significantly worse"). That is the uncomfortable case -- every previous correction in this
    file went against us, so this one gets the full grid printed rather than an assertion, and the
    conclusion it must NOT be allowed to support is "so champ is fine": the sign is negative in
    every specification, and the comparison that decides publishability (global-dit - rwkv7) is a
    tie in every specification too.

    ⚠️ AND THE FIRST VERSION OF THIS VERY FUNCTION GOT THAT WRONG IN OUR FAVOUR. I keyed the
    verdict label on CI-exclusion alone, so `nexcl == 0` printed "NULL / tie in every spec" -- and
    that label landed on `Chunked - CONTROL`, whose point estimates are -3.76..-2.81 with the sign
    negative in 36/36. A consistent deficit that fails a significance threshold is NOT a tie, and
    calling it one erased our arm's deficit on the very block I wrote to stop myself overclaiming.
    Sign-stability and CI-exclusion are ORTHOGONAL axes; a verdict label must name both. Guarding
    against a bias in the numbers does not make the guard itself unbiased.
    """
    arms = {}
    for nm in ("champ-dynamic-30k", "global-dit-50k", "rwkv7-world3", "gla-1.3b-bos-loop"):
        d, _s = load_best_cell(nm, aliases=PUBLISHABLE_ALIASES.get(nm, ()))
        if d is None:
            return
        arms[nm] = (row_scores(d), {(r["dataset"], r["id"]): r for r in d.get("rows", [])})

    def pred(nmin, nmax, rep, cov):
        def f(r):
            ids = r.get("prediction_token_ids") or []
            text = r.get("prediction") or ""
            return (not text.strip()
                    or (len(ids) >= 4 and len(set(ids)) <= 2)
                    or phrase_loop(ids, cov=cov)
                    or local_repeat(ids, nmin=nmin, nmax=nmax, rep=rep)
                    or local_repeat(text.split(), nmin=nmin, nmax=nmax, rep=rep))
        return f

    # nmax MUST span the transition. My first grid used nmax in {12, 20} ONLY, which sits
    # entirely ABOVE the point where champ - rwkv7 stops excluding 0 (between nmax=10 and 11),
    # so it reported "significant in 0/36 specs" while never sampling the region where that
    # comparison IS significant. That omission flattered OUR arm: it made our chunked arm's
    # deficit against the official baseline look unmeasurable everywhere. Caught by peer
    # af5680b, whose grid used nmax {9,13,17} and straddled the transition; I reproduced their
    # cross-tab at nmin=4 rep=3 cov=0.8 -- symmetric: EXCL at nmax 9,10 then wide from 11 up.
    # ⭐ A SPECIFICATION CURVE THAT DOES NOT STRADDLE THE TRANSITION IS NOT A ROBUSTNESS CHECK,
    # it is one regime reported as if it were the whole space. nmax now spans 8..20.
    # ⚠️ CORRECTION 9 (2026-09-21). BOTH of us then proposed replacing the fraction k/N with "the
    # THRESHOLD", on the stated ground that exclusion is MONOTONE in nmax so each pair has one
    # crossover. I MEASURED that premise (dense sweep, nmax 6..26, all 9 (rep,cov) at nmin=4,
    # symmetric) instead of assuming it. Two findings, and the SECOND is the one that matters:
    #
    #   (i) Monotonicity FAILS in a large minority, measured two ways. Over all 27 (nmin,rep,cov)
    #       settings, `Chunked - CONTROL` is non-monotone in 12/27 and `Chunked - Global` in 9/27.
    #       At nmin=4 specifically it is 4/9 (rep=3 cov=.80/.95, rep=4 cov=.80/.95): exclusion
    #       holds to nmax=10, is wide at 11-13, and RETURNS at 14 (and 15). `Chunked - Global`
    #       re-enters at 22-23. So "the crossover" is often not even well defined.
    #       (I first wrote "NOT monotone" flatly, then over-corrected to "a 1-in-9 exception" after
    #       reading only the nmin=4 rows; the measured figure is 12/27. The peer's "monotone in
    #       9/9" came from stopping at nmax=20. Both of us generalised from a partial sweep, twice.)
    #  (ii) ⭐ THE CROSSOVER IS A KNIFE EDGE, WHICH IS THE REAL REASON NEITHER FORM IS REPORTABLE.
    #       Printing the interval to 6dp, the exclusions next to every crossover are decided in the
    #       second or third decimal of a 9-cluster t-interval: rep=3 cov=0.60 excludes at nmax=10
    #       with hi=-0.0013, rep=2 cov=0.95 at nmax=8 with hi=-0.0616, rep=4 cov=0.80 re-enters at
    #       nmax=15 with hi=-0.0484. A significant/not-significant flip driven by 0.001 of ROUGE is
    #       not a fact about the models; it is the threshold's own discontinuity. The crossover's
    #       LOCATION is therefore set by whichever config lands on the near side of a hairline,
    #       exactly as k/N is set by how many configs the grid puts below it.
    #
    # Neither is clusters dropping out of the cluster-t: k stays 9 at every nmax in both rows. The
    # mechanism is the per-dataset cells collapsing -- min rows/dataset falls 19 -> 13 -> 7 (at the
    # nmax=14 re-entry) -> 5 -> 2 -> 1, with the half-width growing 3.25 -> 4.08 -> 6.12.
    # ⭐ AND THAT CONTAMINATES THE "ROBUST" BAND ITSELF, not just the significance count. Gating to
    # configs where every dataset keeps >=10 of its 50 rows: `Global - gla` +1.42..+4.36 becomes
    # +2.95..+3.83, `rwkv7 - gla` +1.95..+5.62 becomes +3.31..+3.68, and `Chunked - gla` (which
    # "FLIPS" over all 63) becomes +0.03..+0.08 -- i.e. flat zero, which independently matches the
    # -0.05 measured on gla-clean rows. The wide bands I was about to publish as "the robust
    # object" had their extremes set by configs where a dataset was down to ONE row.
    # ⛔ CORRECTION 9f -- I wrote here "the decisive null is UNAFFECTED: `Global - CONTROL` keeps
    # >=11 rows in every dataset at all 63 settings, so its 0/63 and its -0.53..-0.13 band are
    # estimable throughout". That is a PER-PAIR property and it does not transfer. Per-pair the
    # min-rows/dataset really is 11..25 and 63/63 configs clear a min>=10 gate; on the COMMON kept
    # set of 9c the same grid has min-rows max 8, 0/63 clear min>=10, and 4/63 configs LOSE A WHOLE
    # DATASET (k=8). So an immunity claim inherits the keep convention of whatever column it was
    # measured on -- the n_keep gate and the common-set fix have to be applied TOGETHER or each one
    # launders the other's defect. What survives both: 0/63 exclude 0, in every convention.
    # ⭐ SECOND time a monotonicity assumption about an exclusion sweep produced a tidier story than
    # the data supports (the first was the n=450/300/230 "middle band"). An exclusion sweep is not
    # a trend; it has to be printed, with n_keep.
    grid = [(a, b, c, d) for a, b in ((3, 8), (4, 8), (4, 10), (3, 12), (4, 12), (4, 20), (5, 12))
            for c in (2, 3, 4) for d in (0.6, 0.8, 0.95)]
    claims = [("global-dit-50k", "gla-1.3b-bos-loop", "OUR Global - gla"),
              ("rwkv7-world3", "gla-1.3b-bos-loop", "CONTROL rwkv7 - gla"),
              ("global-dit-50k", "rwkv7-world3", "OUR Global - CONTROL"),
              ("champ-dynamic-30k", "rwkv7-world3", "OUR Chunked - CONTROL"),
              ("champ-dynamic-30k", "gla-1.3b-bos-loop", "OUR Chunked - gla")]
    # ⭐ CORRECTION 9c. Each row below keeps rows PER PAIR (`shared` is that pair's two arms),
    # so the five rows are computed on FIVE DIFFERENT kept sets: |K| 165..366 depending on the
    # row, and the two rows the prose compares (Global-gla vs CONTROL-gla) differ by 37..69 rows
    # at EVERY config. That is the right unit for each row taken ALONE but makes the rows
    # non-comparable, and my prose compared them. The test that catches it is the algebraic
    # identity (Global-gla) - (CONTROL-gla) - (Global-CONTROL), which must be 0: under per-pair
    # keeps its residual reaches 1.0698, FOUR TIMES the +0.29..+0.38 gap that CORRECTION 9b was
    # about. So a COMMON-keep-set column is printed alongside, and the residual is printed.
    # ⭐⭐ It is not cosmetic: on a common kept set `Global - CONTROL` STOPS being sign-stable
    # (-0.53..-0.13 becomes -2.00..+0.79), so "negative in 63/63 specifications" was true only
    # of the per-pair keeps and must not be quoted. Exclusion is 0/63 either way, so the verdict
    # is unchanged -- the honest phrasing just gets WEAKER: indistinguishable, point estimate
    # usually behind. Peer af5680b found this defect in their own block first (their residual
    # 0.3811); their predicted mechanism ("the arm that degenerates more gains more") is
    # MEASURED FALSE, so neither of us may argue the defect is conservative in any direction.
    allkeys = set.intersection(*[set(arms[n][0]) for n in arms])
    common_band: dict = {}
    print()
    print("===== SPECIFICATION CURVE over the degeneracy detector (symmetric exclusion) =====")
    print(f"Every cell below drops rows degenerate in EITHER arm, over {len(grid)} detector settings")
    print("(nmin/nmax x rep x phrase_loop coverage), nmax spanning 8..20. Neither the fraction nor")
    print("a crossover is a property of the models: exclusion is non-monotone in nmax in 12/27")
    print("settings and every near-crossover exclusion is knife-edge (hi as small as -0.001).")
    print("Quote the SIGN and the BAND -- and prefer the ESTIMABLE-ONLY band on each row.")
    print(f"{'claim':24s} {'point range':>18s} {'sign':>7s} {'excludes 0':>12s}  verdict")
    summary: dict = {}
    for a, b, label in claims:
        sa, ra = arms[a]
        sb, rb = arms[b]
        shared = sorted(set(sa) & set(sb))
        means, nexcl = [], 0
        min_keep, min_per_ds = 10 ** 9, 10 ** 9
        gated_means, gated_excl = [], 0
        for nmin, nmax, rep, cov in grid:
            f = pred(nmin, nmax, rep, cov)
            keep = [k for k in shared
                    if not (k in ra and f(ra[k])) and not (k in rb and f(rb[k]))]
            m, lo, hi, _p, _k = cluster_t([(k[0], sa[k] - sb[k]) for k in keep])
            means.append(m)
            if lo > 0 or hi < 0:
                nexcl += 1
            # COMMON kept set: drop a row if ANY of the FOUR arms is degenerate on it. This is the
            # only version whose rows may be compared with each other or subtracted (see 9c).
            ck = [k for k in allkeys
                  if not any(f(arms[n][1][k]) for n in arms if k in arms[n][1])]
            mc, loc, hic, _pc, _kc = cluster_t([(k[0], sa[k] - sb[k]) for k in ck])
            cb = common_band.setdefault(label, [[], 0, [], []])
            cb[0].append(mc)
            cb[1] += (loc > 0 or hic < 0)
            # ⭐ CORRECTION 9f. The ESTIMABLE-ONLY bracket below gates on `keep` (PER-PAIR), so
            # after 9c the report had a common-set band with NO n_keep discipline attached -- the
            # 9c defect surviving inside the fix for 9c. Peer af5680b asked the right question:
            # is the common-set SIGN FLIP estimable? Measured on my grid: the 3 positive point
            # estimates for `Global - CONTROL` have min-rows/dataset in {1,2}, 0 of them survive
            # a min>=6 gate, and the largest is +0.79 [-4.70,+6.28] at min-rows=1. Gated, the
            # band returns to -0.66..-0.28, all negative. So record min-rows per config HERE, on
            # the common set, and print the gated common band. Their 567-config sweep reaches
            # min-rows 0 (a whole dataset vanishing, k<9), which my 63 configs also reach.
            cds: dict = {}
            for k in ck:
                cds[k[0]] = cds.get(k[0], 0) + 1
            c_ndone = len(cds)
            c_min = min(cds.values()) if cds else 0
            cb[2].append((c_min, c_ndone))
            cb[3].append((mc, loc, hic))
            # ⭐ MY OWN STANDING RULE, APPLIED HERE FOR THE FIRST TIME: any "after excluding
            # artefact rows" delta must print how many rows are LEFT, and a dataset reduced to a
            # couple of rows is not estimable. The dense sweep shows exactly why -- exclusion
            # re-enters significance at nmax=14 when min rows/dataset has fallen to 7, i.e. the
            # "significance" there is a handful of rows moving a dataset mean, not an effect.
            by_ds: dict = {}
            for k in keep:
                by_ds.setdefault(k[0], []).append(1)
            min_keep = min(min_keep, len(keep))
            this_min = min((len(v) for v in by_ds.values()), default=0)
            min_per_ds = min(min_per_ds, this_min)
            # GATED band: only configs where every dataset still has >=10 of its 50 rows. It
            # MATTERS -- the ungated band's extremes are set by configs where some dataset is down
            # to a single row: Global-gla +1.42..+4.36 becomes +2.95..+3.83, rwkv7-gla
            # +1.95..+5.62 becomes +3.31..+3.68. So the wide bands I was about to publish as "the
            # robust object" were themselves partly an artefact of unestimable configs.
            if len(by_ds) == 9 and this_min >= 10:
                gated_means.append(m)
                if lo > 0 or hi < 0:
                    gated_excl += 1
        sgn = "stable" if len({m > 0 for m in means}) == 1 else "FLIPS"
        worse = means[0] < 0
        # WHOSE arm is the left operand? `CONTROL rwkv7 - gla` is baseline-vs-baseline, so
        # "ours better/worse" is meaningless there -- and printing "ours better" on the CONTROL row
        # is the single most misleading thing this block could say, since the whole point of that
        # row is that the margin belongs to the official baseline, not to us. Caught when the
        # SIGN-ONLY label landed on it.
        ours = a in OURS
        direction = ("OURS WORSE" if worse else "ours better") if ours else \
                    ("left arm lower" if worse else "left arm higher") + " (BOTH baselines)"
        # THREE-WAY, not two-way. My first version keyed the verdict on CI-exclusion alone, so
        # `nexcl == 0` printed "NULL / tie in every spec" -- which labelled Chunked - CONTROL
        # (-3.76..-2.81, sign negative in 36/36) a TIE. That is a real effect with a consistent
        # direction that merely fails a significance threshold, and calling it a tie FLATTERED OUR
        # OWN ARM by erasing a consistent deficit. Sign-stability and CI-exclusion are separate
        # axes and the label has to name both.
        if nexcl == len(grid):
            v = f"ROBUST: significant in every spec ({direction})"
        elif sgn == "FLIPS":
            v = "NULL: sign not even stable"
        elif nexcl == 0:
            v = (f"consistent direction {len(grid)}/{len(grid)} but NOT significant in any spec"
                 f" -- {direction}")
        else:
            # The count is printed in its own column for auditability, but the LABEL must not turn
            # it into a claim: k/N ranges 0%..100% on identical data depending on how the grid is
            # weighted in nmax, and the crossover it implies is not even monotone (CORRECTION 9).
            v = ("SIGN-ONLY: direction stable, significance grid-dependent -- quote the band, "
                 + direction)
        print(f"{label:24s} {min(means):+8.2f}..{max(means):+7.2f} {sgn:>7s} "
              f"{nexcl:6d}/{len(grid):<5d} {v}")
        gb = (f"{min(gated_means):+.2f}..{max(gated_means):+.2f}, {gated_excl}/{len(gated_means)} excl"
              if gated_means else "NO config estimable")
        print(f"{'':24s} {'':>18s} {'':>7s} {'':>12s}  "
              f"[PER-PAIR worst-case n_keep {min_keep}/{len(shared)}, min rows in any dataset "
              f"{min_per_ds}; PER-PAIR ESTIMABLE-ONLY (every dataset >=10 rows): {gb}"
              f"  -- does NOT transfer to the common set, see 9f]")
        summary[label] = (min(means), max(means), nexcl, len(grid), sgn, min_keep, min_per_ds)
    # ⭐ CORRECTION 9c: the common-keep-set column and the identity residual. The per-pair column
    # above is the correct unit for each row ALONE; only this column may be compared across rows.
    print()
    print("  -- SAME 63 SPECS ON ONE COMMON KEPT SET (drop a row degenerate in ANY of the 4 arms).")
    print("     The per-pair bands above each use their own kept set (|K| 165..366), so they are")
    print("     DIFFERENT ESTIMANDS and must not be subtracted or ranked against each other.")
    print(f"  {'claim':24s} {'per-pair band':>19s} {'excl':>7s} | {'COMMON-SET band':>19s} {'excl':>7s}"
          f" {'minrows':>8s} {'k<9':>4s}")
    for _a, _b, label in claims:
        pl, ph, nx, ng2, _sg, _mk, _mp = summary[label]
        cm, cx, cmr, cci = common_band[label]
        flip = "  <-- SIGN FLIPS vs per-pair" if (min(cm) < 0) != (max(cm) < 0) and pl * ph > 0 else ""
        wmr = min(r[0] for r in cmr)
        nlost = sum(1 for r in cmr if r[1] < 9)
        print(f"  {label:24s} {pl:+8.2f}..{ph:+8.2f} {nx:4d}/{ng2} | "
              f"{min(cm):+8.2f}..{max(cm):+8.2f} {cx:4d}/{ng2} {wmr:8d} {nlost:4d}{flip}")
    resid = max(abs(common_band["OUR Global - gla"][0][i]
                    - common_band["CONTROL rwkv7 - gla"][0][i]
                    - common_band["OUR Global - CONTROL"][0][i]) for i in range(len(grid)))
    print(f"     identity |(G-gla)-(C-gla)-(G-C)| on the common set: {resid:.2e} (must be ~0);")
    print("     under the per-pair keeps printed above it reaches 1.0698 -- four times the")
    print("     +0.29..+0.38 gap that the attribution argument turns on. If three rows of a table")
    print("     satisfy A-B=C, PRINT THE RESIDUAL; mine was 1.07 and I would never have looked.")
    _gc = common_band["OUR Global - CONTROL"][0]
    print(f"  ⚠️ `Global - CONTROL` loses sign-stability on the common set "
          f"({min(_gc):+.2f}..{max(_gc):+.2f}), so")
    print("     'negative in 63/63 specifications' was a property of the per-pair keeps ONLY.")
    print("     This does NOT help us: exclusion is 0/63 either way, and the honest phrasing gets")
    print("     WEAKER -- indistinguishable from the official same-size baseline, point estimate")
    print("     usually behind. Do not read a lost deficit as a gain.")
    # ⭐ CORRECTION 9f: the n_keep gate applied to the COMMON set, which is where it was missing.
    # The `minrows`/`k<9` columns above and the gated band below are the common set's own; the
    # ESTIMABLE-ONLY bracket on each per-pair row is computed from the PER-PAIR keeps and does
    # NOT transfer. That mattered: per-pair, `Global - CONTROL` keeps >=11 rows in every dataset
    # at all 63 settings, which is what let me write "estimable throughout"; on the common set NO
    # config on this grid reaches min-rows 10 (max 8) and 4 configs lose a whole dataset (k=8).
    print("  -- n_keep GATE ON THE COMMON SET (my own rule, previously applied only to the")
    print("     per-pair column -- the 9c defect surviving inside the fix for 9c):")
    for _a, _b, label in claims:
        cm, cx, cmr, cci = common_band[label]
        pos = [i for i in range(len(cm)) if cm[i] > 0]
        mrs = sorted({cmr[i][0] for i in pos})
        g6 = [i for i in range(len(cm)) if cmr[i][0] >= 6 and cmr[i][1] == 9]
        gs = (f"{min(cm[i] for i in g6):+.2f}..{max(cm[i] for i in g6):+.2f}, "
              f"{sum(1 for i in g6 if cm[i] > 0)}/{len(g6)} pos, "
              f"{sum(1 for i in g6 if cci[i][1] > 0 or cci[i][2] < 0)}/{len(g6)} excl"
              if g6 else "no config estimable at min>=6")
        print(f"     {label:24s} POS {len(pos):2d}/{len(cm)} at min-rows {mrs if pos else '-'};"
              f" GATED min>=6: {gs}")
    # ⭐ ADJACENCY GUARD (peer af5680b, and it is their own 53i error and my 9b error both). The two
    # rows printed above against gla -- `Global - gla` and `rwkv7 - gla` -- are two margins against
    # a SHARED THIRD ARM. Displayed side by side, and especially once the gate lifts both to
    # "excludes 0 in every config", a reader reads a ranking off them. That ranking is not theirs to
    # read: the paired difference is `Global - CONTROL`, it is ALGEBRAICALLY the headline (residual
    # below), and it excludes 0 in NO configuration. TWO EXCLUSIONS DO NOT COMPOSE INTO A THIRD.
    # Rule: never print two margins against a shared arm without the paired row beside them.
    _gg, _cg2, _gc2 = (common_band["OUR Global - gla"], common_band["CONTROL rwkv7 - gla"],
                       common_band["OUR Global - CONTROL"])
    _gate = [i for i in range(len(grid)) if _gc2[2][i][0] >= 6 and _gc2[2][i][1] == 9]

    def _bd(cbx, idx):
        m = [cbx[0][i] for i in idx]
        e = sum(1 for i in idx if cbx[3][i][1] > 0 or cbx[3][i][2] < 0)
        return f"{min(m):+.2f}..{max(m):+.2f} excl {e}/{len(idx)}" if idx else "no config"
    print("     ⚠️ ADJACENCY: the two rows above against gla are margins against a SHARED arm.")
    print(f"        GATED min>=6 ({len(_gate)} configs):  OURS-gla {_bd(_gg, _gate)}  |  "
          f"CTRL-gla {_bd(_cg2, _gate)}")
    print(f"        PAIRED difference (= `Global - CONTROL`, the ONLY licensed comparison of those")
    print(f"        two margins):        {_bd(_gc2, _gate)}")
    _ri = (max(abs(_cg2[0][i] - _gg[0][i] - (-_gc2[0][i])) for i in _gate) if _gate else 0.0)
    print(f"        identity |(C-gla)-(O-gla)+(O-C)| on the gated set: {_ri:.2e} -- so that gap is")
    print(f"        not independent evidence, it IS the headline tie restated. Both margins can")
    print(f"        exclude 0 in every config while their difference excludes 0 in none.")
    # ⭐ SECOND ROUTE to the same illegitimate ranking, found in my own output while verifying the
    # guard above: the per-row `excl` COUNTS are also printed adjacently, and ungated they differ
    # (45/63 ours vs 51/63 the control). A reader can rank off "significant in more specifications"
    # without ever comparing the bands, so a guard that only covers the intervals is incomplete.
    # The counts are k/N over a grid, which MEMORY already records as unreportable on its own.
    _eo = sum(1 for i in range(len(grid)) if _gg[3][i][1] > 0 or _gg[3][i][2] < 0)
    _ec = sum(1 for i in range(len(grid)) if _cg2[3][i][1] > 0 or _cg2[3][i][2] < 0)
    _ep = sum(1 for i in range(len(grid)) if _gc2[3][i][1] > 0 or _gc2[3][i][2] < 0)
    print(f"        ⚠️ SAME for the EXCLUSION COUNTS printed per row above: ungated OURS-gla")
    print(f"           {_eo}/{len(grid)} vs CTRL-gla {_ec}/{len(grid)} -- 'significant in more specs'")
    print(f"           is the same ranking by another route, and it is refuted by the same number:")
    print(f"           the paired difference is {_ep}/{len(grid)}. Do not rank two k/N counts either.")
    print("     => On THIS grid the positive point estimates of `Global - CONTROL` all sit at")
    print("        min-rows in {1,2} and none survives the gate, so the accurate statement is not")
    print("        'the sign is not stable' but 'the sign flip exists only in configurations that")
    print("        are NOT estimable'. Peer af5680b measures the same on a disjoint 567-config")
    print("        grid (63 positives, all min-rows in {1,2}, 0 estimable, gated -1.11..-0.28).")
    print("     ⚠️ Report the PAIR, not either half: a sign flip and an ESTIMABLE sign flip are")
    print("        different objects, as are a sign flip and a CI exclusion. The gated version")
    print("        favours us, and it is conditional on a filter introduced after seeing the flip,")
    print("        so it does not restore 'consistently behind'. What is invariant across per-pair,")
    print("        common AND gated is the only thing that mattered: 0/63 exclude 0 (0/567 theirs).")
    print("     ⚠️ min-rows 0 means an entire dataset vanished (k<9), so those configs are not")
    print("        9-cluster estimates at all -- a stronger failure than a small n_keep.")
    print()
    # EVERY number in the prose below is interpolated from `summary`. The previous version wrote
    # "negative in 36/36 specs ... clears significance in NO spec" as literal text, and widening
    # the grid to straddle the nmax transition turned that into a FALSE statement printed directly
    # under a table contradicting it (Chunked - CONTROL is significant in 23/63, not 0). That is
    # the same hardcoding defect as the old pre-EOS-fix numbers under a post-fix table, and it
    # again ran in our favour -- "significant in no spec" understates our own arm's deficit.
    lo_c, hi_c, nx_c, ng, _s, mk_c, mp_c = summary["OUR Chunked - CONTROL"]
    lo_g, hi_g, nx_g, _n, _s2, mk_g, mp_g = summary["OUR Global - CONTROL"]
    # ⭐ "stays negative on the common kept set" and the Global-CONTROL band are both COUNTED from
    # `common_band` here. They were literal text; a claim about the common set must be read off the
    # common set, especially in the block whose whole subject is that the two keep conventions
    # disagree.
    _cc = common_band["OUR Chunked - CONTROL"][0]
    _ccneg = sum(1 for v in _cc if v < 0)
    _gcm = common_band["OUR Global - CONTROL"][0]
    print(f"  => The SIGN is the robust object; CI-exclusion is not. `Chunked - CONTROL` is negative")
    print(f"     in {ng}/{ng} specs ({lo_c:+.2f}..{hi_c:+.2f}) ON THE PER-PAIR KEEPS, and negative in")
    print(f"     {_ccneg}/{ng} ({min(_cc):+.2f}..{max(_cc):+.2f}) on the common kept set, so our chunked")
    print(f"     arm IS worse than the official 2.9B baseline; it clears significance in {nx_c}/{ng}")
    print(f"     per-pair. Report the SIGN and")
    print(f"     the BAND only: '{lo_c:+.2f}..{hi_c:+.2f}, negative in {ng}/{ng} specifications'. Do NOT")
    print(f"     quote {nx_c}/{ng} and do NOT quote a threshold -- see the two reasons below.")
    print(f"     ⚠️ The same sentence is NOT available for `Global - CONTROL`: that row is negative")
    print(f"     {ng}/{ng} per-pair but spans {min(_gcm):+.2f}..{max(_gcm):+.2f} on the common kept")
    print(f"     set (see the 9c/9f blocks above).")
    print(f"  => WHY NO FRACTION: {nx_c}/{ng} is a property of how many configs this grid happens to")
    print(f"     place below the crossover, not of the models. Same data, same predicate, same")
    print(f"     symmetric exclusion, champ-rwkv7: nmax{{12,20}} -> 0%, this grid -> {100*nx_c//ng}%,")
    print(f"     peer's nmax{{9,13,17}} -> 33%, nmax{{8,9,10}} -> 100%. Credit peer af5680b.")
    print(f"  => WHY NO THRESHOLD EITHER (measured, not assumed): reporting 'excludes 0 only for")
    print(f"     nmax<=N' fails twice. (1) Exclusion is NOT monotone in nmax in a large minority of")
    print(f"     settings -- 12/27 for Chunked-CONTROL, 9/27 for Chunked-Global; at nmin=4 rep=3")
    print(f"     cov=0.80 it holds to nmax=10, is wide at 11-13 and RETURNS at 14, so there is")
    print(f"     often no single crossover. (2) Every exclusion next to a crossover is a KNIFE EDGE:")
    print(f"     hi=-0.0013 at nmax=10 (rep=3 cov=0.60), -0.0616 at nmax=8, -0.0484 at the nmax=15")
    print(f"     re-entry. A significance flip driven by 0.001 of ROUGE is a property of the")
    print(f"     threshold, not of the models. k stays 9 throughout, so this is not clusters")
    print(f"     dropping out -- it is per-dataset cells collapsing (min rows/dataset 19 -> 7 -> 1).")
    print(f"     Worst case on this grid: n_keep {mk_c}, min rows in a dataset {mp_c}.")
    print(f"     'Excludes 0 only for nmax<=10' is the same single-config claim as the fraction.")
    print(f"     (The -3.69 I first published as 'significant' fails for a THIRD, independent")
    print(f"     reason: it is the asymmetric excl-rwkv7 column, which drops only the BASELINE's")
    print(f"     degenerate rows and excludes 0 at every nmax 9..25 -- asymmetry alone produced it.)")
    # ⭐ The common-set band here is INTERPOLATED, not typed. I hardcoded "-2.00..+0.79" in this
    # sentence when 9c added it -- the same defect as the pre-EOS-fix numbers under a post-fix
    # table and as the "significant in no spec" line the widened grid falsified. A derived number
    # typed into prose stops tracking the data the moment anything upstream moves.
    _cg = common_band["OUR Global - CONTROL"]
    _cgm, _cgx = _cg[0], _cg[1]
    print(f"  => `Global - CONTROL` is never SIGNIFICANTLY positive in any spec ({lo_g:+.2f}..{hi_g:+.2f}")
    print(f"     per-pair, {min(_cgm):+.2f}..{max(_cgm):+.2f} on the common kept set with {_cgx}/{ng}")
    print(f"     excluding 0 there; {nx_g}/{ng} significant either way per-pair, and it")
    print(f"     excludes 0 at NO nmax from 6 to 26 under any of the")
    print(f"     27 other-parameter settings). That is the comparison deciding whether our method")
    print("     beats a same-size official baseline: it is at best a tie, never a win, and no")
    print("     detector setting rescues it. Both 'ours better' rows above are against gla ONLY,")
    print("     and the CONTROL row shows the official baseline collects the same margin from the")
    print("     same fix.")


def whose_rows_drive_champ_gla() -> None:
    """Split champ - gla by WHOSE rows are degenerate. Exists because I got this backwards.

    I wrote "the chunked arm's entire apparent edge over gla was carried by rows where champ is
    itself looping" and cited +0.03 -- but +0.03 is the excl-GLA column, which removes GLA's
    degenerate rows. A claim about rows where CHAMP loops needs the excl-CHAMP column.

    ⭐ AN EXCLUSION COLUMN IS NAMED FOR THE ROWS IT REMOVES. Both columns pointed the same
    direction (both are nulls), which is exactly why the mislabel survived my own review: a
    conclusion can be correct while its stated mechanism names the wrong arm. Credit: peer agent
    af5680b, who split it rather than accepting a number that happened to agree with them.

    The truth is WORSE FOR US than what I wrote. My version blamed a champ artefact, which sounds
    like something a better champ checkpoint would fix. The split says the comparison never
    contained a champ advantage at all:

      dropping CHAMP's degenerate rows leaves the edge INTACT  +1.52 -> +1.60
      dropping GLA's degenerate rows DESTROYS it               +1.52 -> +0.03

    and the means say why: on gla-degenerate rows champ 11.75 vs gla 6.62 (+5.14); on gla-CLEAN
    rows champ 7.82 vs gla 7.86 = -0.05, i.e. nothing. gla scores 6.62 on its own degenerate rows
    against 7.86 on its clean ones, so the +1.52 is GLA BEING PENALISED FOR LOOPING, not champ
    doing anything. The two degenerate sets are largely disjoint (gla-only 81, champ-only 54,
    both 55), so this is not one slice appearing under two names.

    Printed, not just documented, because "champ has no measurable advantage over gla on rows
    where gla is not degenerate" is the publishable sentence and it should be re-derived on every
    run.
    """
    cdoc, _cs = load_best_cell("champ-dynamic-30k")
    gdoc, _gs = load_best_cell("gla-1.3b-bos-loop")
    if cdoc is None or gdoc is None:
        return
    sc, sg = row_scores(cdoc), row_scores(gdoc)
    rc = {(r["dataset"], r["id"]): r for r in cdoc.get("rows", [])}
    rg = {(r["dataset"], r["id"]): r for r in gdoc.get("rows", [])}
    shared = sorted(set(sc) & set(sg))
    dg = {k for k in shared if k in rg and is_degenerate_row(rg[k])}
    dc = {k for k in shared if k in rc and is_degenerate_row(rc[k])}
    print("          WHOSE degeneracy drives champ - gla (an exclusion column is named for the")
    print("          rows it REMOVES -- I first cited excl-gla as evidence about champ's rows):")
    for label, keys in (("all rows", shared),
                        ("ON gla-degenerate", [k for k in shared if k in dg]),
                        ("OFF gla-degen (excl-gla)", [k for k in shared if k not in dg]),
                        ("ON champ-degenerate", [k for k in shared if k in dc]),
                        ("OFF champ-degen (excl-champ)", [k for k in shared if k not in dc]),
                        ("neither degenerate",
                         [k for k in shared if k not in dg and k not in dc])):
        if len(keys) < 2:
            continue
        mm, lo, hi, _pos, _k = cluster_t([(k[0], sc[k] - sg[k]) for k in keys])
        mc = 100 * statistics.mean([sc[k] for k in keys])
        mg = 100 * statistics.mean([sg[k] for k in keys])
        print(f"            {label:30s} {mm:+6.2f} [{lo:+6.2f},{hi:+6.2f}] n={len(keys):3d}"
              f"  champ {mc:5.2f} vs gla {mg:5.2f}")
    print(f"            sets largely disjoint: gla-only {len(dg - dc)}, champ-only {len(dc - dg)},"
          f" both {len(dg & dc)}")
    print("            => dropping CHAMP's bad rows leaves the edge intact; dropping GLA's")
    print("               destroys it. The +1.52 is gla being penalised for looping, NOT champ")
    print("               capability. Publishable form: champ has NO measurable advantage over")
    print("               gla on rows where gla is not degenerate.")


def gla_warning() -> None:
    """The gla-1.3b comparison, RECOMPUTED -- the one place a positive verdict now survives.

    This block used to print hardcoded pre-EOS-fix numbers ("+2.06 and +1.52, both UNRESOLVED
    once clustered ... neither arm beats ANY baseline"). Honouring rwkv7/global-dit's stop ids
    moved our arm, and global-dit vs gla is now +4.49 [+0.78,+8.21], which EXCLUDES 0. The stale
    prose therefore contradicted the table printed 20 lines above it, in the direction of
    understating our own result -- so it cannot simply be re-asserted, and it cannot be left
    either. It is computed live.

    THE VERDICT IS STILL NOT QUOTABLE AS A WIN, but the reason is NOT the one I first wrote, and
    the retraction is recorded because the discarded version favoured us:

    ✗ RETRACTED: "the margin is a stopping artefact -- gla never emits EOS, so it spends its whole
      512-token budget against a ~213-word gold and ROUGE-L F1 charges the surplus as lost
      precision." Two controls killed it. (a) LENGTH-MATCHED RESCORE: truncating each gla
      prediction to the word count our arm used and rescoring with the benchmark's own
      rouge==1.0.1 moves gov_report 9.51 -> 9.52, qmsum 9.01 -> 9.01, multi_news 7.54 -> 7.28 --
      the deltas change by at most 0.27. The control was very nearly a NO-OP, and checking why
      showed the premise was false: gla is SHORTER than our arm in words on the datasets in
      question (multi_news median 150 words vs our 137, qmsum 172 vs 412), because 512 TOKENS is
      only ~150-190 words at its 3.0-3.5 tokens/word -- and that ratio is itself the tell.
      (b) ROUGE-L DECOMPOSED: if surplus text were the mechanism the gap would be in PRECISION.
      It is in RECALL -- gov_report P 45.1 vs our 48.6 (a 3.6 gap) against R 5.4 vs 9.1; qmsum
      P 19.0/30.7 against R 6.4/14.0. A precision story cannot explain a recall gap.

    ✓ WHAT IT ACTUALLY IS: phrase-level degeneration, which is a real defect of that cell but
      also the thing the deg% column already flags, so it must be quoted as an exclusion rather
      than as our margin. gla's long-cap rows are literally "1. The Occupy movement in Los
      Angeles has been peaceful. 2. <same sentence> 3. <same sentence> ..." and
      "Passage 1:\\n\\nPassage 2:\\n\\n... Passage 31:" -- high per-token precision against the
      source, almost no coverage of the gold, at 3.5 tokens/word. Degeneracy is 46.7% on the
      512-cap datasets vs 15.2% on the <=64 ones, i.e. concentrated exactly where the lead is.
      (Rates are from the WIDENED predicate -- see `local_repeat`. The narrower one gave 39.3%
      and 11.6%, and understated OUR OWN arm far more than gla's.)

    So the three objections that stand are, in order of strength:

    1. ATTRIBUTION -- THIS IS DECISIVE AND IT IS NOT ABOUT DEGENERACY OR SIZE. Apply the same
       EOS fix to the OFFICIAL RWKV-7 2.9B baseline and it reproduces the margin at the same
       magnitude: rwkv7-world3 - gla = +4.48 [+0.27,+8.69] 8/9 against our +4.49 [+0.78,+8.21]
       8/9 -- and, because those are two margins against a SHARED arm, the licensed form is the
       PAIRED one: global-dit - rwkv7 = +0.01 [-1.63,+1.66] 4/9, measured on the identical 450
       rows (symdiff 0). So the effect is what honouring the RWKV-7 tokenizer's stop ids does for
       ANYTHING sharing that tokenizer, not something our architecture does. This is the control
       that decides it: an effect an official baseline reproduces at equal size belongs to the
       fix, not to the method.
       ⛔ RETRACTED HERE (CORRECTIONS 9e and 9i): this paragraph used to add "on the
       degenerate-excluded set the BASELINE's margin is LARGER than ours (+3.26 vs +2.78)". Both
       numbers are correct and the row sets are identical (314/314, symdiff 0), but quoting two
       margins against a shared arm to establish an ORDERING is exactly the defect 9i is about.
       The paired difference on those same 314 rows is +0.481 [-1.589,+2.550] -- it INCLUDES 0 --
       and it is algebraically this same `global-dit - rwkv7` restricted to those rows (identity
       residual 4.44e-16). So it adds no evidence: it is the headline tie restated on a subset.
       The attribution argument does not need it and never did.

       I did not find this. A peer agent ran it against my +4.49 and I reproduced it before
       accepting it. My own framing at the time ("size, plus both sides degenerate") conceded
       strictly less than the data required -- those objections make it a WEAK result; this one
       makes it NOT OURS. Recorded because the error was in our favour.

    2. SIZE. `gla-1.3B-100B` is hidden 2048 x 24 layers = 1.3B params vs our 2.9B, a 2.2x gap.
       ("100B tokens" is the HF REPO NAME, not a measured or configured quantity -- `config.json`
       carries no token-budget field, so it is cited as repo metadata and not as our datum.)
       Beating it would be a scale result; the paper's claim is about architecture.

    3. BOTH SIDES DEGENERATE on the slice supplying the lead, OURS INCLUDED. Excluding gla's
       degenerate rows with the SAME union detector the deg% column uses does NOT remove the
       margin (+2.78, still excluding 0), and neither does the SYMMETRIC exclusion that drops
       rows degenerate in EITHER arm (+3.46 [+0.90,+6.02]) -- asymmetric exclusion favours
       whoever is excluded against, so the symmetric form is the one to quote. Exclusion is
       therefore NOT what disposes of this; objection 1 is. What degeneracy does establish is
       that the slice is unfit: on the three 512-cap datasets (150 rows each) the union detector
       gives gla 40.0%, CHAMP 45.3%, rwkv7 11.3%, global-dit 10.7%.

       CHAMP IS THE MOST DEGENERATE ARM ON THAT SLICE, and I reported the opposite for a day.
       My `phrase_loop` gates on a repeat covering >=80% of the output, which sees gla's
       whole-output loops but is blind to champ's LOCALIZED 3x repeats (coverage 0.23-0.76).
       Measured asymmetry: adding `local_repeat` moves gla 59->60 and rwkv7 19->17 but champ
       28->68 on that slice. So a threshold tuned on the baseline's failure mode was exact for
       the baseline and blind for us. The narrower detector understated OUR arm by 3x, and I had
       published it while telling the peer their champ figure was unreproducible -- their 58.7%
       was right. Their two "88"s were two correct numbers on two different denominators
       (88/150 champ on the 512-cap slice, 88/450 gla across all nine) presented as one vector,
       which is why it looked like a copy error. If degeneracy disqualifies gla's rows it cannot
       be waved past for champ. But champ - gla +1.52 [-2.14,+5.18] is NOT what carries the
       disclosure, because it is a null once gla's rows are removed, and the mechanism is GLA's
       failure, not champ's -- see `whose_rows_drive_champ_gla` below. I first wrote that the
       edge 'came from rows where champ is itself looping', which named the wrong arm: that
       claim needs the excl-CHAMP column and I read it off the excl-GLA one.

    Reported rather than suppressed: hiding a positive verdict because it is inconvenient would
    be the same selective editing as quoting it, in the opposite direction. The honest sentence
    is the peer's: "the EOS fix lifts the whole RWKV-7 tokenizer family ~2.4 points on LongBench;
    both it and our arm then score above a 1.3B baseline with ~39% degenerate long-output rows;
    our arm does not separate from the 2.9B baseline (+0.01 [-1.63,+1.66])."
    """
    ours_cells = [("global-dit-50k", "global-dit-50k"), ("champ-dynamic-30k", "champ-dynamic-30k")]
    gla_doc, gla_src = load_best_cell("gla-1.3b-bos-loop")
    if gla_doc is None:
        return
    gla_rows = {(r["dataset"], r["id"]): r for r in gla_doc.get("rows", [])}
    exclude = {k for k, r in gla_rows.items() if is_degenerate_row(r)}
    gs = row_scores(gla_doc)
    print("WARNING : ours vs gla-1.3b-bos-loop is the ONE comparison where a positive verdict")
    print(f"          survives clustering (source {gla_src}) -- but an ATTRIBUTION CONTROL shows")
    print("          it is not ours. Three objections, strongest first; none is fixed by data:")
    for arm, cell in ours_cells:
        doc, _src = load_best_cell(cell)
        if doc is None:
            continue
        a = row_scores(doc)
        shared = sorted(set(a) & set(gs))
        arm_rows = {(r["dataset"], r["id"]): r for r in doc.get("rows", [])}
        allp = [(k[0], a[k] - gs[k]) for k in shared]
        keep = [(k[0], a[k] - gs[k]) for k in shared if k not in exclude]
        # SYMMETRIC exclusion: drop rows degenerate in EITHER arm. Excluding only the
        # baseline's bad rows tilts the test toward whoever is excluded against.
        sym = [(k[0], a[k] - gs[k]) for k in shared
               if k not in exclude and not (k in arm_rows and is_degenerate_row(arm_rows[k]))]
        m, lo, hi, pos, k = cluster_t(allp)
        m2, lo2, hi2, pos2, k2 = cluster_t(keep)
        m3, lo3, hi3, pos3, _k3 = cluster_t(sym)
        print(f"            {arm:18s} all n={len(shared):3d} {m:+.2f} [{lo:+.2f},{hi:+.2f}] {pos}/{k}"
              f" | excl-gla n={len(keep):3d} {m2:+.2f} [{lo2:+.2f},{hi2:+.2f}] {pos2}/{k2}"
              f" | excl-either n={len(sym):3d} {m3:+.2f} [{lo3:+.2f},{hi3:+.2f}] {pos3}/{_k3}")
    # THE CONTROL. Same fix, official 2.9B baseline. If it reproduces the margin, the margin
    # belongs to the fix. Computed live so it can never drift out of step with the table above.
    rw_doc, rw_src = load_best_cell("rwkv7-world3",
                                    aliases=PUBLISHABLE_ALIASES.get("rwkv7-world3", ()))
    if rw_doc is not None:
        rs = row_scores(rw_doc)
        rw_rows = {(r["dataset"], r["id"]): r for r in rw_doc.get("rows", [])}
        sh = sorted(set(rs) & set(gs))
        cm, clo, chi, cpos, ck = cluster_t([(x[0], rs[x] - gs[x]) for x in sh])
        keep_c = [(x[0], rs[x] - gs[x]) for x in sh if x not in exclude]
        em, elo, ehi, epos, ek = cluster_t(keep_c)
        print(f"          CONTROL -- the SAME fix applied to the official 2.9B baseline ({rw_src}):")
        print(f"            rwkv7-world3 - gla      all n={len(sh):3d} {cm:+.2f} [{clo:+.2f},{chi:+.2f}] {cpos}/{ck}"
              f" | excl-gla n={len(keep_c):3d} {em:+.2f} [{elo:+.2f},{ehi:+.2f}] {epos}/{ek}")
        od, _os = load_best_cell("global-dit-50k")
        if od is not None:
            oa = row_scores(od)
            o_rows = {(r["dataset"], r["id"]): r for r in od.get("rows", [])}
            sh2 = sorted(set(oa) & set(rs))
            dm, dlo, dhi, dpos, dk = cluster_t([(x[0], oa[x] - rs[x]) for x in sh2])
            dsym = [(x[0], oa[x] - rs[x]) for x in sh2
                    if not (x in rw_rows and is_degenerate_row(rw_rows[x]))
                    and not (x in o_rows and is_degenerate_row(o_rows[x]))]
            sm, slo, shi, spos, sk = cluster_t(dsym)
            print(f"            global-dit - rwkv7      all n={len(sh2):3d} {dm:+.2f} [{dlo:+.2f},{dhi:+.2f}] {dpos}/{dk}"
                  f" | excl-either n={len(dsym):3d} {sm:+.2f} [{slo:+.2f},{shi:+.2f}] {spos}/{sk}")
        print("            => the baseline reproduces the margin at the same magnitude and our")
        print("               arm does not separate from it. The effect is what honouring the")
        print("               RWKV-7 tokenizer's stop ids does for anything using that")
        print("               tokenizer -- NOT an architectural result. This is objection (1).")
    whose_rows_drive_champ_gla()
    short = [d for d in DATASETS if MAXLEN[d] <= 64]
    long_ds = [d for d in DATASETS if MAXLEN[d] == 512]
    doc, _s = load_best_cell("global-dit-50k")
    a = row_scores(doc)
    ours_rows = {(r["dataset"], r["id"]): r for r in doc.get("rows", [])}
    shared = sorted(set(a) & set(gs))
    for tag, sub in ((f"gen cap <=64 ({len(short)} ds)", short),
                     (f"gen cap 512  ({len(long_ds)} ds)", long_ds)):
        v = [(k[0], a[k] - gs[k]) for k in shared if k[0] in sub]
        m, _lo, _hi, _pos, k = cluster_t(v)
        nd = [(k2[0], a[k2] - gs[k2]) for k2 in shared if k2[0] in sub and k2 not in exclude]
        m3 = cluster_t(nd)[0]
        insub = [k2 for k2 in gla_rows if k2[0] in sub]
        dg = sum(1 for k2 in insub if k2 in exclude)
        dgo = sum(1 for k2 in insub if k2 in ours_rows and is_degenerate_row(ours_rows[k2]))
        # champ must appear here too: if degeneracy disqualifies gla's rows it cannot be waved
        # past for our chunked arm, whose rate on this slice is HIGHER than global-dit's.
        cdoc, _cs = load_best_cell("champ-dynamic-30k")
        crows = {(r["dataset"], r["id"]): r for r in (cdoc or {}).get("rows", [])}
        dgc = sum(1 for k2 in insub if k2 in crows and is_degenerate_row(crows[k2]))
        rdoc, _rs2 = load_best_cell("rwkv7-world3",
                                    aliases=PUBLISHABLE_ALIASES.get("rwkv7-world3", ()))
        rrows = {(r["dataset"], r["id"]): r for r in (rdoc or {}).get("rows", [])}
        dgr = sum(1 for k2 in insub if k2 in rrows and is_degenerate_row(rrows[k2]))
        n = max(1, len(insub))
        print(f"            {tag}: all {m:+.2f} | non-degen {m3:+.2f} over {k} ds | degenerate"
              f" gla {100*dg/n:.1f}% global-dit {100*dgo/n:.1f}%"
              f" champ {100*dgc/n:.1f}% rwkv7 {100*dgr/n:.1f}%")
    print("          (2) SIZE: gla is 1.3B (hidden 2048 x 24L) vs our 2.9B, a 2.2x gap -- that")
    print("              would make this a scale result, not a long-context one. ('100B tokens'")
    print("              is the HF REPO NAME `gla-1.3B-100B`, not a measured or configured")
    print("              value: config.json has no token-budget field. Cited as repo metadata.)")
    print("          (3) BOTH SIDES DEGENERATE on the slice supplying the lead, OURS THE WORST.")
    print("              gla's long-cap rows are literal sentence loops ('1. <s>  2. <s> ...'")
    print("              and 'Passage 1: .. Passage 31:'). Excluding gla's degenerate rows does")
    print("              NOT remove the margin, and neither does the SYMMETRIC exclusion that")
    print("              drops rows degenerate in EITHER arm (excl-either above) -- so exclusion")
    print("              is not what disposes of this, objection (1) is. What it does show is an")
    print("              unfit slice: see the per-cap rates above, where CHAMP is the MOST")
    print("              degenerate arm on the 512-cap slice. I reported the opposite for a day --")
    print("              `phrase_loop` needs a repeat to cover >=80% of the output, so it saw")
    print("              gla's whole-output loops and was blind to champ's LOCALIZED 3x repeats")
    print("              (coverage 0.23-0.76). Adding `local_repeat` moved gla 59->60 and rwkv7")
    print("              19->17 but champ 28->68: a detector tuned on the baseline's failure mode")
    print("              understated OUR OWN arm by 3x. (champ - gla is nonetheless a null for a")
    print("              DIFFERENT reason -- gla's failure, not champ's. See the split below.)")
    print("          RETRACTED, and it favoured us: I first wrote that the margin was a STOPPING")
    print("          artefact (gla never emits EOS, so its surplus text loses ROUGE-L precision).")
    print("          Two controls killed it. Length-matching each gla prediction to our arm's")
    print("          word count and rescoring with the benchmark's own rouge==1.0.1 moves the")
    print("          deltas by <=0.27 -- a near NO-OP, because gla is SHORTER in words than our")
    print("          arm (512 tokens is ~150 words at its 3.5 tok/word). And ROUGE-L decomposes")
    print("          to a RECALL gap (gov_report 5.4 vs 9.1) with precision nearly equal (45.1")
    print("          vs 48.6), which a surplus-text mechanism cannot produce.")
    print("          Against every baseline that is size-comparable AND saw the full prompt")
    print("          (qwen2.5-3b, llama-3.2-3b-bos) both arms remain clearly WORSE.")


def macro_from_rows(doc) -> float:
    """Macro over the 9 datasets, recomputed from `rows` (weight by DATASET, never by row)."""
    if not doc:
        return float("nan")
    by_ds: dict = {}
    for r in doc.get("rows", []):
        if "score" in r:
            by_ds.setdefault(r.get("dataset"), []).append(r["score"])
    vals = [100 * statistics.mean(by_ds[ds]) for ds in DATASETS if ds in by_ds]
    return statistics.mean(vals) if len(vals) == len(DATASETS) else float("nan")


def main() -> None:
    header = f"{'model':20s}" + "".join(f"{d[:9]:>10s}" for d in DATASETS) + f"{'AVG':>8s}"
    for label, loader in (("FIXED RUN", lambda m: load(NEW, m)),
                          ("OLD RUN (broken budget)", load_prefix)):
        print(f"\n===== {label} -- LongBench score x100 =====")
        print(header)
        for m in MODELS:
            doc = loader(m)
            if doc is None:
                note = "  (no pre-fix file)" if label.startswith("OLD") else "  (no result file)"
                print(f"{m:20s}{note:>40s}")
                continue
            aggs = doc.get("aggregates", {})
            cells, vals = "", []
            for ds in DATASETS:
                if ds in aggs:
                    v = 100 * aggs[ds]["score"]
                    vals.append(v)
                    cells += f"{v:10.1f}"
                else:
                    cells += f"{'-':>10s}"
            avg = f"{statistics.mean(vals):8.1f}" if len(vals) == len(DATASETS) else f"{'(partial)':>8s}"
            star = " *" if m in OURS else ""
            # This table deliberately reads ONE directory by literal cell name, because that is
            # what makes its decode/BOS contrasts apples-to-apples. The cost is that it can print
            # a cell the EOS fix has since superseded in ANOTHER directory (it printed 9.5 for
            # rwkv7 and global-dit while the publishable block below printed 12.0). Flag that
            # rather than repoint: swapping in the best cell per NAME would make `rwkv7-world3`
            # show 11.97 (loop) against `rwkv7-world3-loop` 9.55 (also loop), which reads as
            # "hf beats loop" and is the exact opposite of the truth. A stale number must be
            # labelled, not quietly replaced by one that invents a different wrong contrast.
            sup = ""
            if label == "FIXED RUN":
                # Check the alias set too, keyed by whichever publishable model pins this cell.
                # Without this the `rwkv7-world3-loop` row -- the exact file whose staleness
                # started all this -- goes UNMARKED, because its own name only ever resolves to
                # itself. The row that misled me must be the first one flagged.
                al = next((PUBLISHABLE_ALIASES.get(mod, ()) for mod, (c, _w) in PUBLISHABLE.items()
                           if c == m), ())
                best, best_src = load_best_cell(m, aliases=al)
                a, b = macro_from_rows(doc), macro_from_rows(best)
                if best_src and best_src != f"{NEW.name}/{m}" and a == a and b == b and abs(a - b) > 0.05:
                    sup = f"  SUPERSEDED -> {best_src} = {b:.1f}"
            print(f"{m:20s}{cells}{avg}{star}{sup}")
    print("\nSUPERSEDED -> = a more-corrected cell for that same model exists in another directory;")
    print("                the value shown on the left is stale. Use the PUBLISHABLE block below,")
    print("                which selects by provenance. Rows without the marker are current.")
    print("                It means 'do not quote the left-hand value'. It does NOT mean the EOS")
    print("                fix is worth the gap: the two cells can differ on more than one axis at")
    print("                once. rwkv7-world3 2.9 -> 12.0 is decode AND EOS (hf 2.9 -> loop 9.5 ->")
    print("                EOS-fixed 12.0); only the rwkv7-world3-loop and global-dit-50k rows")
    print("                isolate the EOS fix alone, at +2.4 each.")

    print("\n===== CELLS OUTSIDE THE SELECTION ALLOWLIST (repo-wide) =====")
    unread = audit_unread_cells()
    if not unread:
        print("none -- every LongBench cell for a publishable model lives in a read directory.")
    for path, tier, mac, rel, sel_src in unread:
        flag = "  <== INVESTIGATE" if rel == "OUTRANKS SELECTED" else ""
        print(f"  tier={tier} macro={mac:6.2f}  {rel:18s} vs {str(sel_src):28s} {path}{flag}")
    print("Listed because the allowlist protects the table SILENTLY. A peer's repo-wide sweep found")
    print("4 live pre-fix alias files for our two arms; this report had accounted for only 2. The")
    print("other two (longctx_decodefix/rwkv7-world3 md5 b3b4d7ce, longctx_globalarm/global-dit-50k")
    print("md5 ff3d353c, both macro 9.55) sit outside the allowlist, so they never reached the")
    print("table -- but that was luck, not design: the allowlist predates knowing they existed.")
    print("Nothing here needs action unless a row says OUTRANKS SELECTED.")

    print("\n===== TRUNCATION AUDIT (fixed run) =====")
    print(f"{'model':20s} {'window':>8s} {'budget':>8s} {'reason':<34s} {'kept%':>7s} "
          f"{'cut':>8s} {'empty%':>7s}")
    for m in MODELS:
        doc = load(NEW, m)
        if doc is None:
            print(f"{m:20s} {'-':>8s} {'-':>8s} {'(no result file)':<34s}")
            continue
        cfg = doc.get("config", {})
        win = cfg.get("positional_window")
        bud = cfg.get("effective_input_budget")
        reason = str(cfg.get("truncation_reason", "?"))[:33]
        rows = doc.get("rows", [])
        # `input_tokens` is the POST-truncation length, so bud/input_tokens is 1.0 by
        # construction and says nothing. Prefer `prompt_tokens_untruncated` (truncate_ids'
        # `stats`); jobs launched before that field existed do not have it, so fall back to
        # joining on the OLD run's `input_tokens`, which was measured at a 32768 budget that
        # bound only the ~20 narrativeqa rows over 32k -- close enough to the true prompt
        # length for every other row, and exact for the phi-2/gemma comparison that matters.
        # Without either, a row cannot distinguish "answered a 1.5k prompt badly" from
        # "was shown 1.5k of an 11k document".
        base = {(r["dataset"], r["id"]): r["input_tokens"]
                for r in (load_prefix(m) or {}).get("rows", [])}
        fracs = []
        for r in rows:
            full = r.get("prompt_tokens_untruncated") or base.get((r["dataset"], r["id"]))
            if full:
                fracs.append(min(1.0, r["input_tokens"] / max(1, full)))
        cut = sum(1 for f in fracs if f < 1.0)
        kept = statistics.median(fracs) * 100 if fracs else float("nan")
        cutcell = f"{cut}/{len(fracs)}" if fracs else "n/a"
        print(f"{m:20s} {str(win):>8s} {str(bud):>8s} {reason:<34s} {kept:7.1f} "
              f"{cutcell:>8s} {degenerate_rate(doc):7.1f}")
    print("\nkept%  = median fraction of the prompt the model actually saw "
          "(post-truncation / pre-truncation).")
    print("cut    = rows whose prompt was middle-truncated at all.")
    print("empty% = rows whose prediction was the empty string (the degeneracy fingerprint).")
    print("*      = our arms.")

    # 2026-09-21, SECOND INSTANCE of the stale-alias bug: this block called `load(NEW, m)` while
    # the scores table and the publishable block had already been repointed at `load_best_cell`.
    # So the audit was describing DIFFERENT FILES from the ones being scored, for exactly the two
    # cells the EOS fix touched: it reported rwkv7 stop%=5.6 / deg%=76.9 and global-dit
    # stop%=0.0 / deg%=12.0, where the SELECTED cells are 31.3 / 15.3 and 17.6 / 12.4. The 0.0
    # was the worst of it -- it is the fingerprint of the never-stopping defect, printed next to
    # a score produced by a cell that stops on 79/450 rows. Finding a bug I had already fixed one
    # block over is the argument for fixing it at the SELECTOR, not per call site: every read of a
    # publishable score must come from `load_best_cell`, and `load(NEW, ...)` must appear only in
    # the explicit runner-matched comparisons that want one specific directory.
    print("\n===== VALIDITY AUDIT (fixed run) -- read this before quoting any score =====")
    print(f"{'model':20s} {'decode':>7s} {'bos':>5s} {'empty%':>7s} {'collapse%':>10s} {'stop%':>7s} "
          f"  {'source cell (must match the scores table)':42s}")
    for m in MODELS:
        al = next((PUBLISHABLE_ALIASES.get(mod, ()) for mod, (c, _w) in PUBLISHABLE.items()
                   if c == m), ())
        doc, src = load_best_cell(m, aliases=al)
        if doc is None:
            print(f"{m:20s} {'(no result file)':>34s}")
            continue
        cfg = doc.get("config", {})
        dec = str(cfg.get("decode", "hf"))
        # A MISSING prepend_bos key means the run predates the BOS fix entirely, which is not
        # the same as "ran with BOS off" -- those cells cannot be compared with later ones at
        # all. Distinguish the two rather than collapsing both to False.
        bos = "n/a" if "prepend_bos" not in cfg else str(bool(cfg["prepend_bos"]))
        print(f"{m:20s} {dec:>7s} {bos:>5s} {degenerate_rate(doc):7.1f} {collapse_rate(doc):10.1f} "
              f"{stop_rate(doc):7.1f}   {src:42s}")
    print("\ncollapse% = NON-EMPTY rows that are degenerate, by the SAME predicate as the deg% column")
    print("            in the PUBLISHABLE block (<=2 distinct ids, whole-output phrase loop, or a")
    print("            localized 3x repeat in ids or words). empty% + collapse% = deg%, by")
    print("            construction. Until 2026-09-21 this column re-implemented a NARROWER test and")
    print("            printed champ 8.0% where deg% said 24.2% -- see collapse_rate's docstring.")
    print("            Non-empty but worthless; a high value means the row is a harness/decode")
    print("            defect, not a score.")
    print("stop%     = rows that emitted EOS before the official cap. Near 0 means the model "
          "burns the whole")
    print("            cap, which dilutes token-overlap F1 against a 1-3 word gold -- a "
          "cross-model comparability")
    print("            caveat, not a bug in the model.")
    print("bos       = the resolved per-model BOS decision (tokenizer probed, not declared).")
    print("            'n/a' = the run predates the fix. That alone does NOT condemn a cell: for")
    print("            the 7 models with no leading BOS, n/a and False are the same tokenization.")
    print("            It is disqualifying only for gemma-2-2b / gla-1.3b / llama-3.2-3b, whose")
    print("            tokenizers do prepend one -- intersect the two facts, do not use n/a alone.")

    print("\n===== PUBLISHABLE ROW PER MODEL =====")
    print(f"{'model':20s} {'selected cell (provenance)':34s} {'AVG':>7s} {'95% CI':>16s} "
          f"{'deg%':>6s}  why")
    for model, (cell, why) in PUBLISHABLE.items():
        doc, src = load_best_cell(cell, aliases=PUBLISHABLE_ALIASES.get(model, ()))
        if doc is None:
            print(f"{model:20s} {cell+' (pending)':34s} {'pending':>7s} {'':>16s} {'':>6s}  {why}")
            continue
        # Recompute the macro from `rows` rather than reading `aggregates`. The two can disagree:
        # `aggregates` is a stored summary, so a partially-rewritten file can carry a stale one,
        # and the macro is the number every verdict below is built on. Verified equal to the
        # stored aggregates on all 10 selected cells at the time of writing, so this is a guard,
        # not a correction. Weight by DATASET (mean of 9 dataset means), never by row: an
        # unweighted mean over 450 rows is a different estimand whenever a cell is partial.
        by_ds: dict = {}
        for r in doc.get("rows", []):
            if "score" in r:
                by_ds.setdefault(r.get("dataset"), []).append(r["score"])
        per_ds_mean = [100 * statistics.mean(by_ds[ds]) for ds in DATASETS if ds in by_ds]
        avg = (f"{statistics.mean(per_ds_mean):.1f}" if len(per_ds_mean) == len(DATASETS)
               else f"partial {len(per_ds_mean)}/9")
        rows = [r["score"] for r in doc.get("rows", []) if "score" in r]
        lo, hi = boot_ci(rows)
        # Degeneracy: empty OR single-token-repeat. A model can only be compared on score if
        # this is near zero; otherwise the cell measures our harness, not the model.
        deg = degenerate_union_rate(doc)
        print(f"{model:20s} {src:34s} {avg:>7s} {f'[{lo:.1f}, {hi:.1f}]':>16s} {deg:6.1f}  {why}")
    print("\nAVG    = mean over the 9 official datasets (each 50 samples), recomputed from rows.")
    print("cell   = the file actually read, chosen by PROVENANCE (stop_ids > budget key > neither)")
    print("         over longctx/, longctx_fix/ and longctx_eosfix/. It is printed because the")
    print("         directories are live: the EOS fix landed in longctx/rwkv7-world3 while the")
    print("         name this report had pinned (longctx_fix/rwkv7-world3-loop) stayed at 9.5,")
    print("         and re-running produced a byte-identical table. Silent staleness, so the")
    print("         selected path now has to appear in the output.")

    # Our arms vs every baseline, PAIRED. This is the block that answers "are we competitive or
    # far behind", and it has to be paired: all models see the same 450 rows, so an unpaired
    # comparison throws away the pairing and is both weaker and (when two marginal CIs merely
    # touch) uninterpretable.
    print("\n===== OUR ARMS vs BASELINES (paired, per-row difference) =====")
    print("Positive delta = OUR ARM IS BETTER. CI excluding 0 = the margin is real. The CI is a")
    print("cluster-t over the 9 datasets; a row-level CI here is WRONG and inflates our arms.")
    print(f"{'ours':18s} {'vs baseline':22s} {'n':>4s} {'delta':>7s} {'95% CI (paired)':>18s} "
          f"{'up/down':>10s} {'worst':>7s}  verdict")
    # Every cell here goes through load_best_cell, the SAME selector as the publishable block.
    # Using `load(NEW, ...)` here (as this block originally did) silently differenced the stale
    # 9.55 global-dit against EOS-corrected baselines: two different generation configurations,
    # which is the "differencing two different scorers" error one level up. Mixing tiers is only
    # legitimate because the correction provably cannot touch the un-corrected cells: champ emits
    # id 0 zero times and id 2 zero times across all 86270 of its generated tokens, so unioning
    # {0,2} into its stop set cannot change a single row. That is a measurement, not an argument
    # from the metric -- and note it is NOT refuted by break-before-append, which hides ids that
    # FIRED, whereas this count is over ids appearing mid-generation in rows that ran to the cap.
    for ours in ("global-dit-50k", "champ-dynamic-30k"):
        ours_doc, ours_src = load_best_cell(ours)
        ours_rows = row_scores(ours_doc)
        if not ours_rows:
            print(f"{ours:18s} {'(no result file yet)':22s}")
            continue
        for base, (cell, _why) in PUBLISHABLE.items():
            if base in OURS:
                continue
            base_doc, base_src = load_best_cell(cell, aliases=PUBLISHABLE_ALIASES.get(base, ()))
            base_rows = row_scores(base_doc)
            if not base_rows:
                print(f"{ours:18s} {cell:22s} {'pending':>4s}")
                continue
            n, d, lo, hi, up, dn, row_lo, row_hi, k, per_ds = boot_ci_paired(ours_rows, base_rows)
            # A null has to report what it still permits, or it reads as reassurance. This is
            # the most negative difference the interval does not exclude: if it is large
            # relative to the spread of the table, "indistinguishable" means "this slice cannot
            # resolve the comparison", NOT "equal". Borrowed from the MemEval line, where a
            # -0.080 null permitted a 22pp deficit against a table spanning only 28pp.
            worst = min(lo, 0.0)
            verdict = ("ours better" if lo > 0 else
                       "ours worse" if hi < 0 else
                       f"unresolved (<={abs(worst):.1f} deficit allowed)")
            # Flag every comparison whose verdict the row-level interval would have changed, so
            # the inflation is visible in the table and not only in the docstring.
            row_verdict = ("ours better" if row_lo > 0 else
                           "ours worse" if row_hi < 0 else "unresolved")
            if row_verdict != verdict.split(" (")[0]:
                verdict += f"  [row-level would say '{row_verdict}': WRONG UNIT]"
            pos = sum(1 for v in per_ds.values() if v > 0)
            # An "ours better" verdict against gla is REAL as a number but MISATTRIBUTED as a
            # result: the official RWKV-7 2.9B baseline earns the same margin from the same EOS
            # fix (see gla_warning). Labelling it "ours better" with no qualifier is the one
            # place this table can overclaim, so the attribution is carried in the label itself
            # rather than only in a WARNING printed 40 lines later.
            if lo > 0 and "gla" in base_src:
                verdict += f"  [BUT NOT OURS: rwkv7-world3 gets {control_vs_gla():+.2f} from the same EOS fix]"
            print(f"{ours:18s} {base_src.split('/')[-1]:22s} {n:4d} {d:+7.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>18s} "
                  f"{f'{up}/{dn}':>10s} {worst:+7.1f}  {verdict}")
            if lo <= 0 <= hi and pos:
                print(f"{'':41s} datasets positive {pos}/{k}, per-dataset range "
                      f"{min(per_ds.values()):+.2f}..{max(per_ds.values()):+.2f}")
    print("\nn       = rows scored by BOTH models (paired). A short n means one cell is partial.")
    print("CI      = cluster-t over the 9 DATASETS, NOT a bootstrap over the 450 rows. Rows in one")
    print("          dataset share a document pool, prompt template and generation cap, so they")
    print("          are not independent draws. This matters and it mattered in OUR favour: the")
    print("          row-level version manufactured 'ours better' verdicts vs gla-1.3b that the")
    print("          clustered unit did not support, while every 'ours worse' verdict survived")
    print("          both units -- a one-directional error. Rows marked [WRONG UNIT] are where")
    print("          the two disagree.")
    print("up/down = rows where ours scored strictly higher / lower. This answers 'wins more")
    print("          often', which is NOT the same question as the interval ('wins by more on")
    print("          average'); the two can disagree, so quote whichever the claim needs.")
    print("worst   = the largest deficit the interval does NOT exclude. Compare it to the spread")
    print("          of the AVG column: if it is of that order, the comparison is unresolved")
    print("          rather than a tie, and must not be reported as parity.")
    print("NOTE    : baselines marked as truncation-bound floors (phi-2, gemma-2-2b) or as a")
    print("          real length collapse (mamba-2.8b) are not capability measurements, so")
    print("          beating them is not evidence of long-context skill. Compare against")
    print("          qwen2.5-3b / llama-3.2-3b-bos, which saw the full prompt.")
    gla_warning()
    spec_curve()
    print("WARNING : the summarization third (gov_report/qmsum/multi_news) does NOT clear a")
    print("          trivial floor on 2 of its 3 datasets for the four cells named below, ours")
    print("          included. Measured with this")
    print("          benchmark's own rouge==1.0.1 ROUGE-L F1 (rescore reproduces the published")
    print("          means exactly), paired per-row bootstrap vs LEAD-128 = 'copy the first 128")
    print("          words of the source document, no model' (19.63):")
    print("            global-dit-50k  -2.82 [-4.84,-0.84]   champ-dynamic-30k -3.74 [-5.50,-2.09]")
    print("            rwkv7-loop      -3.14 [-5.15,-1.16]   gla-1.3b-bos-loop -10.94 [-13.02,-8.91]")
    print("          !! THOSE FOUR POOLED INTERVALS ARE NOT A VALID TEST and are kept only")
    print("          because they are the published numbers. They resample 150 ROWS; pooling")
    print("          across datasets makes the DATASET the unit, and at k=3 (t=4.30) cluster-t")
    print("          has so little power that EVERY cell becomes a tie -- including all five")
    print("          positional baselines. So no POOLED SUM3 verdict is supportable in either")
    print("          direction. The informative test is per dataset (n=50 within a dataset,")
    print("          where rows really are different documents). Decomposed, the sign is NOT")
    print("          stable for exactly the three cells the headline rested on:")
    print("                                  gov_report        qmsum         multi_news")
    print("            global-dit (OURS)    -9.41 BELOW    +6.42 ABOVE     -5.49 BELOW")
    print("            champ      (OURS)    -8.63 BELOW    +3.06 ABOVE     -5.64 BELOW")
    print("            rwkv7-loop           -9.75 BELOW    +5.54 ABOVE     -5.19 BELOW")
    print("            gla-1.3b            -14.61 BELOW    -2.52 tie      -15.69 BELOW")
    print("            gemma-2-2b (noBOS)  -11.78 BELOW    -1.88 tie       -7.69 BELOW")
    print("          So the defensible claim is '2 of 3 summarization datasets do not clear the")
    print("          floor', NOT 'SUM3 does not clear the floor'. On qmsum our arms and rwkv7")
    print("          genuinely beat copy-the-first-128-words; gla and gemma-noBOS never do.")
    print("          !! TWO OF THE FOUR ALSO WEAKEN once a decode defect is removed. The")
    print("          rwkv7 and global-dit cells were generated with the WRONG eos id (rwkv7's")
    print("          tokenizer.eos_token_id=65530='\\n\\n' was the ONLY stop id passed; the real")
    print("          end-of-text")
    print("          is id 0 from generation_config), so 141/450 and 79/450 rows carry post-EOS")
    print("          junk that costs ROUGE-F1 precision. Truncating at id 0: rwkv7 -2.04")
    print("          [-4.14,+0.01] and global-dit -1.72 [-3.74,+0.25] (same invalid pooled unit);")
    print("          champ -3.74 and gla -10.94 are unaffected (0 rows with id 0).")
    print("          The headline survives in its narrowed form -- on 2 of 3 datasets our arms")
    print("          lose to 'copy the first 128 words' -- but do not claim all four are below")
    print("          the floor, and do not quote the pooled interval as the evidence.")
    print("            Positional baselines: qwen2.5-1.5b beats the floor on 3/3 datasets, but")
    print("          THIS IS ALSO OVERSTATED POOLED -- qwen3b/llama/phi-2 beat it on 2/3")
    print("          (multi_news is a tie) and gemma-2-2b-bos on only 1/3 (gov_report -0.84 and")
    print("          multi_news +1.87 are both ties). The correction cuts against my own")
    print("          'ALL 5 positional baselines BEAT it' phrasing too, not just against us.")
    print("          NOT an architecture split: same gemma weights and runner, BOS the only")
    print("          difference, land on OPPOSITE sides -- gemma-2-2b-bos +3.85 [+1.88,+5.72]")
    print("          BEATS the floor, gemma-2-2b (no BOS, never-EOS) -7.11 [-8.77,-5.54] is")
    print("          BELOW it, worse than either of our arms. The split tracks never-EOS /")
    print("          configuration, not attention-vs-recurrent. Do not label it architectural.")
    print("          Also: champ's margin over a shuffled-gold floor is n.s. (+0.88 [-0.32,")
    print("          +2.09]), and on gov_report BOTH our arms score BELOW shuffled gold. The")
    print("          floor is reachable by every model -- truncation keeps the head and the")
    print("          smallest retained head is 289-316 tokens vs ~170 for 128 words -- so this")
    print("          is not a window artefact. Do NOT quote SUM3 as partial summarization")
    print("          ability. See ledger 'The SUM3 column does not clear a trivial floor'.")
    print("WARNING : mamba-2.8b shows 'pending' above and must STAY out of the comparison, in")
    print("          both directions. Its 171 finished rows give ours a significant-looking win")
    print("          (global +2.91 [+1.19,+4.62]), but re-paired on the SAME rows at mamba's own")
    print("          best budget (1024) the sign flips to -1.45 [-4.70,+1.71]. mamba degrades")
    print("          monotonically with MORE input (8.18@1024 -> 2.49@8192), so any comparison")
    print("          drawn at our 32768 cap measures our cap, not mamba. That win was OUR")
    print("          artefact. See ledger 'mamba-2.8b: the one cell I could not finish'.")
    print("RESOLVED: the rwkv7 and global-dit cells above were DEFECTIVE until 2026-09-21 and are")
    print("          now the MEASURED, re-run, EOS-fixed cells. Both were generated with rwkv7's")
    print("          tokenizer.eos_token_id=65530 ('\\n\\n') as the ONLY stop id, so every row ran")
    print("          to the full budget (0/450 ending below their dataset's max_gen). State it")
    print("          that way, not as '65530 occurs in 0 rows': the loop breaks BEFORE appending,")
    print("          so a stop id that fired is invisible in the stored ids -- see header note.")
    print("          The true end-of-text for THESE two is id 0 (generation_config). Re-run and")
    print("          rescored, with a 450/450 control pass on each and distinct md5+elapsed")
    print("          proving genuine reruns rather than copies: rwkv7 9.55 -> 11.97 (141/450 rows")
    print("          now stop), global-dit 9.55 -> 11.98 (79/450). Both landed on the value the")
    print("          rescore had predicted, which is what validates the rescore method.")
    print("          On global-dit the fix is a PURE TRUNCATION: the new token ids are an exact")
    print("          PREFIX of the old on 450/450 rows, input_tokens identical on all 450, so no")
    print("          token changed value -- only where each sequence ends. gov_report went DOWN")
    print("          on rwkv7 (-0.72), the expected shape for a ROUGE-L F1 where a junk tail")
    print("          costs precision. champ 9.01 is unchanged as a FACT, not a mechanism: across")
    print("          all 86270 of its generated tokens it emits id 0 zero times and id 2 zero")
    print("          times, so unioning either into its stop set cannot touch any champ row.")
    print("          65530 is meanwhile champ's one WORKING stop id (32/450 rows stop on it), so")
    print("          it must be KEPT, not replaced. gla unaffected, and for a DIFFERENT reason:")
    print("          its eos=2 agrees across all three config files, so its stop condition was")
    print("          live and it simply never emits it.")
    print("          The defect was near-symmetric between the baseline and our arm (+2.42 vs")
    print("          +2.43), so global-dit vs rwkv7 stays a tie (+0.01 [-1.63,+1.66]). champ vs")
    print("          rwkv7 does not exclude 0 either (-2.96 [-6.00,+0.08]), but it is 7/9 negative")
    print("          and survives LOO only 2/9 -- report it as point + interval, not as a tie.")
    print("          ** I earlier published this as -3.69 [-6.66,-0.73] 'significantly worse'.")
    print("          RETRACTED as a significance claim: -3.69 is the excl-RWKV7 column, which")
    print("          drops only the BASELINE's degenerate rows, and asymmetric exclusion favours")
    print("          whoever is excluded against -- a rule I had already written for gla and then")
    print("          failed to apply to the comparison that hurts us. Symmetric exclusion gives")
    print("          -3.20 [-6.55,+0.15], which includes 0, and CI-exclusion is grid-dependent")
    print("          (see SPECIFICATION CURVE below; peer af5680b reaches the same conclusion on a")
    print("          disjoint 24-config grid). Neither a fraction of significant specs nor an")
    print("          'excludes 0 below nmax=N' threshold is reportable: the fraction spans 0%-100%")
    print("          with the grid's nmax weighting, and exclusion is NOT monotone in nmax. The SIGN")
    print("          is negative in EVERY spec of the curve below, so champ IS worse than the official")
    print("          baseline -- it")
    print("          is the SIGNIFICANCE, not the direction, that does not survive. **")
    print("          The genuine loss the fix exposes is champ vs global-dit: -2.97")
    print("          [-4.92,-1.03], 8/9 datasets negative, survives leave-one-dataset-out 9/9 --")
    print("          and BOTH of those are OUR arms, so it is an internal writer comparison")
    print("          (global beats dynamic), NOT a win over any baseline.")
    print("          gla's degenerate-row exclusion is 136/450 under the widened predicate, but")
    print("          the gla COMPARISON is")
    print("          NOT unaffected by the EOS fix: honouring global-dit's stop ids moved")
    print("          global-dit vs gla from unresolved to +4.49 [+0.78,+8.21], and it stays")
    print("          positive after excluding gla's degenerate rows (+2.78 [+0.67,+4.89]). That")
    print("          is the only clustered 'ours above a baseline' interval on this benchmark, so")
    print("          it is NOT dismissible on statistics -- but it is NOT OURS either: the same")
    print("          fix gives the official rwkv7-world3 2.9B baseline the same margin over gla")
    print(f"          ({control_vs_gla():+.2f} vs our +4.49), and our arm does not separate from that")
    print("          baseline (+0.01 [-1.63,+1.66]). So it measures the tokenizer fix, not the")
    print("          architecture. See the gla WARNING above for that control, for the size and")
    print("          degeneracy objections, and for the stopping-artefact explanation I had to")
    print("          retract.")
    print("          See ledger 'A wrong EOS id inflated the rwkv7 AND global-dit cells'.")
    print("CI UNIT : every interval in this WARNING block is a cluster-t over the 9 DATASETS, not")
    print("          a bootstrap over the 450 rows. Rows inside a LongBench dataset share a")
    print("          document pool, a prompt template and a generation cap, so they are not")
    print("          independent draws; resampling them understates the interval. Measured here:")
    print("          champ-rwkv7 row-level [-4.43,-1.59] 'significant' vs cluster-t [-6.00,+0.08]")
    print("          TIE -- 2.14x wider, and the verdict changes. Leave-one-dataset-out separates")
    print("          the two cases cleanly: champ-global-dit stays significant 9/9, champ-rwkv7")
    print("          only 2/9. Per-dataset deltas for champ-rwkv7 span -9.33 to +1.84, i.e. the")
    print("          SIGN is not stable across datasets; that is what the row-level CI hid.")


if __name__ == "__main__":
    main()

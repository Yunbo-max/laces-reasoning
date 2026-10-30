#!/usr/bin/env python3
"""Build the binding-capacity curve from the RULER expansion, and audit its generations.

Three jobs, in the order a reader should trust them:

1. **Table.** One row per (task, n_golds, length), columns = the four models, every cell at
   n=100. New cells come from `results/benchmarks/ruler_expand/`; the 9 previously published
   cells are re-derived at n=100 from the archived 200-row predictions in
   `results/benchmarks/longctx/` (their rows are a parquet prefix with ids 0..199, so
   `id < 100` is exactly the same slice the new runs evaluated). Re-derived cells are marked.

2. **Per-gold-rank hit rates.** For every multi-gold task, the fraction of rows in which
   gold *k* appeared in the prediction. This is the check that caught `cwe`: a flat profile
   plus a high rate on rank 0 means one repeated string is filling every slot, not ten
   retrieved items. `distinct_golds_hit` (how many of the golds were distinct strings the
   model actually produced) and `repetition_ratio` (most frequent token-n-gram share)
   separate real multi-retrieval from repetition luck.

3. **Generation dump.** Verbatim predictions for the multiquery/multivalue cells, because a
   nonzero recall score cannot be read as retrieval without looking at the text.

Writes ruler_expand_curve.json + the markdown table; reads nothing canonical, writes nothing
canonical.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
EXPAND = REPO / "results/benchmarks/ruler_expand"
CANON = REPO / "results/benchmarks/longctx"

N = 100
LENGTHS = ["4096", "8192", "16384"]
TASKS = ["niah_single_1", "niah_single_2", "niah_single_3",
         "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "qa_2", "fwe", "niah_multiquery", "niah_multivalue", "vt", "qa_1", "cwe"]
# Verified against the cached parquet, all three lengths (not assumed).
N_GOLDS = {"niah_single_1": 1, "niah_single_2": 1, "niah_single_3": 1,
           "niah_multikey_1": 1, "niah_multikey_2": 1, "niah_multikey_3": 1,
           "qa_2": 1, "fwe": 3, "qa_1": 3, "niah_multiquery": 4, "niah_multivalue": 4,
           "vt": 5, "cwe": 10}
# DISTINCT golds -- the number the binding law is actually about. It differs from N_GOLDS
# only for qa_1, whose gold list is alternative SPELLINGS of one answer
# (e.g. ['France','France','France','France']): 2.98 golds but only 1.54 distinct strings,
# measured over all 500 rows. qa_1 therefore sits at ~1 binding, not 3, and RULER scores it
# with string_match_part (max over golds) precisely because of this. Putting it on the curve
# at 3 would manufacture a data point that does not exist. Every other task's golds are
# distinct, so N_GOLDS == N_BINDINGS there.
N_BINDINGS = {**N_GOLDS, "qa_1": 1}
# Same-type distractor needles per row: needle lines present in the context MINUS the golds,
# counted from the cached parquet (mean over the first 30 rows; the generator makes this
# near-constant within a cell). This is the SECOND axis of the law, and it is the one that
# turns the 8 niah tasks into a real 2-factor design instead of a list of names:
#
#           bindings=1                                        bindings=4
#   0 distractors   single_1 (99% filler), single_2, single_3  multiquery, multivalue
#   3               multikey_1
#   50-900          multikey_2, multikey_3
#
# So single_* vs multikey_* varies distractor pressure at FIXED bindings=1, and
# single_2 vs multiquery/multivalue varies bindings at FIXED distractors=0. Only `None`
# where the task is not a needle task.
DISTRACTORS = {
    **{("4096", t): d for t, d in [("niah_single_1", 0), ("niah_single_2", 0),
                                   ("niah_single_3", 0), ("niah_multikey_1", 3),
                                   ("niah_multikey_2", 200), ("niah_multikey_3", 50),
                                   ("niah_multiquery", 0), ("niah_multivalue", 0)]},
    # Measured, all three lengths (median over 30 rows; min==max on every cell except
    # 16384/multikey_2 which is 899-900). The distractor count SCALES WITH LENGTH for
    # multikey_2/_3 and is CONSTANT for every other needle task -- that asymmetry is what
    # makes those two cells a length-confounded test of distractor pressure and the rest a
    # clean one.
    **{("8192", t): d for t, d in [("niah_single_1", 0), ("niah_single_2", 0),
                                   ("niah_single_3", 0), ("niah_multikey_1", 3),
                                   ("niah_multikey_2", 425), ("niah_multikey_3", 125),
                                   ("niah_multiquery", 0), ("niah_multivalue", 0)]},
    **{("16384", t): d for t, d in [("niah_single_1", 0), ("niah_single_2", 0),
                                    ("niah_single_3", 0), ("niah_multikey_1", 3),
                                    ("niah_multikey_2", 900), ("niah_multikey_3", 275),
                                    ("niah_multiquery", 0), ("niah_multivalue", 0)]},
}
# Haystack composition -- the THIRD variable, and the one the original 3-point curve had no
# way to see. Measured as the fraction of context lines that are not needle lines:
# niah_single_1's haystack is 99% repeated filler ("The grass is green. The sky is blue...")
# while niah_single_2/_3 and the multiquery/multivalue pair sit in real Paul Graham essay
# prose (~50% filler), and multikey_2/_3 have essentially NO filler -- their context is a
# solid wall of same-shaped needle lines. So `niah_single_1` differs from `niah_single_2` in
# haystack type at IDENTICAL bindings (1) and distractors (0), which makes that pair the
# clean test of whether the saturation at 1 item was about the item count at all.
HAYSTACK = {"niah_single_1": "repeat-filler", "niah_single_2": "essays",
            "niah_single_3": "essays", "niah_multikey_1": "essays",
            "niah_multikey_2": "needles-only", "niah_multikey_3": "needles-only",
            "niah_multiquery": "essays", "niah_multivalue": "essays"}
# uuid keys/values are much longer strings than the 7-digit numbers, so single_3 vs single_2
# and multikey_3 vs multikey_2 additionally vary the surface form.
KEYTYPE = {"niah_single_3": "uuid", "niah_multikey_3": "uuid"}
MODELS = ["champ-dynamic-30k", "global-dit-50k", "llama-3.2-3b", "qwen2.5-3b"]
# Which shard files belong to which model. The driver names shards by arm+length band.
SHARD_PREFIX = {"champ-dynamic-30k": "champ-", "global-dit-50k": "global-",
                "llama-3.2-3b": "llama-3.2-3b", "qwen2.5-3b": "qwen2.5-3b"}


def cell_key(dataset: str) -> tuple[str, str]:
    """'16384_niah_multikey_2' -> ('16384', 'niah_multikey_2')."""
    head, _, rest = dataset.partition("_")
    return (head, rest) if head.isdigit() else ("?", dataset)


def collect_rows() -> dict:
    """{model: {(length, task): [row, ...]}} restricted to the first N ids."""
    out: dict[str, dict[tuple[str, str], list]] = {m: {} for m in MODELS}
    for path in sorted(glob.glob(str(EXPAND / "*__ruler_expand.json"))):
        base = os.path.basename(path).replace("__ruler_expand.json", "")
        model = next((m for m, p in SHARD_PREFIX.items()
                      if base == p or (p.endswith("-") and base.startswith(p))), None)
        if model is None:
            print(f"  [warn] unmatched shard file {base}")
            continue
        for row in json.load(open(path)).get("rows", []):
            if row.get("suite") != "ruler" or row["id"] >= N:
                continue
            out[model].setdefault(cell_key(row["dataset"]), []).append(row)
    # Re-derive the 9 published cells at n=100 from the archived 200-row predictions.
    # Bucket the whole file FIRST, then merge only the cells with no fresh measurement. An
    # earlier version tested `key not in out[model]` per row, so the first row created the
    # key and every later row of the same cell was skipped -- every re-derived cell held
    # exactly 1 row and `fwe` read 33.3 (one row's recall) instead of 42.7.
    for model in MODELS:
        canon = CANON / f"{model}__ruler.json"
        if not canon.exists():
            continue
        archived: dict[tuple[str, str], list] = {}
        for row in json.load(open(canon)).get("rows", []):
            if row.get("suite") != "ruler" or row["id"] >= N:
                continue
            archived.setdefault(cell_key(row["dataset"]), []).append(
                {**row, "_rederived": True})
        for key, rows in archived.items():
            if key not in out[model]:          # never overwrite a fresh measurement
                out[model][key] = rows
    return out


def score_of(rows: list) -> float:
    return sum(float(r["score"]) for r in rows) / len(rows) if rows else float("nan")


# Answer-shaped tokens: a RULER gold is a uuid, a 6+ digit magic number, an all-caps 5-letter
# vt label, or a lowercase word (cwe/fwe). Used only for the volume-luck precision guard.
ANSWER_TOK = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
                        r"|\d{6,}|\b[a-z]{4,}\b")


def rank_profile(rows: list) -> dict:
    """Per-gold-rank hit rate + the repetition and volume-luck diagnostics.

    The rank-k rate is divided by the number of rows that actually HAVE a gold at rank k,
    not by the row count. Only qa_1 has a ragged gold list (2-4 golds; 476/500 rows have 3),
    and dividing by n there put rank 4 at 0.03 for every model -- an artefact of 8 rows, not
    a finding, and it made the spread look like 0.68 for llama.
    """
    if not rows:
        return {}
    k = max(len(r["answers"]) for r in rows)
    hits, seen = [0] * k, [0] * k
    distinct_hit, rep, emitted, prec = [], [], [], []
    for r in rows:
        pred = r["prediction"].lower()
        got = [g.lower() in pred for g in r["answers"]]
        for i, g in enumerate(got):
            hits[i] += int(g)
            seen[i] += 1
        distinct_hit.append(len({g.lower() for g, ok in zip(r["answers"], got) if ok}))
        # Volume-luck guard. string_match_all rewards a prediction that merely ENUMERATES
        # many candidates, so count the distinct answer-shaped tokens emitted and the share
        # of them that are golds. Surgical retrieval sits at emitted ~= n_golds and
        # precision ~= 1.0 (measured: llama 8192/multiquery 3.97 emitted, prec 1.00).
        # The numerator must be DISTINCT gold hits: qa_1's gold list repeats the same string
        # up to 4x (['France','France','France','France']), and summing `got` there pushed
        # precision above 1.0 (observed 1.123 for qwen 4096/qa_1).
        toks = set(ANSWER_TOK.findall(pred))
        emitted.append(len(toks))
        prec.append(distinct_hit[-1] / len(toks) if toks else 0.0)
        # Repetition ratio: share of the single most common word in the generation. cwe's
        # '1. forest 2. forest ...' sits near 0.37 here while a real 10-item list sits at
        # 0.06 (measured: champ 4096/cwe 0.368 vs llama 0.061).
        words = re.findall(r"[a-z0-9']+", pred)
        rep.append(max(collections.Counter(words).values()) / len(words) if words else 0.0)
    n = len(rows)
    return {"n": n,
            "per_rank_hit_rate": [round(h / s, 4) if s else None
                                  for h, s in zip(hits, seen)],
            "per_rank_n": seen,
            "mean_distinct_golds_hit": round(sum(distinct_hit) / n, 3),
            "max_possible_distinct": k,
            "mean_repetition_ratio": round(sum(rep) / n, 4),
            "mean_distinct_tokens_emitted": round(sum(emitted) / n, 3),
            "mean_gold_precision": round(sum(prec) / n, 4)}


# Needle lines have a fixed shape in every niah task, e.g.
#   "One of the special magic numbers for green-bustle is: 1781180."
# so the full set of in-context needle VALUES is recoverable from the context, and a wrong
# answer can be classified instead of merely counted.
NEEDLE_RE = re.compile(r"magic (?:number|uuid|word)s? for [^\n]*? is:?\s*"
                       r"([0-9a-f-]{6,}|\d+)", re.I)


def failure_modes(rows: list, contexts: dict) -> dict:
    """Split non-hits into 'retrieved the wrong needle' vs 'retrieved nothing'.

    This is the diagnostic that separates the law's two candidate mechanisms, and score
    alone cannot express it. A model that answers with a *different needle from the same
    context* has held something and failed to bind it to the right key -- a discrimination
    failure. A model that answers with no needle-shaped token at all never retrieved
    anything -- a readout/capacity failure. Measured on the published 4096/niah_multikey_1
    cell these are wildly different: global-dit emits a real-but-wrong needle on 42% of rows
    (gold 25.5%), while champ emits a non-needle number on 45% and nothing on 39%.
    """
    if not rows or not contexts:
        return {}
    cat = collections.Counter()
    for r in rows:
        ctx = contexts.get(r["id"])
        if ctx is None:
            continue
        golds = {g.lower() for g in r["answers"]}
        in_ctx = {v.lower() for v in NEEDLE_RE.findall(ctx)} - golds
        pred = r["prediction"].lower()
        if any(g in pred for g in golds):
            cat["gold_hit"] += 1
        elif any(v in pred for v in in_ctx):
            cat["wrong_needle_from_context"] += 1
        elif re.search(r"\d{5,}|[0-9a-f]{8}-[0-9a-f]{4}", pred):
            cat["fabricated_needle_shape"] += 1
        else:
            cat["no_needle_emitted"] += 1
    total = sum(cat.values())
    if not total:
        return {}
    return {"n_classified": total,
            **{k: round(v / total, 4) for k, v in sorted(cat.items())}}


def load_contexts(length: str, task: str) -> dict:
    """{row_id: context} from the cached parquet, for failure-mode classification."""
    import pandas as pd
    path = (Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/"
                 "datasets--simonjegou--ruler/snapshots/"
                 "24adceac8a0e6532936e8d721cd9e9084d2e4686") / length
            / "test-00000-of-00001.parquet")
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    df = df[df["task"] == task].reset_index(drop=True)
    return {i: c for i, c in enumerate(df["context"].tolist()[:N])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_generations", default="niah_multiquery,niah_multivalue",
                    help="comma-separated tasks whose raw predictions to print")
    ap.add_argument("--dump_n", type=int, default=6)
    args = ap.parse_args()

    data = collect_rows()
    print("=== cells found (n=%d) ===" % N)
    for m in MODELS:
        got = len(data[m])
        red = sum(1 for rs in data[m].values() if rs and rs[0].get("_rederived"))
        print(f"  {m:20s} {got:2d}/39 cells  ({red} re-derived at n=100 from archived n=200)")

    # ---------------- table ----------------
    lines = ["| task | golds | bind | distr | haystack | len | champ-dyn | global-dit "
             "| llama-3.2-3b | qwen2.5-3b |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for task in TASKS:
        for length in LENGTHS:
            cells = []
            for m in MODELS:
                rows = data[m].get((length, task), [])
                s = score_of(rows)
                red = bool(rows and rows[0].get("_rederived"))
                table[f"{length}/{task}/{m}"] = {
                    "score": None if rows == [] else round(s, 4), "n": len(rows),
                    "n_golds": N_GOLDS[task], "n_bindings": N_BINDINGS[task],
                    "same_type_distractors": DISTRACTORS.get((length, task)),
                    "haystack": HAYSTACK.get(task), "keytype": KEYTYPE.get(task, "word"),
                    "rederived_from_n200": red}
                cells.append("--" if not rows else
                             f"{100*s:.1f}{'*' if red else ''}")
            d = DISTRACTORS.get((length, task))
            lines.append(f"| {task} | {N_GOLDS[task]} | {N_BINDINGS[task]} | "
                         f"{'' if d is None else d} | {HAYSTACK.get(task, '')} | {length} | "
                         + " | ".join(cells) + " |")
    table_md = "\n".join(lines)
    print("\n=== binding-capacity curve (score x100, n=%d; * = re-derived) ===" % N)
    print(table_md)

    # ---------------- per-gold-rank profiles ----------------
    print("\n=== per-gold-rank hit rates (multi-gold tasks) ===")
    profiles = {}
    for task in [t for t in TASKS if N_GOLDS[t] > 1]:
        for length in LENGTHS:
            for m in MODELS:
                rows = data[m].get((length, task), [])
                if not rows:
                    continue
                p = rank_profile(rows)
                profiles[f"{length}/{task}/{m}"] = p
                # Ranks backed by fewer than 10 rows are excluded from the spread: qa_1's
                # rank 4 exists on ~2 of 100 rows and would otherwise dominate it.
                flat = [v for v, s in zip(p["per_rank_hit_rate"], p["per_rank_n"])
                        if v is not None and s >= 10]
                spread = max(flat) - min(flat) if flat else 0.0
                print(f"  {length:>5}/{task:<16} {m:<20} score={100*score_of(rows):5.1f} "
                      f"ranks={p['per_rank_hit_rate']} spread={spread:.3f} "
                      f"distinct={p['mean_distinct_golds_hit']}/{p['max_possible_distinct']} "
                      f"rep={p['mean_repetition_ratio']:.3f} "
                      f"emit={p['mean_distinct_tokens_emitted']:.2f} "
                      f"prec={p['mean_gold_precision']:.3f}")

    # ---------------- failure modes on the niah family ----------------
    print("\n=== failure modes: wrong needle (discrimination) vs none (readout) ===")
    fmodes = {}
    niah = [t for t in TASKS if t.startswith("niah_")]
    for task in niah:
        for length in LENGTHS:
            ctx = None
            for m in MODELS:
                rows = data[m].get((length, task), [])
                if not rows:
                    continue
                if ctx is None:
                    ctx = load_contexts(length, task)
                f = failure_modes(rows, ctx)
                if not f:
                    continue
                fmodes[f"{length}/{task}/{m}"] = f
                print(f"  {length:>5}/{task:<16} {m:<20} "
                      f"gold={f.get('gold_hit', 0):.3f} "
                      f"wrong_needle={f.get('wrong_needle_from_context', 0):.3f} "
                      f"fabricated={f.get('fabricated_needle_shape', 0):.3f} "
                      f"none={f.get('no_needle_emitted', 0):.3f}")

    # ---------------- generations ----------------
    print("\n=== verbatim generations (the cwe lesson: never trust a score alone) ===")
    for task in [t.strip() for t in args.dump_generations.split(",") if t.strip()]:
        for length in LENGTHS:
            for m in MODELS:
                rows = data[m].get((length, task), [])
                if not rows:
                    continue
                print(f"\n--- {length}/{task} :: {m} (score={100*score_of(rows):.1f}) ---")
                for r in rows[:args.dump_n]:
                    pred = r["prediction"].replace("\n", " | ")[:300]
                    print(f"   golds={r['answers']}")
                    print(f"   score={r['score']:.2f} pred={pred!r}")

    out = EXPAND / "ruler_expand_curve.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"n_per_cell": N, "n_golds": N_GOLDS, "models": MODELS,
         "cells": table, "rank_profiles": profiles, "failure_modes": fmodes,
         "table_markdown": table_md},
        indent=2) + "\n")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()

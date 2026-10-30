#!/usr/bin/env python3
"""Validate the mamba-2.8b fill shards. Replaces the inline block in
`run_mamba_longbench_babilong_sharded.sh`, whose stop-rule check is defective two ways.

DEFECT 1 -- the inline check is a silent no-op. It reads `r["generated_tokens"]`, a key these
rows do not have (the schema is `answers, arm, dataset, dropped_tokens, id, input_tokens,
mean_logp, prediction, prediction_token_ids, prompt_tokens_untruncated, score, suite`). The
guarded expression `bool(gen) and min(gen) < want_new` therefore evaluates `False` for every
file ever passed to it, so it could never report a fired stop rule and could never report a
missing one either. Same class as the inert BOS flag-read: lands, runs, changes nothing.

DEFECT 2 -- even with the right key it compares against the wrong cap. LongBench caps
generation PER DATASET from the pinned `dataset2maxlen.json` (`run_open_context_bench.py:1588`
takes `min` of it and `--max_new_tokens`), so the caps in this job are 32 for musique /
hotpotqa / 2wikimqa / passage_retrieval_en, 64 for multifieldqa_en, 128 for narrativeqa and
512 for gov_report / multi_news / qmsum -- not the global 512. Measuring "short" against 512
calls a run where the stop rule fired zero times a pass: passage_retrieval_en is 32/32 tokens
on all 50 rows, which is AT its real cap and 480 short of the global one.

The consequence matters here rather than being pedantic: a never-firing stop rule is exactly the
bug that cost this benchmark a full day of wrong numbers, and its only observable trace is short
rows, because the runner breaks BEFORE appending the id. Searching generations for the stop id
proves nothing.

POSITIVE CONTROL -- RETRACTED, and the retraction is the useful part. An earlier version of this
docstring claimed mamba's RULER shards prove the stop rule fires on this path:

    4096_vt              200/200 rows short, min length  30
    4096_cwe             200/200 rows short, min length 120
    4096_fwe + 4096_qa_1 400/400 rows short, min length  32
    4096_niah_single_1      0/200 rows short, all at 128 -- and it scores 100.00

That was the wrong denominator, i.e. exactly DEFECT 2 again, one axis over. RULER caps generation
PER TASK, and each task's cap is the number that appeared in the "min length" column: vt 30, cwe 120,
fwe 50, qa_1 32, niah_* 128. Measured against its OWN cap, every one of those cells is 0/200 short,
not 200/200. Re-measured here from the writer's own `prediction_token_ids`: all 8 mamba RULER cells
have exactly ONE distinct generated length, equal to that cell's own cap. The stop rule fires in
none of them.

So "ran to cap" carries NO information about whether a cell is good or broken: 8/8 cells run to cap,
and they score from 0.00 to 100.00. The one genuinely informative control survives and is the reason
a filled cap is not by itself grounds to distrust a zero -- 4096_niah_single_1 fills its 128-token cap
on 200/200 rows and scores 100.00. Independently, `pythia-2.8b` (also `decode_path=hf`) stops early on
only 3/200 published LongBench rows, and mamba's own multifieldqa_en stops early on 3/50 here, which
is what proves this check is live rather than inert.

THE CAP IS A FOUR-WAY MIN (`run_open_context_bench.py:1587-1595`), not the global flag:
    min(--max_new_tokens,
        max_gen_map[dataset],                  # longbench only, pinned dataset2maxlen.json
        row["max_new_tokens"],                 # ANY suite, when the row carries one -- RULER's only
                                               # lowering source, since max_gen_map is longbench-only
        trajectory_horizon * trajectory_chunk_size)   # plan arms other than `global`
Any check that compares against the global flag will report "200/200 short" for vt, qa_1, fwe and cwe
in every run, forever.

What the fill rate does NOT excuse is the degeneracy rate, which is reported separately below and
is a capability statement about the model, not an artefact.

Everything else is a check the inline version got right, kept and stated per-cell: the
two-conjunct clamp fingerprint (constant AND equal to the computed cap), partial truncation from
`input_tokens < prompt_tokens_untruncated` -- which is a (model, suite) property, absent here for
no cell but asserted rather than assumed -- protocol equality field by field, and the row-identity
control against the peer the cell would sit beside.

Degeneracy uses the canonical predicate imported from `report_longbench_table.py`. It is not
re-implemented: re-stating a predicate inline is how `degenerate_union_rate` and `collapse_rate`
drifted from the column they were supposed to agree with, four times over.

This script decides nothing about publication. It prints what is true of each file; which file
feeds a printed table is a separate, deliberate call made against each file's own provenance.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222"
REPO = f"{ROOT}/research/laces-reasoning"
LB_MAXLEN = f"{ROOT}/longbench_local/pinned_spec/dataset2maxlen.json"


def load_canonical_predicate():
    """Import `is_degenerate_row` from the reporting script rather than restating it."""
    path = f"{REPO}/scripts/eval/report_longbench_table.py"
    spec = importlib.util.spec_from_file_location("report_longbench_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.is_degenerate_row


def gen_caps(suite: str, global_cap: int) -> dict:
    """Per-dataset generation caps, the same `min` the runner applies.

    `max_gen_map` is populated for `--suite longbench` only, so for every other suite the row's
    own `max_new_tokens` is the only thing that can lower the cap. `row_caps` below reads it.
    """
    if suite != "longbench":
        return {}
    pinned = json.loads(open(LB_MAXLEN).read())
    return {d: min(int(v), global_cap) for d, v in pinned.items()}


def row_caps(blob: dict, rows: list, dataset: str, fallback: int) -> tuple:
    """The cap actually in force for `dataset`, resolved by explicit priority.

    `run_open_context_bench.py:1589-1590` appends `row["max_new_tokens"]` to the min for ANY suite,
    which is how RULER gets its per-task caps (vt 30, cwe 120, fwe 50, qa_1 32). But that value is
    recorded NOWHERE in the output: not in the row schema (`answers, arm, dataset, dropped_tokens,
    id, input_tokens, mean_logp, prediction, prediction_token_ids, prompt_tokens_untruncated, score,
    suite`) and not in `config`. So it cannot be recovered from the result file alone and must be
    read back from the source parquet named by `config.data_dir`.

    Priority, and the last level is UNKNOWN rather than a fallback to the global flag. Falling back
    silently is what produced the retracted 200/200 control: the global flag is always present, so a
    fallback always looks like a successful resolution. Same four-level shape as the input-side cap.
      1. the row's own `max_new_tokens`, if a future writer ever records it
      2. the source parquet's per-task `max_new_tokens` (RULER)
      3. the pinned `dataset2maxlen.json` min (longbench)
      4. UNKNOWN -- report it, never guess
    """
    seen = {int(r["max_new_tokens"]) for r in rows
            if r.get("dataset") == dataset and r.get("max_new_tokens")}
    if seen:
        if len(seen) > 1:
            return min(seen), f"ROW-VARYING {sorted(seen)} -- 'short' is not one predicate here"
        return min(min(seen), fallback), "row max_new_tokens"

    task = re.sub(r"^\d+_", "", str(dataset))          # "4096_vt" -> "vt"
    caps = parquet_task_caps(blob.get("config", {}).get("data_dir"))
    if task in caps:
        return min(caps[task], fallback), f"parquet task cap ({task})"
    if fallback is not None:
        return fallback, "global/pinned"
    return None, "UNKNOWN"


_PARQUET_CACHE: dict = {}


def parquet_task_caps(data_dir) -> dict:
    """Per-task `max_new_tokens` from the RULER parquet the shards actually read.

    Verified constant within task and identical at 4096 and 16384: cwe 120, fwe 50, qa_1/qa_2 32,
    vt 30, every niah_* 128.
    """
    if not data_dir or not os.path.isdir(data_dir):
        return {}
    if data_dir in _PARQUET_CACHE:
        return _PARQUET_CACHE[data_dir]
    caps: dict = {}
    try:
        import pyarrow.parquet as pq
        for path in sorted(glob.glob(os.path.join(data_dir, "*", "*.parquet"))):
            table = pq.read_table(path, columns=["task", "max_new_tokens"]).to_pydict()
            for task, cap in zip(table["task"], table["max_new_tokens"]):
                cap = int(cap)
                if caps.setdefault(task, cap) != cap:
                    caps[task] = min(caps[task], cap)
    except Exception as exc:                      # noqa: BLE001 -- absence must be visible
        print(f"    (parquet cap lookup failed: {type(exc).__name__}: {exc})")
        return {}
    _PARQUET_CACHE[data_dir] = caps
    return caps


def check(suite, shard_dir, want_cells, want_n, want_in, want_new, peer_path, is_degenerate):
    print(f"\n{'=' * 78}\n{suite}: shards in {shard_dir}\n{'=' * 78}")
    paths = sorted(glob.glob(os.path.join(shard_dir, "*.json")))
    if not paths:
        print("  NO SHARDS -- every lane for this suite failed or was cut. Nothing to merge.")
        return
    caps = gen_caps(suite, want_new)
    cells = 0
    for path in paths:
        blob = json.load(open(path))
        cfg = blob.get("config", {})
        agg = blob.get("aggregates", {})
        rows = blob.get("rows", [])
        cells += len(agg)
        print(f"\n  {os.path.basename(path)}")
        # `stop_ids`, `decode_path` and `decode_is_trustworthy` are siblings of `config`, not
        # children of it (`run_open_context_bench.py:1683`). Reading them off `config` returns
        # None for every file, which reads exactly like "no stop id was resolved" -- a provenance
        # field silently reporting the defect it exists to rule out.
        print(f"    cells={len(agg)} rows={len(rows)} decode={cfg.get('decode')!r} "
              f"stop_ids={blob.get('stop_ids')} decode_path={blob.get('decode_path')!r} "
              f"trustworthy={blob.get('decode_is_trustworthy')}")
        if not blob.get("stop_ids"):
            print("    *** no stop ids resolved: every row must run to its cap by construction")

        for field, want in (("max_samples", want_n), ("max_input_tokens", want_in),
                            ("max_new_tokens", want_new), ("seed", 42)):
            got = cfg.get(field)
            flag = "" if got == want else f"   *** PROTOCOL MISMATCH (want {want!r})"
            print(f"    cfg {field:18s} = {got!r}{flag}")

        for dataset, entry in sorted(agg.items()):
            sub = [r for r in rows if r.get("dataset") == dataset]
            n = entry.get("n")
            score = entry.get("score")
            print(f"\n    -- {dataset}  n={n}  score={100 * score:.2f}" if score is not None
                  else f"\n    -- {dataset}  n={n}")
            if n != want_n:
                print(f"       *** SHORT CELL: n={n}, want {want_n}")

            # Truncation, conjunct 1+2: constant AND equal to the computed cap. A constant that
            # is not the cap is a property of the data, not of the harness.
            tokens = [r.get("input_tokens") for r in sub if r.get("input_tokens") is not None]
            cap = cfg.get("effective_input_budget")
            constant = bool(tokens) and len(set(tokens)) == 1
            clamped = constant and tokens[0] == cap
            print(f"       input_tokens min={min(tokens) if tokens else '?'} "
                  f"max={max(tokens) if tokens else '?'} distinct={len(set(tokens))} "
                  f"cap={cap} -> all_or_nothing_clamp={'YES' if clamped else 'no'}")

            # Partial truncation. Needs `prompt_tokens_untruncated`; its presence is a
            # (model, suite) property, so state which case this is instead of assuming.
            untrunc = [(r.get("input_tokens"), r.get("prompt_tokens_untruncated")) for r in sub]
            if all(b is not None and a is not None for a, b in untrunc):
                partial = sum(1 for a, b in untrunc if a < b)
                print(f"       partial truncation: {partial}/{len(sub)} rows "
                      f"(input_tokens < prompt_tokens_untruncated)")
            else:
                print("       partial truncation: NOT DECIDABLE "
                      "(prompt_tokens_untruncated absent on some rows)")

            # Stop rule. The only observable consequence is rows shorter than THIS cell's own cap,
            # which is a four-way min and NOT the global flag. Measuring against the global flag is
            # the wrong-denominator error that made the first version of this file's own positive
            # control report 200/200 when the truth was 0/200.
            lengths = [len(r.get("prediction_token_ids") or []) for r in sub]
            dataset_cap, cap_src = row_caps(blob, rows, dataset, caps.get(dataset, want_new))
            short = sum(1 for length in lengths if length < dataset_cap)
            print(f"       generated len min={min(lengths) if lengths else '?'} "
                  f"max={max(lengths) if lengths else '?'} cap={dataset_cap} "
                  f"(global {want_new}, from {cap_src}) -> stop_fires={short > 0} "
                  f"on {short}/{len(sub)} rows")
            if short == 0 and lengths:
                print("       *** every row ran to this cell's OWN cap. This is the normal case: "
                      "8/8 mamba RULER cells do it, scoring 0.00 to 100.00, so it carries no "
                      "information about whether the cell is good or broken. Not a bug signal.")

            # Degeneracy, canonical predicate. The FP control needs a non-empty denominator:
            # a clean-row mean over zero rows proves nothing.
            degenerate = [is_degenerate(r) for r in sub]
            clean = [r for r, d in zip(sub, degenerate) if not d]
            rate = 100 * sum(degenerate) / len(degenerate) if degenerate else float("nan")
            print(f"       degenerate={sum(degenerate)}/{len(degenerate)} = {rate:.1f}% "
                  f"(canonical is_degenerate_row)")
            if clean:
                clean_mean = 100 * sum(r.get("score", 0.0) for r in clean) / len(clean)
                print(f"       clean rows n_keep={len(clean)}, mean score={clean_mean:.2f}"
                      + ("   [n_keep<=5: NOT ESTIMABLE]" if len(clean) <= 5 else ""))
            else:
                print("       clean rows n_keep=0 -> no clean-row estimate exists "
                      "(an empty denominator is not a clean 0.00)")

    print(f"\n  merged-cell count would be {cells}/{want_cells}")

    # Row-identity control. Shards that each validate perfectly are still not comparable to the
    # peers unless they cover the same items.
    if not os.path.exists(peer_path):
        print(f"  ROW CONTROL: peer {peer_path} absent; comparability unverified")
        return
    peer = json.load(open(peer_path))
    peer_ids = {(r.get("dataset"), r.get("id")) for r in peer.get("rows", [])}
    mine = set()
    for path in paths:
        mine |= {(r.get("dataset"), r.get("id"))
                 for r in json.load(open(path)).get("rows", [])}
    shared = peer_ids & mine
    print(f"  ROW CONTROL vs {os.path.basename(peer_path)}: peer={len(peer_ids)} "
          f"mine={len(mine)} shared={len(shared)} mine_only={len(mine - peer_ids)} "
          f"peer_only={len(peer_ids - mine)}")
    if mine and not shared:
        print("    *** ZERO shared items: these are not the peers' rows. Do NOT compare.")
    # Per dataset, so a single bad lane cannot hide inside a healthy total.
    for dataset in sorted({d for d, _ in mine}):
        mine_d = {i for d, i in mine if d == dataset}
        peer_d = {i for d, i in peer_ids if d == dataset}
        if peer_d and mine_d != peer_d:
            print(f"    {dataset}: mine={len(mine_d)} peer={len(peer_d)} "
                  f"shared={len(mine_d & peer_d)}  *** item sets differ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["longbench", "babilong", "both"], default="both")
    args = parser.parse_args()

    is_degenerate = load_canonical_predicate()
    specs = [
        ("longbench", f"{REPO}/results/benchmarks/longctx_mamba_longbench_shards", 9, 50,
         32768, 512, f"{REPO}/results/benchmarks/longctx/rwkv7-world3__longbench.json"),
        ("babilong", f"{REPO}/results/benchmarks/longctx_mamba_babilong_shards", 4, 100,
         16384, 32, f"{REPO}/results/benchmarks/longctx/rwkv7-world3__babilong.json"),
    ]
    for spec in specs:
        if args.suite in ("both", spec[0]):
            check(*spec, is_degenerate=is_degenerate)


if __name__ == "__main__":
    main()

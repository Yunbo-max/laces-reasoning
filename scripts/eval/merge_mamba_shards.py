#!/usr/bin/env python3
"""Stitch the per-dataset mamba-2.8b shards into one result file.

mamba-2.8b had to be run one dataset per GPU (see
``scripts/eval/run_infinitebench_mamba_sharded.sh``): with no mamba fast kernels on this image
it decodes at ~2.9 min/row, so 250 rows serially exceeds the 12 h job cap -- and the runner
writes its JSON only once, at the end, so an overrunning serial run yields nothing at all.

The shards are independent runs of the same protocol, so merging is a concatenation of ``rows``
plus a union of ``aggregates``. ``config`` is taken from the first shard and checked against the
others on the fields that must not differ; a mismatch is fatal rather than silently averaged,
because a merged row that was produced under a different budget is not comparable.

Usage:
  python3 scripts/eval/merge_mamba_shards.py \
      [--shard_dir results/benchmarks/longctx_mamba_shards] \
      [--output results/benchmarks/longctx_fix/mamba-2.8b__infinitebench.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# Fields that describe the measurement itself: if two shards disagree here they were not
# running the same experiment and must not be pooled.
MUST_MATCH = ("max_input_tokens", "max_new_tokens", "effective_input_budget", "seed",
              "truncation_reason", "max_samples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_dir", default="results/benchmarks/longctx_mamba_shards")
    parser.add_argument("--glob", default="mamba-2.8b__*__infinitebench.json",
                        help="Shard filename pattern inside --shard_dir. RULER is also sharded "
                             "(one task per GPU), so the suite is not hardcoded.")
    parser.add_argument("--output",
                        default="results/benchmarks/longctx_fix/mamba-2.8b__infinitebench.json")
    parser.add_argument("--expect_cells", type=int, default=None,
                        help="Refuse to write unless exactly this many dataset cells merged. The "
                             "driver runs this merge after `wait`, so if one lane died the merge "
                             "would otherwise happily emit a SHORT file at the publish path, and a "
                             "partial row published as a matched row is the exact error this whole "
                             "ledger exists to prevent. RULER matched protocol = 9 cells.")
    parser.add_argument("--expect_rows_per_cell", type=int, default=None,
                        help="Refuse to write unless every cell has exactly this many rows "
                             "(RULER: 200). Guards the `rows[:max_samples]` silent-shortfall case.")
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(args.shard_dir, args.glob)))
    if not shards:
        raise SystemExit(f"no mamba shards matching {args.glob!r} in {args.shard_dir}")

    merged: dict = {}
    rows: list[dict] = []
    aggregates: dict = {}
    for path in shards:
        blob = json.load(open(path))
        if not merged:
            merged = {k: v for k, v in blob.items() if k not in ("rows", "aggregates")}
        else:
            base, other = merged.get("config", {}), blob.get("config", {})
            for field in MUST_MATCH:
                if field in base and field in other and base[field] != other[field]:
                    raise SystemExit(
                        f"shard {os.path.basename(path)} disagrees on {field!r}: "
                        f"{other[field]!r} vs {base[field]!r}. Shards produced under different "
                        f"settings must not be pooled."
                    )
        rows.extend(blob.get("rows", []))
        for dataset, entry in blob.get("aggregates", {}).items():
            if dataset in aggregates:
                raise SystemExit(f"dataset {dataset!r} appears in two shards; refusing to merge")
            aggregates[dataset] = entry

    merged["rows"] = rows
    merged["aggregates"] = aggregates
    merged.setdefault("config", {})["merged_from_shards"] = [os.path.basename(p) for p in shards]
    # State plainly that this row was produced by a sharded run, so a reader of the JSON alone
    # does not have to reconstruct it from the ledger.
    merged["config"]["sharding_note"] = (
        "Run one dataset/task per GPU because this image has no mamba fast kernels, so mamba "
        "decodes through a Python sequential scan (measured: ~2.9 min/row at 32768 tokens on "
        "InfiniteBench, ~23.5 s/row at 4096 on RULER); protocol is otherwise identical to the "
        "other models in the table."
    )

    # Completeness gates, checked BEFORE anything is written: a short file at the publish path is
    # worse than no file, because it looks like a finished matched row.
    if args.expect_cells is not None and len(aggregates) != args.expect_cells:
        raise SystemExit(
            f"REFUSING TO WRITE: merged {len(aggregates)} cells, expected {args.expect_cells}. "
            f"Got: {', '.join(sorted(aggregates))}. A lane is missing or still running -- publishing "
            f"this would put a partial row in the table. Nothing was written."
        )
    if args.expect_rows_per_cell is not None:
        short = {ds: entry.get("n") for ds, entry in aggregates.items()
                 if isinstance(entry, dict) and entry.get("n") != args.expect_rows_per_cell}
        if short:
            raise SystemExit(
                f"REFUSING TO WRITE: these cells do not have exactly "
                f"{args.expect_rows_per_cell} rows: {short}. Nothing was written."
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        handle.write(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")

    scores = [entry["score"] for entry in aggregates.values() if "score" in entry]
    print(f"merged {len(shards)} shards -> {args.output}")
    print(f"  datasets: {', '.join(sorted(aggregates))}")
    print(f"  rows: {len(rows)}")
    if scores:
        print(f"  mean over {len(scores)} datasets: {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    main()

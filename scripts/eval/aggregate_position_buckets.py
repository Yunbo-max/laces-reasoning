#!/usr/bin/env python3
"""Bucket sharded-babilong items by fact position/distance and report accuracy by bucket.

Reads the same `outputs_eval/sharded_babilong*/*.json` files as
`aggregate_sharded_babilong.py`, but breaks each (config, task) down by:
  - qa1: `position_bucket` (begin/middle/end tertile of the gold fact's char offset)
  - qa2/qa3: `fact_span` tertile (near/mid/far -- fractional distance between the
    first and last located supporting fact)
Items with no located fact (`position_bucket`/`fact_span` is None) are reported
separately as `unlocated` rather than silently dropped.

Usage:
  python scripts/eval/aggregate_position_buckets.py outputs_eval/sharded_babilong_v2
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def _span_bucket(span, edges=(1 / 3, 2 / 3)):
    if span is None:
        return "unlocated"
    if span < edges[0]:
        return "near"
    if span < edges[1]:
        return "mid"
    return "far"


def main(res_dir="outputs_eval/sharded_babilong_v2"):
    files = sorted(Path(res_dir).glob("*.json"))
    if not files:
        print(f"no results in {res_dir}")
        return

    for f in files:
        d = json.load(open(f))
        for task, blob in d.items():
            s, items = blob["summary"], blob["items"]
            conds = list(s["accuracy"].keys())
            is_qa1 = task == "qa1"

            buckets = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # bucket -> cond -> [hits, n]
            for rec in items:
                if is_qa1:
                    bucket = rec.get("position_bucket") or "unlocated"
                else:
                    bucket = _span_bucket(rec.get("fact_span"))
                for cond in conds:
                    g = rec["gen"].get(cond)
                    if g is None:
                        continue
                    buckets[bucket][cond][1] += 1
                    buckets[bucket][cond][0] += int(g.get("hit", False))

            label = f"M{s['num_agents']}_L{s['local_tokens']}_{s['babilong_length']}"
            bucket_order = ["begin", "middle", "end"] if is_qa1 else ["near", "mid", "far"]
            bucket_order = [b for b in bucket_order if b in buckets] + \
                (["unlocated"] if "unlocated" in buckets else [])
            if not bucket_order:
                continue
            print(f"\n=== {f.name} :: {task} ({label}) ===")
            hdr = f"{'bucket':10s}{'n':>5s}" + "".join(f"{c[:11]:>12s}" for c in conds)
            print(hdr)
            for bucket in bucket_order:
                cond_stats = buckets[bucket]
                n_b = max((v[1] for v in cond_stats.values()), default=0)
                line = f"{bucket:10s}{n_b:>5d}"
                for c in conds:
                    hits, n = cond_stats.get(c, [0, 0])
                    pct = f"{hits/n*100:.1f}%" if n else "n/a"
                    line += f"{pct:>12s}"
                print(line)


if __name__ == "__main__":
    main(*sys.argv[1:])

#!/usr/bin/env python3
"""Reconstruct per-dataset means for a RULER cell whose JSON was never written.

run_open_context_bench.py writes its result file exactly once, at the end, so a run that stalled
leaves NO partial JSON -- but every scored row was already echoed to stdout as
  [ruler/<dataset>/raw] <i>/<n> score=<float>
Those lines are sufficient to rebuild the aggregates. Use for evidence, not for publication:
the log does not carry predictions, so the cell cannot be rescored and cannot be gated on
decode_path.
"""
import argparse, collections, re, sys

LINE = re.compile(r'^\[(?P<suite>[a-z_]+)/(?P<ds>[A-Za-z0-9_.]+)/(?P<arm>[a-z0-9_]+)\] '
                  r'(?P<i>\d+)/(?P<n>\d+) score=(?P<score>[0-9.]+)')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--expect-n", type=int, default=None,
                    help="only report datasets that reached this row count")
    args = ap.parse_args()

    agg = collections.defaultdict(list)
    for line in open(args.log, errors="replace"):
        m = LINE.match(line)
        if m:
            agg[m.group("ds")].append(float(m.group("score")))

    if not agg:
        sys.exit("no scored rows found -- wrong log?")

    complete, partial = {}, {}
    for ds, v in agg.items():
        (complete if args.expect_n and len(v) == args.expect_n else partial)[ds] = v

    for title, group in (("COMPLETE", complete), ("PARTIAL", partial)):
        if not group:
            continue
        print(f"-- {title}")
        for ds in sorted(group):
            v = group[ds]
            mean = sum(v) / len(v)
            binary = set(v) <= {0.0, 1.0}
            flag = "  <-- exactly 0: metric-independent (legacy >= official)" if mean == 0.0 else ""
            print(f"   {ds:<24} n={len(v):<4} mean={mean:.4f} binary={binary}{flag}")
    if complete:
        print(f"\n   MACRO over {len(complete)} complete datasets: "
              f"{sum(sum(v)/len(v) for v in complete.values())/len(complete):.4f}")
    print("\n   NOTE: log-reconstructed. No predictions => cannot rescore, cannot verify decode_path.")


if __name__ == "__main__":
    main()

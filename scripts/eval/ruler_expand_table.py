#!/usr/bin/env python3
"""Build the RULER-expansion binding-capacity table from the driver logs.

Per-row `score=` lines in logs/qz/ruler-expand-*.log reproduce each cell aggregate exactly
(verified against llama-3.2-3b__ruler_expand.json: all 30 cells matched to 4 dp), so the
table can be built while the job is still running and does not depend on the shard JSONs.
Only cells with a full 100 rows are reported.
"""
import collections
import glob
import os
import pathlib
import re
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
LOGS = REPO / "logs" / "qz"
ARM_OF = {"champ": "champ-dynamic-30k", "global": "global-dit-50k",
          "llama": "llama-3.2-3b", "qwen": "qwen2.5-3b"}
ORDER = ["champ-dynamic-30k", "global-dit-50k", "llama-3.2-3b", "qwen2.5-3b"]
ROW = re.compile(r"\[ruler/(\S+?)/\w+\] (\d+)/100 score=([0-9.]+)")

# Three-factor design, measured from the parquet (see logs/qz/RULER_EXPAND_JOBS.md).
GOLDS = {"niah_single_1": 1, "niah_single_2": 1, "niah_single_3": 1, "niah_multikey_1": 1,
         "niah_multikey_2": 1, "niah_multikey_3": 1, "qa_2": 1, "fwe": 3, "qa_1": 3,
         "niah_multiquery": 4, "niah_multivalue": 4, "vt": 5, "cwe": 10}
BIND = {**GOLDS, "qa_1": 1}
DISTR = {("4096", "niah_multikey_2"): 200, ("8192", "niah_multikey_2"): 425,
         ("16384", "niah_multikey_2"): 900, ("4096", "niah_multikey_3"): 50,
         ("8192", "niah_multikey_3"): 125, ("16384", "niah_multikey_3"): 275}
for L in ("4096", "8192", "16384"):
    DISTR[(L, "niah_multikey_1")] = 3
    for t in ("niah_single_1", "niah_single_2", "niah_single_3",
              "niah_multiquery", "niah_multivalue"):
        DISTR[(L, t)] = 0
# uniq-sentence fraction of the haystack, measured from the parquet.
HAY = {"niah_single_1": "filler(0.01)", "niah_single_2": "essay(1.0)",
       "niah_single_3": "essay(1.0)", "niah_multikey_1": "essay(1.0)",
       "niah_multikey_2": "needles(1.0)", "niah_multikey_3": "needles(1.0)",
       "niah_multiquery": "essay(1.0)", "niah_multivalue": "essay(1.0)"}


def collect_shards(res):
    """Prefer the shard JSONs: full-precision scores.

    The log-line fallback prints score= at 3 decimal places, which truncates thirds
    (0.333333 -> 0.333) and shifts a 3-gold cell by up to 0.1 pp (measured on
    champ 8192_fwe: 25.644 from logs vs 25.667 from the JSON). Cells are taken from the
    JSON wherever the file exists and fall back to the logs only while a shard is still
    running.
    """
    import json
    for path in sorted(glob.glob(str(REPO / "results" / "benchmarks" / "ruler_expand"
                                    / "*__ruler_expand.json"))):
        base = os.path.basename(path)
        arm = next((a for a in ARM_OF if base.startswith(a)), None)
        if arm is None:
            continue
        by = collections.defaultdict(list)
        for r in json.loads(pathlib.Path(path).read_text()).get("rows", []):
            by[r["dataset"]].append(float(r["score"]))
        for cell, sc in by.items():
            if len(sc) == 100:
                res[ARM_OF[arm]][cell] = statistics.mean(sc) * 100
    return res


def collect_archived(res):
    """Fold in the 9 pre-existing cells, re-derived at n=100 (rows with id<100).

    The archived files store 200 rows whose ids are a parquet prefix 0..199, so the n=100
    value is the mean over id<100 -- the same 100 samples the new cells use. Canonical files
    are read only, never written.
    """
    import json
    for model in ORDER:
        path = REPO / "results" / "benchmarks" / "longctx" / f"{model}__ruler.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text()).get("rows", [])
        by = collections.defaultdict(list)
        for r in rows:
            if int(r["id"]) < 100:
                by[r["dataset"]].append(float(r["score"]))
        for cell, sc in by.items():
            if len(sc) == 100 and cell not in res[model]:
                res[model][cell] = statistics.mean(sc) * 100
                ARCHIVED.add((model, cell))
    return res


ARCHIVED = set()


def collect():
    res = collections.defaultdict(dict)
    for path in sorted(glob.glob(str(LOGS / "ruler-expand-*.log"))):
        base = os.path.basename(path)
        if "driver" in base:
            continue
        arm = next((a for a in ARM_OF if a in base), None)
        if arm is None:
            continue
        by = collections.defaultdict(list)
        for line in open(path, errors="replace"):
            m = ROW.search(line)
            if m:
                by[m.group(1)].append(float(m.group(3)))
        for cell, scores in by.items():
            if len(scores) >= 100:
                res[ARM_OF[arm]][cell] = statistics.mean(scores[:100]) * 100
    return res


def main():
    res = collect_archived(collect_shards(collect()))
    cells = sorted(set().union(*[set(v) for v in res.values()]) if res else set(),
                   key=lambda c: (BIND.get(c.split("_", 1)[1], 99),
                                  c.split("_", 1)[1], int(c.split("_")[0])))
    print("| task | golds | bind | same-type distr | haystack(uniq) | len | "
          + " | ".join(ORDER) + " |")
    print("|---" * (6 + len(ORDER)) + "|")
    for c in cells:
        L, task = c.split("_", 1)
        d = DISTR.get((L, task))
        cols = []
        for m in ORDER:
            if c not in res[m]:
                cols.append("--")
            else:
                # dagger = re-derived at n=100 from the archived n=200 cell, not re-run.
                cols.append(f"{res[m][c]:.1f}" + ("+" if (m, c) in ARCHIVED else ""))
        print(f"| {task} | {GOLDS.get(task,'?')} | {BIND.get(task,'?')} | "
              f"{'' if d is None else d} | {HAY.get(task,'')} | {L} | "
              + " | ".join(cols) + " |")
    total = sum(len(v) for v in res.values())
    print(f"\n[cells at n=100: {total}; of which {len(ARCHIVED)} marked + are re-derived "
          f"from the archived n=200 files, the rest measured in this run]", file=sys.stderr)
    missing = [(m, c) for m in ORDER for c in cells if c not in res[m]]
    print(f"[missing: {len(missing)}] " + ", ".join(f"{m}/{c}" for m, c in missing),
          file=sys.stderr)


if __name__ == "__main__":
    main()

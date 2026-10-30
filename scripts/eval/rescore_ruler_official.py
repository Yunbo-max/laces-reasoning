#!/usr/bin/env python3
"""Recompute archived RULER results under the official per-task metric.

``run_open_context_bench.py`` used to score every RULER task except ``vt`` with
``max`` over the gold list.  RULER itself only does that for the two QA tasks
(``string_match_part``); every other task uses ``string_match_all``, the
per-sample recall.  ``cwe`` ships 10 golds and ``fwe`` 3, so ``max`` inflated
those cells several-fold.

Every run archives its raw ``prediction`` string, so the correction needs no GPU:
this script re-derives each row's score from the stored prediction and rewrites
the aggregates.  Run with ``--write`` to update the JSONs in place (the original
harness score is preserved per row as ``score_legacy``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Mirrors run_open_context_bench.RULER_MATCH_PART_TASKS.
MATCH_PART_TASKS = {"qa_1", "qa_2"}


def official_score(task: str, prediction: str, references: list[str]) -> float:
    prediction = prediction.lower()
    hits = [float(reference.lower() in prediction) for reference in references]
    if not hits:
        return 0.0
    if task in MATCH_PART_TASKS:
        return max(hits)
    return sum(hits) / len(hits)


def task_of(dataset: str) -> str:
    """'4096_niah_single_1' -> 'niah_single_1'; '16384_qa_1' -> 'qa_1'."""
    head, _, rest = dataset.partition("_")
    return rest if head.isdigit() else dataset


def rescore(path: Path, write: bool) -> dict:
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    per_cell: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("suite") != "ruler":
            continue
        dataset = row["dataset"]
        legacy = float(row["score"])
        fixed = official_score(task_of(dataset), row["prediction"], row["answers"])
        per_cell.setdefault(dataset, []).append((legacy, fixed))
        if write:
            row.setdefault("score_legacy", legacy)
            row["score"] = fixed

    cells = {
        dataset: {
            "n": len(pairs),
            "legacy": sum(a for a, _ in pairs) / len(pairs),
            "official": sum(b for _, b in pairs) / len(pairs),
        }
        for dataset, pairs in per_cell.items()
    }
    if not cells:
        return {"path": str(path), "cells": {}, "macro_legacy": None, "macro_official": None}

    summary = {
        "path": str(path.relative_to(REPO) if path.is_relative_to(REPO) else path),
        "cells": cells,
        "macro_legacy": sum(c["legacy"] for c in cells.values()) / len(cells),
        "macro_official": sum(c["official"] for c in cells.values()) / len(cells),
    }
    if write:
        payload["aggregates"] = {k: v["official"] for k, v in cells.items()}
        payload["aggregates_legacy_max_metric"] = {k: v["legacy"] for k, v in cells.items()}
        payload["metric_note"] = (
            "RULER string_match_all (per-sample recall) for all tasks except qa_1/qa_2, "
            "which use string_match_part (max over golds). Rescored offline from the "
            "archived predictions by scripts/eval/rescore_ruler_official.py."
        )
        path.write_text(json.dumps(payload, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="result JSONs (default: results/benchmarks/**/*__ruler*.json)")
    parser.add_argument("--write", action="store_true", help="rewrite the JSONs in place")
    parser.add_argument("--ledger", type=Path, default=None, help="also dump the comparison as JSON")
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths] or sorted(
        (REPO / "results" / "benchmarks").rglob("*__ruler*.json"))

    summaries = [rescore(p, args.write) for p in paths]
    summaries = [s for s in summaries if s["cells"]]

    width = max(len(Path(s["path"]).parent.name + "/" + Path(s["path"]).name) for s in summaries)
    print(f"{'result file':<{width}}  {'cells':>5} {'legacy':>8} {'official':>9} {'delta':>8}")
    for s in sorted(summaries, key=lambda s: -s["macro_official"]):
        label = Path(s["path"]).parent.name + "/" + Path(s["path"]).name
        print(f"{label:<{width}}  {len(s['cells']):>5} {s['macro_legacy']:>8.4f} "
              f"{s['macro_official']:>9.4f} {s['macro_official'] - s['macro_legacy']:>+8.4f}")

    if args.ledger:
        args.ledger.write_text(json.dumps(summaries, indent=2))
        print(f"wrote {args.ledger}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Re-score existing MemEval result JSONs in place with the current scorer.

Why this exists: the generations are the expensive part (GPU minutes per row) and the
metric is pure post-processing, so a metric fix must not require a re-run. This recomputes
``rows[*].score`` and ``aggregates`` from the stored ``prediction`` strings, and re-attaches
the question from the dataset (early runs did not store it).

Usage:
  python3 scripts/eval/rescore_memeval.py results/benchmarks/longctx/*__memeval.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORACLE = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/"
          "datasets--xiaowu0162--longmemeval/snapshots/"
          "2ec2a557f339b6c0369619b1ed5793734cc87533/longmemeval_oracle")


def load_scorer():
    spec = importlib.util.spec_from_file_location(
        "rocb", REPO / "scripts" / "eval" / "run_open_context_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.longmemeval_score_approx


def main(paths: list[str]) -> int:
    score_fn = load_scorer()
    questions = {str(item.get("question_id", index)): item.get("question", "")
                 for index, item in enumerate(json.loads(Path(ORACLE).read_text()))}
    for raw in paths:
        path = Path(raw)
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            # `longctx/rwkv7-world3__memeval_loopdecode.json` is a bare 50-element list (a probe,
            # not a cell). Without this guard `.get` raises AttributeError and kills the whole batch.
            print(f"[skip] {path.name}: bare list, not a result cell")
            continue
        if data.get("config", {}).get("suite") != "memeval":
            print(f"[skip] {path.name}: not a memeval result")
            continue
        per_dataset: dict[str, list[float]] = {}
        for row in data["rows"]:
            question = row.get("question") or questions.get(str(row["id"]), "")
            row["score"] = score_fn(row["prediction"], row["answers"][0], question)
            per_dataset.setdefault(row["dataset"], []).append(row["score"])
        before = {k: v["score"] for k, v in data["aggregates"].items()}
        data["aggregates"] = {k: {"n": len(v), "score": sum(v) / len(v) if v else 0.0}
                              for k, v in per_dataset.items()}
        data.setdefault("rescored", []).append(
            {"by": "scripts/eval/rescore_memeval.py", "previous_aggregates": before})
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        after = {k: round(v["score"], 4) for k, v in data["aggregates"].items()}
        print(f"[rescored] {path.name}: {before} -> {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

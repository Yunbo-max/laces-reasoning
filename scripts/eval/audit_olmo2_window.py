#!/usr/bin/env python3
"""Decide, BEFORE spending GPU time, which long-context cells OLMo-2-1B can actually see.

OLMo-2-0425-1B declares ``max_position_embeddings: 4096`` (verified: so do the 7B, 13B and
32B -- the whole family is 4k, so no size choice escapes this). The harness therefore clamps
its input budget to ``4096 - max_new_tokens``, which is 4064 for babilong (32 new), 3968 for
ruler/infinitebench/memeval (128 new) and 3584 for longbench (512 new).

The published precedent for what happens if this is ignored is phi-2: at
``--max_input_tokens 32768`` against a 2048-position table it emitted nothing but newlines
and scored an exact 0.0 on all five InfiniteBench datasets. Clamping moved its RULER
aggregate 0.0000 -> 0.4103 -- and one cell 0.000 -> 0.990 while the model saw 44% of the
document. Neither number is a capability measurement. This script computes the
``seen_fraction`` each cell WOULD have, using OLMo-2's own tokenizer on the same local
dataset files the harness reads, so cells can be classified as reportable or
not-measurable-within-this-window without running the model at all.

Tokenizer-only: no GPU, no weights. Safe to run while the checkpoint is still downloading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
CACHE = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub")
SNAP = (CACHE / "models--allenai--OLMo-2-0425-1B/snapshots"
        / "a1847dff35000b4271fa70afc5db10fd29fedbdf")

# Exactly the cells the 9 published baselines were run on, read from the `config` blocks of
# results/benchmarks/longctx/qwen2.5-3b__*.json rather than from prose.
SUITES = {
    "babilong": {
        "data_dir": CACHE / "datasets--RMT-team--babilong/snapshots"
                    / "ee0d588794c7ac098062ee0d247c733d62e94fe2/data",
        "datasets": ["qa1/4k", "qa1/8k", "qa1/16k", "qa2/4k"],
        "max_samples": 100, "max_new_tokens": 32,
    },
    "ruler": {
        "data_dir": CACHE / "datasets--simonjegou--ruler/snapshots"
                    / "24adceac8a0e6532936e8d721cd9e9084d2e4686",
        "datasets": ["4096/niah_single_1", "4096/niah_multikey_1", "4096/vt", "4096/cwe",
                     "4096/fwe", "4096/qa_1", "16384/niah_single_1", "16384/vt",
                     "16384/cwe"],
        "max_samples": 200, "max_new_tokens": 128,
    },
    "longbench": {
        "data_dir": Path("/inspire/hdd/global_user/zhangjiaquan-253108540222"
                         "/longbench_local/extract/data"),
        "datasets": ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa"],
        "max_samples": 50, "max_new_tokens": 512,
    },
    "infinitebench": {
        "data_dir": CACHE / "datasets--xinrongzhang2022--InfiniteBench/snapshots"
                    / "90f0394333616266d9fe85824ceaf505093cbaa5",
        "datasets": ["longbook_qa_eng", "longbook_choice_eng", "kv_retrieval", "code_debug",
                     "math_find"],
        "max_samples": 50, "max_new_tokens": 128,
    },
    "memeval": {
        "data_dir": CACHE / "datasets--xiaowu0162--longmemeval/snapshots"
                    / "2ec2a557f339b6c0369619b1ed5793734cc87533",
        "datasets": ["longmemeval_oracle"],
        "max_samples": 50, "max_new_tokens": 128,
    },
}

WINDOW = 4096   # OLMo-2 max_position_embeddings, all sizes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/benchmarks/longctx"
                                        / "olmo-2-1b__window_audit.json"))
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO))
    from transformers import AutoTokenizer
    from scripts.eval.run_open_context_bench import (
        load_babilong_rows, load_ruler_rows, load_longbench_rows, load_infinitebench_rows,
        load_memeval_rows, tokenizer_wants_bos,
    )

    tok = AutoTokenizer.from_pretrained(str(SNAP), trust_remote_code=True,
                                        local_files_only=True)
    wants_bos = tokenizer_wants_bos(tok)
    print(f"[tok] {type(tok).__name__} bos_id={tok.bos_token_id} wants_bos={wants_bos}")

    prompt_map = json.loads((Path("/inspire/hdd/global_user/zhangjiaquan-253108540222"
                                 "/longbench_local/pinned_spec/dataset2prompt.json")
                             ).read_text())

    report = {"model": "allenai/OLMo-2-0425-1B",
              "revision": "a1847dff35000b4271fa70afc5db10fd29fedbdf",
              "positional_window": WINDOW, "prepend_bos": wants_bos,
              "note": ("seen_fraction is computed from OLMo-2's own tokenizer over the same "
                       "local files the harness reads. budget = window - max_new_tokens, "
                       "matching effective_input_budget() in run_open_context_bench.py."),
              "cells": {}}

    for suite, spec in SUITES.items():
        budget = max(2, WINDOW - spec["max_new_tokens"])
        data_dir = Path(spec["data_dir"])
        for ds in spec["datasets"]:
            key = f"{suite}/{ds}"
            try:
                if suite == "babilong":
                    rows = load_babilong_rows(data_dir / f"{ds}.json")
                elif suite == "ruler":
                    length, task = ds.split("/", 1)
                    cand = data_dir / length / "validation.jsonl"
                    rows = (load_ruler_rows(cand, task) if cand.exists()
                            else load_ruler_rows(data_dir / length, task))
                elif suite == "longbench":
                    rows = load_longbench_rows(data_dir / f"{ds}.jsonl", ds, prompt_map)
                elif suite == "infinitebench":
                    rows = load_infinitebench_rows(data_dir / f"{ds}.jsonl")
                else:
                    rows = load_memeval_rows(data_dir / ds)
            except Exception as exc:                      # loud, never silent
                report["cells"][key] = {"error": repr(exc)}
                print(f"[{key}] LOAD FAILED {exc!r}")
                continue

            rows = rows[: spec["max_samples"]]
            lens = [len(tok(r["prompt"], add_special_tokens=False).input_ids) for r in rows]
            room = budget - (1 if wants_bos else 0)
            seen = [min(n, room) for n in lens]
            frac = sum(seen) / sum(lens) if sum(lens) else None
            n_trunc = sum(1 for n in lens if n > room)
            cell = {"n": len(rows), "budget": budget, "room": room,
                    "mean_prompt_tokens_untruncated": sum(lens) / len(lens) if lens else None,
                    "max_prompt_tokens_untruncated": max(lens) if lens else None,
                    "mean_input_tokens_seen": sum(seen) / len(seen) if seen else None,
                    "rows_truncated": n_trunc,
                    "seen_fraction": frac,
                    "reportable": bool(n_trunc == 0)}
            report["cells"][key] = cell
            flag = "OK" if cell["reportable"] else "NOT-MEASURABLE"
            print(f"[{key:42s}] n={cell['n']:3d} mean_full={cell['mean_prompt_tokens_untruncated']:9.0f} "
                  f"seen={frac:.4f} trunc={n_trunc:3d}/{len(rows)}  {flag}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

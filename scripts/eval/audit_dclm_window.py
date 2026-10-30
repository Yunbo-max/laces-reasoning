#!/usr/bin/env python3
"""Decide, BEFORE spending GPU time, which long-context cells DCLM-1B can actually see.

TRI-ML/DCLM-1B is an ``openlm`` checkpoint and declares its context as ``seq_len: 2048``
(config.json, revision f43c272dab1b06bc5fdd6bbb5819f8a30ef2833b). That is the SHORTEST window
of any baseline in this paper -- shorter than phi-2's 2048-position table only in the sense
that it is the same, and phi-2 is the published precedent for what happens when the window is
ignored: at ``--max_input_tokens 32768`` it indexed 16x past its own position table, emitted
newlines, and scored an exact 0.0 on all five InfiniteBench datasets. Clamping moved its RULER
aggregate 0.0000 -> 0.4103 and one cell 0.000 -> 0.990 -- earned while seeing 44% of the
document, which is not a capability measurement either.

Two traps are specific to this family and both are checked here rather than assumed:

  1. ``positional_window()`` in run_open_context_bench.py reads ``max_position_embeddings``
     then ``n_positions``. An ``openlm`` config has NEITHER -- it says ``seq_len``. So the
     unpatched harness returns ``window=None`` for DCLM, classifies it as having no positional
     ceiling (the branch meant for recurrent models), and feeds it 16k/32k-token prompts
     straight into a 2048 rotary/qk-norm attention stack. That is the phi-2 failure with a
     different config key, and it fails OPEN, silently. The audit below reports what the
     harness would compute vs. the true window so the gap is on the record.
  2. ``add_special_tokens=False`` suppresses BOS. DCLM's tokenizer_config declares
     ``bos_token: "<|endoftext|>"`` (a GPTNeoX-style tokenizer where BOS==EOS==UNK==id 0),
     so the empirical probe, not the flag, decides whether the harness must prepend it.

Tokenizer-only: no GPU, no weights, no network. Safe to run while the checkpoint downloads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
CACHE = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub")
REVISION = "f43c272dab1b06bc5fdd6bbb5819f8a30ef2833b"
SNAP = CACHE / "models--TRI-ML--DCLM-1B/snapshots" / REVISION

# Exactly the cells the published baselines were run on, read from the `config` blocks of
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


def declared_window() -> tuple[int, dict]:
    """DCLM's real context length, read from its own config.json -- not assumed.

    Returns (window, evidence). ``openlm`` nests nothing: the fields sit at the top level of
    config.json, but under ``seq_len`` rather than the ``max_position_embeddings`` the harness
    looks for. Both are reported so the mismatch is auditable.
    """
    cfg = json.loads((SNAP / "config.json").read_text())
    evidence = {k: cfg.get(k) for k in
                ("model_type", "architectures", "seq_len", "max_position_embeddings",
                 "n_positions", "dim", "n_layers", "n_heads", "vocab_size",
                 "positional_embedding_type", "qk_norm", "norm_type", "ffn_type", "model")}
    window = cfg.get("seq_len")
    if window is None:
        raise SystemExit(f"config.json has no seq_len; cannot establish the window: {evidence}")
    # What the UNPATCHED harness would compute -- the silent-failure-open case.
    evidence["harness_positional_window_unpatched"] = (
        cfg.get("max_position_embeddings") or cfg.get("n_positions") or None)
    return int(window), evidence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/benchmarks/longctx"
                                        / "dclm-1b__window_audit.json"))
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO))
    from transformers import AutoTokenizer
    from scripts.eval.run_open_context_bench import (
        load_babilong_rows, load_ruler_rows, load_longbench_rows, load_infinitebench_rows,
        load_memeval_rows, tokenizer_wants_bos,
    )

    window, cfg_evidence = declared_window()
    print(f"[config] {json.dumps(cfg_evidence, indent=2)}")
    print(f"[config] true window = seq_len = {window}; "
          f"unpatched harness would see {cfg_evidence['harness_positional_window_unpatched']!r}")

    tok = AutoTokenizer.from_pretrained(str(SNAP), trust_remote_code=True,
                                        local_files_only=True)
    wants_bos = tokenizer_wants_bos(tok)
    probe = {"class": type(tok).__name__,
             "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
             "vocab_size": len(tok),
             "with_special": tok("x", add_special_tokens=True).input_ids,
             "without_special": tok("x", add_special_tokens=False).input_ids,
             "add_bos_token_attr": getattr(tok, "add_bos_token", None),
             "wants_bos_probed": wants_bos}
    print(f"[tok] {json.dumps(probe)}")

    prompt_map = json.loads((Path("/inspire/hdd/global_user/zhangjiaquan-253108540222"
                                 "/longbench_local/pinned_spec/dataset2prompt.json")
                             ).read_text())

    report = {"model": "TRI-ML/DCLM-1B", "revision": REVISION,
              "positional_window": window, "window_source": "config.json:seq_len",
              "config_evidence": cfg_evidence, "tokenizer_probe": probe,
              "prepend_bos": wants_bos,
              "note": ("seen_fraction is computed from DCLM's own tokenizer over the same "
                       "local files the harness reads. budget = seq_len - max_new_tokens, "
                       "mirroring effective_input_budget() in run_open_context_bench.py. A "
                       "cell with rows_truncated > 0 is NOT reportable as a capability "
                       "number: the model never saw the evidence the question asks about."),
              "cells": {}}

    for suite, spec in SUITES.items():
        budget = max(2, window - spec["max_new_tokens"])
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
            cell = {"n": len(rows), "max_new_tokens": spec["max_new_tokens"],
                    "budget": budget, "room": room,
                    "mean_prompt_tokens_untruncated": sum(lens) / len(lens) if lens else None,
                    "max_prompt_tokens_untruncated": max(lens) if lens else None,
                    "min_prompt_tokens_untruncated": min(lens) if lens else None,
                    "mean_input_tokens_seen": sum(seen) / len(seen) if seen else None,
                    "rows_truncated": n_trunc,
                    "seen_fraction": frac,
                    "reportable": bool(n_trunc == 0)}
            report["cells"][key] = cell
            flag = "OK" if cell["reportable"] else "NOT-MEASURABLE"
            print(f"[{key:42s}] n={cell['n']:3d} "
                  f"mean_full={cell['mean_prompt_tokens_untruncated']:9.0f} "
                  f"seen={frac:.4f} trunc={n_trunc:3d}/{len(rows)}  {flag}")

    n_ok = sum(1 for c in report["cells"].values() if c.get("reportable"))
    report["summary"] = {"cells_total": len(report["cells"]), "cells_reportable": n_ok}
    print(f"[summary] {n_ok}/{len(report['cells'])} cells reportable within a "
          f"{window}-token window")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

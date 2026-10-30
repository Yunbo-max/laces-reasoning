#!/usr/bin/env python3
"""Does DCLM-1B need a leading <|endoftext|>, and does its absence change any answer?

WHY. The brief's trap: the harness tokenises with `add_special_tokens=False`, and reading a
tokenizer's `add_bos_token` flag is worth ZERO information here (all 7 baselines read False while
gemma actually needed one -- +40pp -- and llama did not, McNemar 20:20 p=1.0). So it has to be
measured. DCLM raises the question sharply: `tokenizer_config.json` sets
bos == eos == unk == "<|endoftext|>" (id 0), and `open_lm` trains on *packed* streams where
documents are separated by that very token -- so almost every training document was preceded by
id 0. That is the same structural argument that made it matter for gemma.

BUT "suspicious" is not "wrong" (the llama lesson). The decision rule used here:
  * `prepend_bos` is only worth adopting if it CHANGES PREDICTIONS on real task rows, and
  * changes them for the BETTER on a paired comparison of the SAME rows (McNemar), never by
    comparing two marginal accuracies.
Token-count deltas are explicitly NOT the criterion: on a truncation-bound model the BOS is taken
out of the input budget, so `input_tokens` can be identical while the content shifts. Here we use
prediction-changed as the landing criterion.

Two sub-questions, kept separate:
  (A) mechanical: does the tokenizer inject id 0 on its own? (probe, don't read the flag)
  (B) behavioural: with vs without a hand-prepended id 0, on N real rows, greedy -- how many
      predictions change, and what is the paired win/loss?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222"
DEFAULT_MODEL = (f"{ROOT}/hf_cache/hub/models--TRI-ML--DCLM-1B/snapshots/"
                 "f43c272dab1b06bc5fdd6bbb5819f8a30ef2833b")
TASKDATA = f"{ROOT}/research/DiffRwkv/baseline/Cola-DLM/generate_task_data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=DEFAULT_MODEL)
    ap.add_argument("--task", default="obqa")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    import open_lm.hf  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_generic_gen_match import build_prompt, get_ground_truth

    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True,
                                        local_files_only=True)
    # ---- (A) mechanical probe: never trust the flag, tokenise and look -----------------
    probe = "The capital of France is Paris."
    with_special = tok(probe, add_special_tokens=True).input_ids
    without = tok(probe, add_special_tokens=False).input_ids
    bos_id = tok.bos_token_id
    injects = with_special != without
    print(f"[A] tokenizer={type(tok).__name__} bos_token_id={bos_id} eos_token_id={tok.eos_token_id}")
    print(f"[A] add_bos_token flag reads: {getattr(tok, 'add_bos_token', None)!r} "
          f"(ZERO information -- probing instead)")
    print(f"[A] with_special   = {with_special[:8]}")
    print(f"[A] without_special= {without[:8]}")
    print(f"[A] tokenizer injects a BOS on its own: {injects}")

    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[a.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, trust_remote_code=True, dtype=dt, local_files_only=True
    ).to(a.device).eval()
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token

    rows = []
    with open(Path(TASKDATA) / f"{a.task}.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= a.n:
                break
            if line.strip():
                rows.append(json.loads(line))

    # ---- (B) behavioural: paired, same rows, greedy ------------------------------------
    changed = both_right = both_wrong = bos_only = nobos_only = 0
    details = []
    for item in rows:
        prompt = build_prompt(a.task, item)
        gt = get_ground_truth(item).strip().lower()
        base = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        variants = {}
        for name, ids in (("no_bos", base),
                          ("bos", torch.cat([torch.tensor([[bos_id]]), base], dim=1))):
            ids = ids.to(a.device)
            with torch.no_grad():
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     max_new_tokens=a.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            variants[name] = tok.decode(out[0, ids.shape[1]:].tolist(),
                                        skip_special_tokens=True).strip()
        # Scoring here is deliberately crude but POSITIONAL, not bare-substring: credit only if
        # the ground truth appears at the very start of the continuation, which is where the
        # generate-then-match protocol expects the answer. Bare `gt in gen` would credit a model
        # that merely echoed the option list (the documented false-positive mode).
        def hit(gen: str) -> bool:
            return bool(gt) and gen.strip().lower().startswith(gt)
        h_no, h_bos = hit(variants["no_bos"]), hit(variants["bos"])
        if variants["no_bos"] != variants["bos"]:
            changed += 1
        if h_no and h_bos:
            both_right += 1
        elif not h_no and not h_bos:
            both_wrong += 1
        elif h_bos:
            bos_only += 1
        else:
            nobos_only += 1
        details.append({"gt": gt, "no_bos": variants["no_bos"], "bos": variants["bos"],
                        "hit_no_bos": h_no, "hit_bos": h_bos})

    n = len(rows)
    print(f"\n[B] task={a.task} n={n} greedy max_new_tokens={a.max_new_tokens} "
          f"device={a.device} dtype={a.dtype}")
    print(f"[B] predictions CHANGED by prepending BOS: {changed}/{n} "
          f"({100.0 * changed / max(1, n):.1f}%)")
    print(f"[B] paired (McNemar) table: both_right={both_right} both_wrong={both_wrong} "
          f"bos_only={bos_only} no_bos_only={nobos_only}")
    disc = bos_only + nobos_only
    verdict = "NO BOS ARM NEEDED"
    if disc:
        # Exact binomial (two-sided) on the discordant pairs -- the correct paired test. Two
        # marginal accuracies with overlapping CIs would say nothing about the same rows.
        from math import comb
        k = min(bos_only, nobos_only)
        p = sum(comb(disc, i) for i in range(0, k + 1)) / (2 ** disc) * 2
        p = min(1.0, p)
        print(f"[B] exact McNemar on {disc} discordant pairs: p={p:.4f}")
        if p < 0.05 and bos_only > nobos_only:
            verdict = "BOS HELPS SIGNIFICANTLY -- rerun the suite with a prepended BOS"
        elif p < 0.05:
            verdict = "BOS HURTS SIGNIFICANTLY -- keep add_special_tokens=False"
        else:
            verdict = ("NO SIGNIFICANT PAIRED DIFFERENCE -- keep the published protocol "
                       "(add_special_tokens=False), same call as llama")
    print(f"\n[verdict] {verdict}")
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(
            {"model_dir": a.model_dir, "task": a.task, "n": n, "dtype": a.dtype,
             "tokenizer_class": type(tok).__name__, "bos_token_id": bos_id,
             "tokenizer_injects_bos": injects,
             "add_bos_token_flag": getattr(tok, "add_bos_token", None),
             "predictions_changed": changed, "both_right": both_right,
             "both_wrong": both_wrong, "bos_only": bos_only, "no_bos_only": nobos_only,
             "verdict": verdict, "rows": details}, indent=2))
        print(f"[probe] wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Why does rwkv7's FIRST generated token differ between `generate` and a direct forward?

The fla recurrent-cache bug cannot explain this: it corrupts decode steps 2..N, while the
prefill is shared. Yet the fingerprint shows unpatched-vs-loop agreement of only 2/20 at
k=1 while unpatched-vs-patched is 20/20 at k=1. So something in the PREFILL differs.

Only two inputs differ between the two prefills:

  (a) ``attention_mask``: ``generate`` always passes one; a direct ``model(input_ids=...)``
      passes None. fla's rwkv7 layer does ``hidden_states.mul(am)`` and ``v = v * am``,
      which for an all-ones mask should be an exact no-op -- unless the mask's dtype
      promotes and changes the arithmetic.
  (b) ``logits_to_keep``: ``generate`` passes 1, so the lm_head runs on a [1,1,H] slice;
      a direct call uses the signature default 0, so ``hidden_states[:, -0:]`` is the WHOLE
      sequence and the lm_head runs on [1,T,H]. Same maths, different GEMM shape, so bf16
      rounding can differ.

This isolates which one flips the argmax, by evaluating the last-position logits under each
combination and comparing them to each other and to what generate actually emits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from falsify_zero_baselines import (  # noqa: E402
    load_babilong_rows, load_model, truncate_ids,
)

BABI = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/"
            "datasets--RMT-team--babilong/snapshots/"
            "ee0d588794c7ac098062ee0d247c733d62e94fe2/data")
MODEL = "/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/laces-reasoning/models/RWKV7-Goose-World3-2.9B-HF"


@torch.no_grad()
def last_logits(model, ids, mask=None, logits_to_keep=None):
    kwargs = {"input_ids": ids, "use_cache": True, "return_dict": True}
    if mask is not None:
        kwargs["attention_mask"] = mask
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = logits_to_keep
    out = model(**kwargs)
    return out.logits[0, -1].float()


def main():
    model, tokenizer = load_model(MODEL, "cuda")
    rows = load_babilong_rows(BABI / "qa1/4k.json")[:5]
    records = []
    for row in rows:
        ids = truncate_ids(tokenizer, row["prompt"], 16384, "cuda")
        ones_long = torch.ones_like(ids)                      # what generate builds
        ones_bool = torch.ones_like(ids, dtype=torch.bool)
        variants = {
            "nomask_keep0": last_logits(model, ids),                       # the loop's call
            "nomask_keep1": last_logits(model, ids, logits_to_keep=1),
            "mask_long_keep0": last_logits(model, ids, mask=ones_long),
            "mask_long_keep1": last_logits(model, ids, mask=ones_long, logits_to_keep=1),
            "mask_bool_keep1": last_logits(model, ids, mask=ones_bool, logits_to_keep=1),
        }
        # What generate actually emits for exactly one new token.
        gen = model.generate(input_ids=ids, max_new_tokens=1, do_sample=False,
                             pad_token_id=(getattr(tokenizer, "pad_token_id", None)
                                           or getattr(tokenizer, "eos_token_id", None)))
        gen_token = int(gen[0, ids.shape[1]])

        rec = {"id": row["id"], "input_tokens": int(ids.shape[1]),
               "generate_first_token": gen_token,
               "argmax": {k: int(v.argmax()) for k, v in variants.items()},
               "decoded": {k: tokenizer.decode([int(v.argmax())]) for k, v in variants.items()},
               "generate_decoded": tokenizer.decode([gen_token])}
        base = variants["nomask_keep0"]
        rec["max_abs_diff_vs_loopcall"] = {
            k: float((v - base).abs().max()) for k, v in variants.items()}
        # Is the loop's choice even near-tied, or is the gap large?
        top = base.topk(2).values
        rec["loopcall_top1_minus_top2"] = float(top[0] - top[1])
        records.append(rec)
        print(json.dumps(rec, indent=2), flush=True)

    agree = {k: sum(int(r["argmax"][k] == r["generate_first_token"]) for r in records)
             for k in records[0]["argmax"]}
    print("\n=== how often each variant's argmax == generate's emitted token (n=%d) ==="
          % len(records), flush=True)
    print(json.dumps(agree, indent=2), flush=True)
    out = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/"
               "results/benchmarks/falsify_zero/rwkv7-prefill-divergence.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": records, "agreement_with_generate": agree},
                              indent=2) + "\n")


if __name__ == "__main__":
    main()

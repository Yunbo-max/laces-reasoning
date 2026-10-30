#!/usr/bin/env python3
"""What does `generate` do to rwkv7's PREFILL that a direct forward does not?

Established by probe_rwkv7_prefill_divergence.py: attention_mask (bool or long) and
logits_to_keep are bit-exactly inert -- all five combinations returned identical last-position
logits (max abs diff 0.0000) -- yet every one of them disagrees with the token `generate`
actually emits. So the corruption is inside `generate`'s own call setup, at the prefill, not
in the decode loop the known fla cache bug affects.

Remaining hypothesis: transformers 5.x `generate` ALWAYS materialises a cache object before
prefill and passes it in. fla's RWKV7Model.forward does

    if use_cache and not isinstance(past_key_values, Cache):
        past_key_values = Cache.from_legacy_cache(past_key_values)

where ``Cache`` is *fla's own* class from fla.models.utils, NOT transformers' Cache. An HF
DynamicCache is therefore not an instance of it, so it gets fed to
``fla.Cache.from_legacy_cache``, which expects the legacy tuple format. If that yields a cache
that reads as already-populated, the prefill runs against a bogus conv/recurrent state and the
FIRST token is already wrong -- exactly what we observe.

This tests it directly by calling forward with each cache object generate might pass.
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


def main():
    model, tokenizer = load_model(MODEL, "cuda")
    from fla.models.utils import Cache as FlaCache
    from transformers.cache_utils import DynamicCache

    print(f"fla Cache is HF Cache subclass: "
          f"{issubclass(FlaCache, __import__('transformers').cache_utils.Cache)}", flush=True)
    print(f"DynamicCache is fla Cache: {issubclass(DynamicCache, FlaCache)}", flush=True)

    rows = load_babilong_rows(BABI / "qa1/4k.json")[:5]
    records = []
    for row in rows:
        ids = truncate_ids(tokenizer, row["prompt"], 16384, "cuda")
        variants = {}
        with torch.no_grad():
            variants["cache_None"] = model(input_ids=ids, use_cache=True,
                                           return_dict=True).logits[0, -1].float()
            try:
                variants["cache_HF_DynamicCache"] = model(
                    input_ids=ids, past_key_values=DynamicCache(), use_cache=True,
                    return_dict=True).logits[0, -1].float()
            except Exception as error:
                variants["cache_HF_DynamicCache"] = f"RAISED {error!r}"
            try:
                variants["cache_fla_Cache"] = model(
                    input_ids=ids, past_key_values=FlaCache(), use_cache=True,
                    return_dict=True).logits[0, -1].float()
            except Exception as error:
                variants["cache_fla_Cache"] = f"RAISED {error!r}"

        gen = model.generate(input_ids=ids, max_new_tokens=1, do_sample=False,
                             pad_token_id=(getattr(tokenizer, "pad_token_id", None)
                                           or getattr(tokenizer, "eos_token_id", None)))
        gen_token = int(gen[0, ids.shape[1]])
        rec = {"id": row["id"], "generate_first_token": gen_token,
               "generate_decoded": tokenizer.decode([gen_token]), "argmax": {}}
        for name, value in variants.items():
            if isinstance(value, str):
                rec["argmax"][name] = value
            else:
                token = int(value.argmax())
                rec["argmax"][name] = {"token": token,
                                       "decoded": tokenizer.decode([token]),
                                       "matches_generate": token == gen_token}
        records.append(rec)
        print(json.dumps(rec, indent=2), flush=True)

    print("\n=== which cache object reproduces generate's first token? ===", flush=True)
    for name in records[0]["argmax"]:
        hits = sum(1 for r in records
                   if isinstance(r["argmax"][name], dict)
                   and r["argmax"][name]["matches_generate"])
        print(f"  {name:24s} {hits}/{len(records)}", flush=True)
    out = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/"
               "results/benchmarks/falsify_zero/rwkv7-cache-object-cause.json")
    out.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

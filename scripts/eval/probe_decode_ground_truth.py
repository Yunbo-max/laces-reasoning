#!/usr/bin/env python3
"""Ground-truth decode check: is the single-token recurrent loop's cache actually correct?

Motivation. For rwkv7, HF ``generate`` and a direct forward disagree on the FIRST generated
token, and every candidate cause has been ruled out bit-exactly (probe_rwkv7_prefill_divergence:
attention_mask bool/long, logits_to_keep 0/1; probe_rwkv7_cache_object_cause: cache None /
HF DynamicCache / fla Cache -- all five/three variants returned IDENTICAL logits, max abs diff
0.0000, and none matched generate's emitted token). So "which path is right" cannot be settled
by elimination, and the corrected baseline number depends on the answer.

This settles it WITHOUT needing the mechanism, by decoding a third way that is correct by
construction: at every step, re-run the model over the whole sequence so far from a clean state,
with NO cache at all, and take the last position's logits. That is the definition of
autoregressive greedy decoding. It is O(n^2) and far too slow for a full eval, but for a handful
of rows it is the reference that arbitrates between the loop and generate.

  * if norecache == loop  -> the loop's recurrent cache is correct, generate is broken,
                             and the loop-decoded numbers are the publishable ones.
  * if norecache == generate -> the LOOP is the broken path and the corrected numbers I
                             computed are themselves an artefact. Must not publish them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from falsify_zero_baselines import (  # noqa: E402
    babilong_score, greedy_loop, hf_generate, load_babilong_rows, load_model, truncate_ids,
)

BABI = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/"
            "datasets--RMT-team--babilong/snapshots/"
            "ee0d588794c7ac098062ee0d247c733d62e94fe2/data")
MODELS = {
    "rwkv7": "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/"
             "RWKV7-Goose-World3-2.9B-HF",
    "gla": "/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/"
           "models--fla-hub--gla-1.3B-100B/snapshots/46b15820a4df269e99aed9d709e017677c15d24b",
}


@torch.no_grad()
def greedy_no_cache(model, prefix_ids, max_new_tokens, eos_id=None):
    """Reference decode, correct by construction: no cache, full re-prefill every step."""
    ids: list[int] = []
    for _ in range(max_new_tokens):
        current = prefix_ids if not ids else torch.cat(
            [prefix_ids, torch.tensor([ids], device=prefix_ids.device, dtype=torch.long)], dim=1)
        logits = model(input_ids=current, use_cache=False, return_dict=True).logits[0, -1]
        token = int(logits.argmax())
        if eos_id is not None and token == eos_id:
            break
        ids.append(token)
    return ids


def main():
    n_rows, n_new = 8, 12
    summary = {}
    for name, path in MODELS.items():
        model, tokenizer = load_model(path, "cuda")
        eos = getattr(tokenizer, "eos_token_id", None)
        rows = load_babilong_rows(BABI / "qa1/4k.json")[:n_rows]
        recs, agree = [], {"loop": 0, "generate": 0}
        for row in rows:
            ids = truncate_ids(tokenizer, row["prompt"], 16384, "cuda")
            ref = greedy_no_cache(model, ids, n_new, eos_id=eos)
            loop = greedy_loop(model, ids, n_new, eos_id=eos)
            gen = hf_generate(model, tokenizer, ids, n_new)
            rec = {
                "id": row["id"], "gold": row["answers"][0],
                "ref_ids": ref, "loop_ids": loop, "generate_ids": gen[:n_new],
                "ref": tokenizer.decode(ref, skip_special_tokens=True).strip(),
                "loop": tokenizer.decode(loop, skip_special_tokens=True).strip(),
                "generate": tokenizer.decode(gen[:n_new], skip_special_tokens=True).strip(),
                "ref_eq_loop_first": bool(ref[:1] == loop[:1]),
                "ref_eq_gen_first": bool(ref[:1] == gen[:1]),
                "ref_eq_loop_all": bool(ref == loop),
                "ref_eq_gen_all": bool(ref == gen[:len(ref)]),
            }
            for key, seq in (("ref", ref), ("loop", loop), ("generate", gen[:n_new])):
                rec[f"score_{key}"] = babilong_score(
                    "qa1", tokenizer.decode(seq, skip_special_tokens=True).strip(),
                    row["answers"][0], row["question"])
            agree["loop"] += int(rec["ref_eq_loop_all"])
            agree["generate"] += int(rec["ref_eq_gen_all"])
            recs.append(rec)
            print(f"[{name}] id={rec['id']} ref_eq_loop={rec['ref_eq_loop_all']} "
                  f"ref_eq_gen={rec['ref_eq_gen_all']}", flush=True)
            print(f"    ref  = {rec['ref'][:90]!r}", flush=True)
            print(f"    loop = {rec['loop'][:90]!r}", flush=True)
            print(f"    gen  = {rec['generate'][:90]!r}", flush=True)
        summary[name] = {
            "n": len(recs), "full_seq_agreement": agree,
            "first_token_agreement": {
                "loop": sum(r["ref_eq_loop_first"] for r in recs),
                "generate": sum(r["ref_eq_gen_first"] for r in recs)},
            "scores": {k: sum(r[f"score_{k}"] for r in recs) / len(recs) * 100
                       for k in ("ref", "loop", "generate")},
            "rows": recs,
        }
        print(f"\n[{name}] SUMMARY {json.dumps({k: v for k, v in summary[name].items() if k != 'rows'})}\n",
              flush=True)
        del model
        torch.cuda.empty_cache()

    print("=== VERDICT (which decode matches the no-cache reference?) ===", flush=True)
    for name, s in summary.items():
        print(f"  {name}: full-sequence match loop={s['full_seq_agreement']['loop']}/{s['n']} "
              f"generate={s['full_seq_agreement']['generate']}/{s['n']}; "
              f"scores ref={s['scores']['ref']:.1f} loop={s['scores']['loop']:.1f} "
              f"generate={s['scores']['generate']:.1f}", flush=True)
    out = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/"
               "results/benchmarks/falsify_zero/decode-ground-truth.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pinpoint the line where HF ``generate`` and the repo's greedy loop diverge.

Both paths use the SAME bit-identical weights, the SAME prompt and greedy
decoding, yet share 0/200 generations. ``--mode trace`` in
align_rwkv_two_harnesses.py already proved the loader dtype (fp32 vs bf16) is
NOT the cause: 32/32 tokens agree across dtypes.

So the divergence lives inside ``model.generate``. This probe instruments the
backbone's ``forward`` and records, for every call generate makes:

    * the input_ids actually fed (length + the ids themselves)
    * whether an attention_mask was passed and its length
    * the recurrent cache's reported sequence length before the call

then runs the repo's hand-written loop under the same instrumentation and diffs
the two call logs. If generate re-feeds the prompt (because fla's recurrent
Cache does not report a usable ``get_seq_length`` and generate's
cache_position slicing therefore mis-slices), it shows up here as a second
full-length call instead of a length-1 call.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

HF_DIR = "/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/laces-reasoning/models/RWKV7-Goose-World3-2.9B-HF"
TASK_DIR = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/baseline/Cola-DLM/generate_task_data"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.eval.eval_generic_gen_match import build_prompt

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(HF_DIR, trust_remote_code=True, local_files_only=True)
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = getattr(tok, "eos_token", "") or tok.unk_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            HF_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True
        )
        .to(dev)
        .eval()
    )

    item = json.loads(open(os.path.join(TASK_DIR, "arc_easy.jsonl")).readline())
    prompt = build_prompt("arc_easy", item)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    print(f"[probe] prompt_tokens={ids.shape[1]}")
    print(f"[probe] prompt={prompt!r}\n")

    print(f"[probe] tokenizer.eos_token_id      = {tok.eos_token_id}")
    print(f"[probe] tokenizer.pad_token_id      = {tok.pad_token_id}")
    print(f"[probe] 'pad or eos' as coded       = {tok.pad_token_id or tok.eos_token_id}  <-- 0 is falsy!")
    print(f"[probe] generation_config.eos       = {model.generation_config.eos_token_id}")
    print(f"[probe] generation_config.pad       = {model.generation_config.pad_token_id}")
    print(f"[probe] generation_config.do_sample = {model.generation_config.do_sample}\n")

    calls: list[dict] = []
    orig = type(model).forward

    def patched(self, *a, **kw):
        iid = kw.get("input_ids")
        if iid is None and a:
            iid = a[0]
        am = kw.get("attention_mask")
        pkv = kw.get("past_key_values")
        seqlen = None
        if pkv is not None:
            try:
                seqlen = int(pkv.get_seq_length())
            except Exception as e:
                seqlen = f"ERR:{type(e).__name__}"
        calls.append(
            {
                "in_len": None if iid is None else int(iid.shape[1]),
                "ids": None if iid is None else iid[0].tolist(),
                "mask_len": None if am is None else int(am.shape[1]),
                "cache_seq_len": seqlen,
                "cache_position": None
                if kw.get("cache_position") is None
                else kw["cache_position"].tolist(),
            }
        )
        return orig(self, *a, **kw)

    # ---------------- path A: HF generate
    type(model).forward = patched
    calls.clear()
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=32,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen_calls = list(calls)
    hf_ids = out[0, ids.shape[1]:].tolist()
    type(model).forward = orig

    print("===== PATH A: model.generate() -- per-call log")
    print(f"total forward calls = {len(gen_calls)} (expect 1 prefill + 31 single-token steps = 32)")
    for i, c in enumerate(gen_calls[:8]):
        print(f"  call {i:2d} in_len={c['in_len']:4d} mask_len={c['mask_len']} "
              f"cache_seq_len={c['cache_seq_len']} cache_position={str(c['cache_position'])[:40]}")
        if c["in_len"] and c["in_len"] > 1:
            print(f"          ids={c['ids'][:40]}")
    print(f"  ... [{len(gen_calls)} calls total]")
    long_calls = [i for i, c in enumerate(gen_calls) if (c["in_len"] or 0) > 1]
    print(f"  calls feeding MORE THAN ONE token: {long_calls}")
    print(f"  generated ids  = {hf_ids}")
    print(f"  generated text = {tok.decode(hf_ids, skip_special_tokens=True)!r}\n")

    # ---------------- path B: repo loop
    def rep_pen(logits, generated_ids, penalty):
        if penalty == 1.0 or not generated_ids:
            return logits
        t = torch.tensor(list(set(generated_ids)), device=logits.device, dtype=torch.long)
        t = t.clamp(0, logits.size(-1) - 1)
        ts = logits[t]
        logits = logits.clone()
        logits[t] = torch.where(ts < 0, ts * penalty, ts / penalty)
        return logits

    type(model).forward = patched
    calls.clear()
    with torch.no_grad():
        o = model(input_ids=ids, use_cache=True, return_dict=True)
        cache = o.past_key_values
        all_ids = list(ids[0].tolist())
        new_ids: list[int] = []
        logits = o.logits[0, -1]
        eos = tok.eos_token_id
        for _ in range(32):
            lg = rep_pen(logits.float(), all_ids, 1.1)
            nid = int(torch.argmax(lg).item())
            if eos is not None and nid == eos:
                print(f"  [loop] STOPPED on tokenizer.eos_token_id={eos}")
                break
            new_ids.append(nid)
            all_ids.append(nid)
            o = model(
                input_ids=torch.tensor([[nid]], device=dev),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = o.past_key_values
            logits = o.logits[0, -1]
    loop_calls = list(calls)
    type(model).forward = orig

    print("===== PATH B: repo hand-written loop -- per-call log")
    print(f"total forward calls = {len(loop_calls)}")
    for i, c in enumerate(loop_calls[:8]):
        print(f"  call {i:2d} in_len={c['in_len']:4d} mask_len={c['mask_len']} "
              f"cache_seq_len={c['cache_seq_len']}")
    long_calls_b = [i for i, c in enumerate(loop_calls) if (c["in_len"] or 0) > 1]
    print(f"  calls feeding MORE THAN ONE token: {long_calls_b}")
    print(f"  generated ids  = {new_ids}")
    print(f"  generated text = {tok.decode(new_ids, skip_special_tokens=True).strip()!r}\n")

    print("===== DIFF")
    print(f"  A n_calls={len(gen_calls)}  B n_calls={len(loop_calls)}")
    print(f"  A multi-token calls={long_calls}")
    print(f"  B multi-token calls={long_calls_b}")
    print(f"  A passed attention_mask on every call: {all(c['mask_len'] is not None for c in gen_calls)}")
    print(f"  B passed attention_mask on every call: {all(c['mask_len'] is not None for c in loop_calls)}")
    n = min(len(hf_ids), len(new_ids))
    first = next((i for i in range(n) if hf_ids[i] != new_ids[i]), None)
    print(f"  first differing generated token index: {first}")
    if first is not None:
        print(f"    A[{first}]={hf_ids[first]} ({tok.decode([hf_ids[first]])!r})")
        print(f"    B[{first}]={new_ids[first]} ({tok.decode([new_ids[first]])!r})")

    # Does passing an all-ones attention_mask alone change the logits?
    print("\n===== CONTROL: is an all-ones attention_mask a numerical no-op?")
    with torch.no_grad():
        a = model(input_ids=ids, use_cache=True, return_dict=True).logits[0, -1].float()
        b = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
                  return_dict=True).logits[0, -1].float()
    print(f"  prefill logits maxdiff(no-mask vs ones-mask) = {(a-b).abs().max().item():.6e}")
    print(f"  argmax equal = {int(a.argmax())==int(b.argmax())}  ({int(a.argmax())} vs {int(b.argmax())})")


if __name__ == "__main__":
    main()

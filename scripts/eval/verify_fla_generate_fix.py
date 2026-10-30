#!/usr/bin/env python3
"""Confirm the fla/transformers cache_position incompatibility IS the bug, by
patching prepare_inputs_for_generation to slice to the last token and checking
that model.generate then reproduces the repo loop token-for-token."""
import json, os, sys, torch
from pathlib import Path
REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"/"eval"))
HF_DIR = "/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/laces-reasoning/models/RWKV7-Goose-World3-2.9B-HF"
TASK_DIR = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/baseline/Cola-DLM/generate_task_data"
TASKS = ["piqa","arc_easy","arc_challenge","winogrande","sciq","commonsenseqa"]

def rep_pen(logits, gen_ids, penalty):
    if penalty == 1.0 or not gen_ids: return logits
    t = torch.tensor(list(set(gen_ids)), device=logits.device, dtype=torch.long).clamp(0, logits.size(-1)-1)
    ts = logits[t]; logits = logits.clone()
    logits[t] = torch.where(ts < 0, ts*penalty, ts/penalty)
    return logits

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.eval.eval_generic_gen_match import build_prompt
    dev="cuda"
    tok = AutoTokenizer.from_pretrained(HF_DIR, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(HF_DIR, trust_remote_code=True,
        torch_dtype=torch.bfloat16, local_files_only=True).to(dev).eval()

    # --- THE FIX: for a recurrent cache, only ever feed the newest token.
    cls = type(model)
    orig_prep = cls.prepare_inputs_for_generation
    # NOTE: the **kwargs parameter MUST be named `kwargs` -- transformers'
    # _validate_model_kwargs does `if "kwargs" in model_args: model_args |=
    # signature(self.forward)`, so naming it `kw` makes generate reject
    # attention_mask. functools.wraps also keeps the introspected signature.
    import functools
    @functools.wraps(orig_prep)
    def fixed_prep(self, input_ids=None, past_key_values=None, **kwargs):
        mi = orig_prep(self, input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        if past_key_values is not None:
            try: seen = int(past_key_values.get_seq_length())
            except Exception: seen = 0
            if seen > 0 and mi.get("input_ids") is not None and mi["input_ids"].shape[1] > 1:
                mi["input_ids"] = mi["input_ids"][:, -1:].contiguous()
                mi["cache_position"] = None
        return mi
    cls.prepare_inputs_for_generation = fixed_prep

    N = int(os.environ.get("N", "40"))
    grand_same = grand_tot = 0
    for task in TASKS:
        src = os.path.join(TASK_DIR, f"{task}.jsonl")
        if not os.path.exists(src): continue
        same = tot = 0
        with open(src) as f:
            for i, line in enumerate(f):
                if i >= N: break
                line = line.strip()
                if not line: continue
                item = json.loads(line)
                ids = tok(build_prompt(task, item), return_tensors="pt",
                          add_special_tokens=False).input_ids.to(dev)
                with torch.no_grad():
                    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=32, do_sample=False, repetition_penalty=1.1,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
                hf = out[0, ids.shape[1]:].tolist()
                # repo loop, same weights
                with torch.no_grad():
                    o = model(input_ids=ids, use_cache=True, return_dict=True)
                    cache=o.past_key_values; all_ids=list(ids[0].tolist()); new=[]
                    lg=o.logits[0,-1]; eos=tok.eos_token_id
                    for _ in range(32):
                        l2 = rep_pen(lg.float(), all_ids, 1.1)
                        nid = int(torch.argmax(l2).item())
                        if eos is not None and nid == eos: break
                        new.append(nid); all_ids.append(nid)
                        o = model(input_ids=torch.tensor([[nid]],device=dev),
                                  past_key_values=cache, use_cache=True, return_dict=True)
                        cache=o.past_key_values; lg=o.logits[0,-1]
                # compare on the loop's length (loop may stop early on '\n\n')
                k = len(new)
                tot += 1; same += (hf[:k] == new)
        grand_same += same; grand_tot += tot
        print(f"[{task}] generate==loop (prefix-of-len-{'k'}) {same}/{tot}", flush=True)
    print(f"\n[FIXED] TOTAL identical = {grand_same}/{grand_tot} "
          f"({100.0*grand_same/max(grand_tot,1):.1f}%)")
if __name__ == "__main__":
    main()

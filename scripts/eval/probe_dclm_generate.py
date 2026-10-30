#!/usr/bin/env python3
"""Is `OpenLMForCausalLM.generate()` decoding correctly, or silently re-prefilling?

WHY THIS EXISTS. DCLM-1B loads through mlfoundations' `open_lm`, written against transformers
4.40.2, under transformers 5.3.0. The same version gap already produced a *silent* wrong-result
bug in this repo for fla's recurrent models: `_cache_dependant_input_preparation` was deleted and
`cache_position` changed meaning, so the model's own guard never fired and **every decode step
re-fed the entire sequence**. Nothing crashed; scores were just wrong. The fingerprint was "first
generated token 100% identical, second onward diverges".

open_lm is at even higher risk than fla, because its cache is not a `Cache` object at all: it is a
plain list of per-layer `[keys, vals]` tensors grown by `torch.cat` (open_lm/model.py:172-177),
and its `prepare_inputs_for_generation` slices `input_ids` using `past_key_values[0][0].shape[1]`.
If that length comes back wrong, or the cache silently arrives as None each step, generation still
"works" and still returns plausible text -- at the wrong positions, with the wrong RoPE offsets.

GROUND TRUTH used here is the cache-free loop: re-forward the WHOLE sequence with
`use_cache=False` and take the argmax, every step. That path cannot be fooled by any cache bug,
because there is no cache. Greedy decoding is deterministic, so a correct `generate()` must agree
with it token for token (up to fp non-determinism, which shows up as late, rare disagreements --
not as an immediate divergence at index 1).

A THIRD arm, `--arm loop`, runs the harness's own `greedy_loop()` from
run_open_context_bench.py, so the same comparison also certifies the `--decode loop` path that
the long-context suite can be asked to use.

Run on the LOGIN NODE with --device cpu (no GPU there) or in a job with --device cuda. CPU fp32 is
the stricter test anyway: bf16 on GPU introduces its own tie-breaking noise.
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

# Deliberately mixed: a factual continuation, a QA-formatted prompt like the capability suite
# feeds, and a pattern the model must keep counting -- a position/RoPE bug wrecks the third one
# first, since it is the only prompt whose correct continuation depends on absolute order.
PROMPTS = [
    "The capital of France is",
    "Question: What color is the sky on a clear day?\nAnswer:",
    "1, 2, 3, 4, 5, 6,",
    "Once upon a time, in a small village near the sea, there lived a fisherman who",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    import open_lm.hf  # noqa: F401  registers OpenLMConfig / OpenLMForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True,
                                       local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, trust_remote_code=True, dtype=dt, local_files_only=True
    ).to(a.device).eval()
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_open_context_bench import greedy_loop

    print(f"[probe] {type(model).__name__} device={a.device} dtype={a.dtype} "
          f"n={a.max_new_tokens}", flush=True)
    print(f"[probe] generate exists = {hasattr(model, 'generate')}; "
          f"_supports_default_dynamic_cache = {model._supports_default_dynamic_cache()}",
          flush=True)

    rows, all_ok = [], True
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(a.device)
        # (1) ground truth: no cache anywhere, full re-forward every step.
        ref = ids.clone()
        with torch.no_grad():
            for _ in range(a.max_new_tokens):
                lg = model(input_ids=ref, attention_mask=torch.ones_like(ref),
                           use_cache=False, return_dict=True).logits[0, -1]
                ref = torch.cat([ref, lg.argmax().view(1, 1)], dim=1)
        ref_ids = ref[0, ids.shape[1]:].tolist()

        # (2) what eval_generic_gen_match.py actually calls. No repetition_penalty, so the
        #     comparison against pure greedy is apples to apples.
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen_ids = out[0, ids.shape[1]:].tolist()

        # (3) the long-context harness's own --decode loop path (cache threaded by hand).
        with torch.no_grad():
            loop_ids = greedy_loop(model, ids, a.max_new_tokens, eos_id=None)

        def cmp(x, y):
            n = min(len(x), len(y))
            agree = sum(1 for i in range(n) if x[i] == y[i])
            first = next((i for i in range(n) if x[i] != y[i]), None)
            return agree, n, first

        g_agree, g_n, g_first = cmp(ref_ids, gen_ids)
        l_agree, l_n, l_first = cmp(ref_ids, loop_ids)
        ok = (g_first is None) and (l_first is None)
        all_ok &= ok
        print(f"\nprompt: {prompt!r}")
        print(f"  ref (no cache)  : {tok.decode(ref_ids)!r}")
        print(f"  generate()      : {tok.decode(gen_ids)!r}")
        print(f"  greedy_loop()   : {tok.decode(loop_ids)!r}")
        print(f"  generate vs ref : {g_agree}/{g_n} agree, first divergence {g_first}")
        print(f"  loop     vs ref : {l_agree}/{l_n} agree, first divergence {l_first}")
        if g_first == 1 or l_first == 1:
            print("  *** FINGERPRINT OF THE CACHE/RE-PREFILL BUG: token 0 matches, token 1 "
                  "diverges -- do NOT run the suite on this path ***")
        rows.append({"prompt": prompt, "ref_ids": ref_ids, "generate_ids": gen_ids,
                     "loop_ids": loop_ids,
                     "generate_agree": g_agree, "generate_first_divergence": g_first,
                     "loop_agree": l_agree, "loop_first_divergence": l_first, "ok": ok})

    print("\n" + "=" * 78)
    print(f"VERDICT: {'PASS -- generate() and greedy_loop() both reproduce the cache-free '
                      'reference exactly' if all_ok else 'FAIL -- at least one decode path '
                      'disagrees with the cache-free reference'}")
    print("=" * 78)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(
            {"model_dir": a.model_dir, "device": a.device, "dtype": a.dtype,
             "max_new_tokens": a.max_new_tokens, "all_ok": all_ok, "rows": rows}, indent=2))
        print(f"[probe] wrote {a.output}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

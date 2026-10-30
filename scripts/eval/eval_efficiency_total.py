#!/usr/bin/env python3
"""Per-batch-size decode timing AND per-batch-size peak memory, for a total-time column.

WHY THIS EXISTS AND WHY IT IS A SEPARATE FILE FROM eval_efficiency.py.
The published Table `tab:resources` needs three quantities that no file under
results/benchmarks/efficiency/ carries:

  (1) a per-sequence throughput at the concurrency the paper's claim rests on. The
      published Chunked ms/tok cell is 5.2, which reverse-maps exactly to
      `decode_ms_tok_bs8 / 16` (82.55/16 = 5.1594) while the other eight rows print
      `decode_ms_tok_bs1`. Those are different quantities: time_decode() divides by
      `steps` only, never by batch, so a bs8 figure is the wall-clock per decode STEP
      with 8 sequences resident, not the per-token cost of one sequence.
  (2) a batch-16 measurement at all. 16 = 512/32 concurrent segments is the paper's
      own divisor, but eval_efficiency.py hard-codes `for bs in (1, 8)` and 0 of the
      25 result files carry any bs16 key. The 16-way concurrency is asserted, never timed.
      It is not a safe extrapolation either: measured bs8 is 1.11-1.15x WORSE per step
      than bs1 on the three rows that share the frozen 2.9B backbone.
  (3) peak memory attributable to a specific batch size. eval_efficiency.py calls
      reset_peak_memory_stats() at :409 and reads max_memory_allocated() at :454 with
      `for bs in (1, 8)` in between; empty_cache() does not reset the peak counter, so
      the single published Peak GB is the max over both batches. Every row was measured
      that way, so the column is internally consistent, but no row's memory can be
      attributed to the 16-way parallel decode.

Kept separate because eval_efficiency.py produced the nine published rows. Editing it
would make those rows irreproducible, and it is on the same tree as jobs running now.

WHAT IS MEASURED. For each requested batch size, independently and in this order:
  reset peak counter -> warm up -> timed repeats -> read peak.
So `peak_mem_gb_bs{B}` belongs to that batch size alone. Both a per-step and a
per-sequence-token figure are recorded, with the arithmetic printed, so the two can
never be confused again:
  decode_ms_step_bs{B}    wall-clock ms per decode step with B sequences resident
  decode_ms_tok_seq_bs{B} = decode_ms_step_bs{B} / B, the amortised per-token cost
                            of one sequence when B render concurrently
A total time for an N-token completion then follows from a SINGLE measured batch:
  total_s = plan_s + N * decode_ms_tok_seq_bs{B} / 1000
and `gen_len` is recorded explicitly, because "how many tokens" is part of the claim:
a total-time column is meaningless without it.

WHY THE NUMBERS HERE DISAGREE WITH THE PUBLISHED ONES BY 3.1x ON FOUR ROWS.
The first sweep measured llama-3.2-3b at 17.11 ms/step against the published 53.87. That
gap is not this script: the timed loop is byte-identical to eval_efficiency.py's. It is a
cold-vs-warm split between two measurement eras in the published table, identified by a
conjunction of two independent properties rather than by either alone:

  (a) the file lacks every `_cold` key, which the warmed-up eval_efficiency.py ALWAYS
      writes, and
  (b) its mtime is 09-20 17:2x, before the warm-up patch, whereas the four rows that do
      carry `_cold` are all 09-21 03:49 or later.

Both properties agree on the same partition, 5 files against 4. Confirmed by direct
measurement: for the five pre-patch rows the published number matches this script's COLD
figure (ratios 1.13-1.34x) and not its warm one (0.32-0.33x), while for the post-patch rows
it matches the WARM figure (0.99-1.02x). So four published transformer rows carry one
un-warmed generate() call each, absorbing triton JIT compilation.

DIRECTION. All four inflated rows are BASELINES, so the artefact favours us: it makes the
AR competition look 3.1x slower per token than it is. Mamba is nearly unaffected (1.02x)
because it JITs during load rather than on first decode, which is also why the artefact
was not visible as a uniform offset across the pre-patch group.
"""
import argparse
import json
import time
from pathlib import Path

import torch


def timed_decode(model, ids, mask, steps, warmup, n):
    """Returns (median_ms_per_step, cold_first_ms_per_step). Divides by `steps` only."""
    cold = None
    for _ in range(max(warmup, 0)):
        torch.cuda.synchronize()
        t0 = time.time()
        model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=steps,
                       do_sample=False, pad_token_id=0)
        torch.cuda.synchronize()
        if cold is None:
            cold = (time.time() - t0) * 1000.0 / steps
    runs = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.time()
        model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=steps,
                       do_sample=False, pad_token_id=0)
        torch.cuda.synchronize()
        runs.append((time.time() - t0) * 1000.0 / steps)
    runs.sort()
    return runs[len(runs) // 2], (cold if cold is not None else runs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=["relay", "hf"])
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--len", type=int, default=512, help="prompt length")
    ap.add_argument("--steps", type=int, default=64, help="decode steps per timed call")
    ap.add_argument("--gen_len", type=int, default=512,
                    help="completion length the reported total time is FOR")
    ap.add_argument("--batches", default="1,8,16",
                    help="batch sizes to measure independently")
    ap.add_argument("--plan_steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    device = torch.device(a.device)
    res = {"impl": a.impl, "input_len": a.len, "decode_steps_timed": a.steps,
           "gen_len": a.gen_len, "warmup": a.warmup, "repeats": a.repeats,
           "per_seq_formula": "decode_ms_tok_seq_bs{B} = decode_ms_step_bs{B} / B",
           "total_formula": "total_s = plan_s + gen_len * decode_ms_tok_seq_bs{B} / 1000"}

    t0 = time.time()
    if a.impl == "relay":
        from scripts.eval.relay_utils import load_relay_model
        model, _rwkv, tok, _ck, _cfg = load_relay_model(a.ckpt_dir, device)
        backbone = model.rwkv_model
        model.eval()
        model._prefix_suffix_s2 = True
        dtype = next(model.latent_dit.parameters()).dtype
        res["ckpt"] = a.ckpt_dir
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # trust_remote_code forced on for the same reason eval_efficiency.py does it:
        # RWKV-6/7 ship custom modelling code in auto_map and abort without it.
        tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True,
                                           local_files_only=True)
        kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16,
                  local_files_only=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(a.model_dir, **kw)
        except ValueError as e:
            if "does not recognize this architecture" not in str(e):
                raise
            arch = (json.load(open(Path(a.model_dir) / "config.json"))
                    .get("architectures") or [None])[0]
            import fla.models as _fm  # only here: it also re-registers `mamba`
            model = (getattr(_fm, arch).from_pretrained(a.model_dir, **kw)
                     if arch and hasattr(_fm, arch)
                     else AutoModelForCausalLM.from_pretrained(a.model_dir, **kw))
        model = model.to(device).eval()
        backbone = model
        dtype = torch.bfloat16
        res["model_dir"] = a.model_dir
    res["load_s"] = round(time.time() - t0, 1)

    # a real token sequence: random ids fall outside some vocabs and trigger device asserts
    filler = ("The recurrent state carries the evidence forward and the decoder renders "
              "from it. " * 200)
    enc = tok(filler, return_tensors="pt", add_special_tokens=False).input_ids
    while enc.shape[1] < a.len:
        enc = torch.cat([enc, enc], dim=1)
    enc = enc[:, :a.len]

    for bs in [int(b) for b in a.batches.split(",") if b]:
        ids = enc.expand(bs, -1).contiguous().to(device)
        mask = torch.ones_like(ids)
        # per-batch isolation: reset the peak counter INSIDE the loop so the number
        # that comes out belongs to this batch size and not to the largest one tried.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            step, cold = timed_decode(backbone, ids, mask, a.steps, a.warmup, a.repeats)
        except torch.cuda.OutOfMemoryError:
            # record the OOM rather than dropping the batch size silently
            res[f"decode_ms_step_bs{bs}"] = None
            res[f"oom_bs{bs}"] = True
            del ids, mask
            torch.cuda.empty_cache()
            continue
        res[f"decode_ms_step_bs{bs}"] = round(step, 2)
        res[f"decode_ms_step_bs{bs}_cold"] = round(cold, 2)
        res[f"decode_ms_tok_seq_bs{bs}"] = round(step / bs, 4)
        res[f"tok_per_s_seq_bs{bs}"] = round(1000.0 * bs / step, 2)
        res[f"tok_per_s_aggregate_bs{bs}"] = round(1000.0 * bs / step, 2)
        res[f"peak_mem_gb_bs{bs}"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        del ids, mask
        torch.cuda.empty_cache()

    if a.impl == "relay":
        from scripts.eval.sample_prefix_suffix_cfg import encode_prefix, sample_ddim_cfg
        ids = enc[:, : a.len // 2].to(device)
        mask = torch.ones_like(ids)
        encode_prefix(model, ids, mask)  # warm up before timing the plan
        torch.cuda.synchronize()
        t0 = time.time()
        cond = encode_prefix(model, ids, mask)
        z = sample_ddim_cfg(model, cond, a.plan_steps, a.cfg_scale, device, dtype)
        states = model.predict_states(z)
        model.inject_into_cache(
            backbone(input_ids=ids, attention_mask=mask.bool(),
                     use_cache=True, return_dict=True).past_key_values, states)
        torch.cuda.synchronize()
        res["plan_ms"] = round((time.time() - t0) * 1000.0, 1)
        res["plan_steps"] = a.plan_steps
        res["cfg_scale"] = a.cfg_scale

    # totals for the requested completion length, one per measured batch size
    plan_s = res.get("plan_ms", 0.0) / 1000.0
    res["plan_s"] = round(plan_s, 3)
    for bs in [int(b) for b in a.batches.split(",") if b]:
        k = f"decode_ms_tok_seq_bs{bs}"
        if res.get(k) is None:
            continue
        res[f"total_s_gen{a.gen_len}_bs{bs}"] = round(plan_s + a.gen_len * res[k] / 1000.0, 2)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()

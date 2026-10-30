#!/usr/bin/env python3
"""Allocate the paper's baseline benchmark cells onto GPUs with a per-GPU wall-clock cap.

Requirement comes from the PAPER, not from a wish list: the rows are the baseline models in
main.tex (tab:capability / tab:capability-lang / tab:longcontext / tab:longcontext-suites),
18 baseline models in four groups.

Cost model is anchored on MEASURED QZ run_h values wherever one exists; every other number is
marked EST and must be replaced by a pilot before it is trusted (see --show-costs).

  measured:
    fix2 6 tasks x 200 samples        = 1.07-1.17 h  -> 0.19 h/task  (laces-fix2-*)
    newtasks 6 tasks                  = 0.83 h       -> 0.14 h/task  (laces-newtasks-*)
    wave1topup 1 task (hellaswag)     = 0.30 h       -> 0.30 h/task
    benchmark RACE+bAbILong ~9 units  = 3.28 h       -> 0.36 h/unit  (laces-benchmark-*)
    efficiency                        = 0.05 h
    ppl (7 datasets)                  = 0.19 h
  EST (never run, must pilot): RULER / LB-QA / infinityBench / MemEval

Usage:
  python3 plan_baseline_evals.py                      # plan + per-GPU table
  python3 plan_baseline_evals.py --gpus 50 --cap-h 12
  python3 plan_baseline_evals.py --show-costs
  python3 plan_baseline_evals.py --json out.json      # machine-readable assignment
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

# --- baselines required by main.tex, grouped exactly as the paper groups them -----
BASELINES = {
    "AR Transformer": ["Qwen2.5-3B", "Qwen2.5-1.5B", "Llama-3.2-3B", "Phi-2", "Gemma-2-2B"],
    "linear-attn / SSM": ["RWKV-7 World3", "RWKV-6 World2.1", "Mamba-2"],
    "discrete diffusion LM": ["LLaDA-8B-Base", "Dream-v0-Base", "LLaDA-MoE-7B-A1B",
                              "OpenDFM CharTool", "MDLM-OWT", "SEDD-medium"],
    "continuous latent diffusion": ["CoLA-DLM", "ELF-L", "ELF-M", "STAR-LDM"],
}
ALL_BASELINES = [m for g in BASELINES.values() for m in g]

# --- work units, with measured-or-estimated GPU-hours per model -------------------
# each entry: (dataset/protocol, hours per model, measured?, can be sharded?)
UNITS = [
    # short-context suite: 11 capability columns + 3 language columns = 14 tasks
    ("MMLU",      0.19, "meas", True),  ("OBQA",     0.19, "meas", True),
    ("SIQA",      0.19, "meas", True),  ("StoryC.",  0.19, "meas", True),
    ("Hella.",    0.30, "meas", True),  ("PIQA",     0.14, "meas", True),
    ("ARC-e",     0.14, "meas", True),  ("ARC-c",    0.14, "meas", True),
    ("WinoG.",    0.14, "meas", True),  ("SciQ",     0.14, "meas", True),
    ("CQA",       0.14, "meas", True),
    ("LAMBADA",   0.19, "meas", True),  ("SQuAD",    0.19, "meas", True),
    ("GSM8K",     0.60, "est",  True),   # 200-tok generation, own protocol
    # long-context
    ("RACE",      0.36, "meas", True),
    ("bAbI-1 4k", 0.36, "meas", True),  ("bAbI-1 8k", 0.36, "meas", True),
    ("bAbI-1 16k",0.36, "meas", True),  ("bAbI-2 4k", 0.36, "meas", True),
    # long-context suites -- NEVER RUN, estimated only
    ("RULER 4k",  2.00, "EST",  True),  ("RULER 16k", 3.00, "EST",  True),
    ("LB-QA",     2.00, "EST",  True),  ("infBench",  4.00, "EST",  True),
    ("MemEval",   3.00, "EST",  True),
    # side tables
    ("efficiency",0.05, "meas", False), ("ppl(7)",    0.19, "meas", True),
]

# --- gates that stop work regardless of GPU count --------------------------------
GATES = {
    "LLaDA-8B-Base":        "loader failed 11/11 (laces-difflm-llada-8b*)",
    "Dream-v0-Base":        "loader failed 6/6 (laces-difflm-dream-7b*)",
    "LLaDA-MoE-7B-A1B":     "loader failed 5/5 (+ laces-newtasks-llada-moe-7b)",
    "OpenDFM CharTool":     "never attempted; weights unknown",
    "MDLM-OWT":             "1 attempt, failed; weights unknown",
    "SEDD-medium":          "never attempted; weights unknown",
    "CoLA-DLM":             "harness exists (task data under baseline/Cola-DLM) but model never evaluated",
    "ELF-L":                "never attempted; weights unknown",
    "ELF-M":                "never attempted; weights unknown",
    "STAR-LDM":             "never attempted; weights unknown",
    "Qwen2.5-1.5B":         "never attempted; checkpoint presence unverified",
    "RWKV-6 World2.1":      "never attempted; checkpoint presence unverified",
}
NO_HARNESS_UNITS = {"RULER 4k", "RULER 16k", "LB-QA", "infBench", "MemEval"}


def lpt_pack(items, n_bins, cap):
    """Longest-processing-time first; returns bins and any unit that cannot fit."""
    oversized = [it for it in items if it["hours"] > cap]
    bins = [[] for _ in range(n_bins)]
    load = [0.0] * n_bins
    for it in sorted(items, key=lambda x: -x["hours"]):
        if it["hours"] > cap:
            continue
        i = min(range(n_bins), key=lambda k: load[k])
        # keep a whole (model,dataset) unit on one card, as the runner does
        bins[i].append(it)
        load[i] += it["hours"]
    return bins, load, oversized


def wave_schedule(items, n_gpus, cap):
    """One task per GPU at a time; 50 GPUs run 50 tasks concurrently, and a card takes the
    next task as soon as it frees up.

    Fixed-stride assignment over a cost-descending list is what a real dispatch loop
    produces, and it balances per-card totals: card i takes items[i], items[i+n], ...
    Wall clock is the sum over waves of the longest unit in that wave.
    """
    ordered = sorted(items, key=lambda x: -x["hours"])
    waves = [ordered[i:i + n_gpus] for i in range(0, len(ordered), n_gpus)]
    per_card = [[] for _ in range(n_gpus)]
    for idx, it in enumerate(ordered):
        per_card[idx % n_gpus].append(it)
    loads = [sum(x["hours"] for x in c) for c in per_card]
    wave_spans = [max((x["hours"] for x in w), default=0.0) for w in waves]
    return waves, wave_spans, per_card, loads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=50)
    ap.add_argument("--cap-h", type=float, default=12.0)
    ap.add_argument("--mode", choices=["pack", "wave"], default="wave",
                    help="pack = pre-assign chains per GPU; wave = one task per GPU, dispatched")
    ap.add_argument("--models", default="all", help="'all' or comma list; default all baselines")
    ap.add_argument("--skip-gated", action="store_true", help="drop models blocked by GATES")
    ap.add_argument("--skip-unharnessed", action="store_true", help="drop datasets with no harness")
    ap.add_argument("--show-costs", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.show_costs:
        print(f"{'unit':14s} {'h/model':>8s} {'basis':>6s}  shardable")
        for n, h, b, s in UNITS:
            print(f"{n:14s} {h:8.2f} {b:>6s}  {s}")
        tot = sum(h for _, h, _, _ in UNITS)
        print(f"\nper-model total over {len(UNITS)} units: {tot:.2f} h")
        return

    models = ALL_BASELINES if a.models == "all" else [m.strip() for m in a.models.split(",")]
    units = UNITS
    if a.skip_unharnessed:
        units = [u for u in UNITS if u[0] not in NO_HARNESS_UNITS]
    if a.skip_gated:
        models = [m for m in models if m not in GATES]

    items = [{"model": m, "unit": u, "hours": h, "basis": b, "shardable": s}
             for m in models for (u, h, b, s) in units]

    total = sum(i["hours"] for i in items)
    bins, load, oversized = lpt_pack(items, a.gpus, a.cap_h)

    print(f"models={len(models)}  units/model={len(units)}  cells={len(items)}")
    print(f"total work        : {total:.1f} GPU-h")
    print(f"capacity          : {a.gpus} GPUs x {a.cap_h:.0f} h = {a.gpus*a.cap_h:.0f} GPU-h "
          f"({total/(a.gpus*a.cap_h)*100:.0f}% used)")
    print(f"ideal wall-clock  : {total/a.gpus:.2f} h  (perfect packing, no queueing)")

    if a.mode == "wave":
        waves, spans, per_card, loads = wave_schedule(items, a.gpus, a.cap_h)
        print(f"\n--- wave/queue mode: one task per GPU, {a.gpus} tasks in flight ---")
        print(f"waves             : {len(waves)}  (last wave has {len(waves[-1])} tasks)")
        print(f"wall clock        : {sum(spans):.2f} h  = sum of wave spans")
        print(f"per-card total    : max {max(loads):.2f} h   min {min(loads):.2f} h")
        print(f"cap respected     : {'YES' if max(loads) <= a.cap_h + 1e-9 else 'NO -- needs rebalancing'}")
        print(f"\n{'wave':>4} {'tasks':>5} {'span_h':>7} {'cum_h':>7}  longest unit in wave")
        cum = 0.0
        for k, (w, sp) in enumerate(zip(waves, spans), 1):
            cum += sp
            longest = max(w, key=lambda x: x["hours"])
            print(f"{k:>4} {len(w):>5} {sp:>7.2f} {cum:>7.2f}  {longest['model']}/{longest['unit']} "
                  f"({longest['hours']:.2f}h)")
        print(f"\nper-card totals (first 12): "
              + ", ".join(f"g{i}:{loads[i]:.2f}" for i in range(min(12, a.gpus))))
        if a.json:
            json.dump({"mode": "wave", "gpus": a.gpus, "cap_h": a.cap_h,
                       "total_gpu_h": round(total, 2), "wall_h": round(sum(spans), 2),
                       "per_card": [{"gpu": i, "load_h": round(loads[i], 3),
                                     "tasks": per_card[i]} for i in range(a.gpus)],
                       "waves": [{"wave": k + 1, "span_h": round(sp, 3), "tasks": w}
                                 for k, (w, sp) in enumerate(zip(waves, spans))]},
                      open(a.json, "w"), indent=2)
            print(f"assignment -> {a.json}")
        est = sum(i["hours"] for i in items if i["basis"] == "EST")
        if est:
            print(f"\nESTIMATED portion: {est:.1f} GPU-h ({est/max(total,1e-9)*100:.0f}% of total) -- "
                  f"all of it is the never-run long-context suites. Pilot before trusting.")
        return

    used = [l for l in load if l > 0]
    print(f"GPUs actually used: {len(used)}   max bin {max(load):.2f} h   min bin {min(used):.2f} h")
    print(f"cap respected     : {'YES' if max(load) <= a.cap_h + 1e-9 else 'NO'}")
    if oversized:
        print(f"\n!! {len(oversized)} units exceed the cap alone and must be SHARDED first:")
        for o in oversized:
            print(f"   {o['model']} / {o['unit']}: {o['hours']:.1f} h")

    print(f"\n{'gpu':>3} {'load_h':>7} {'n':>3}  first units")
    for i, b in enumerate(bins):
        if not b:
            continue
        head = ", ".join(f"{x['model']}/{x['unit']}" for x in sorted(b, key=lambda x: -x["hours"])[:3])
        print(f"{i:>3} {load[i]:7.2f} {len(b):>3}  {head}")

    est = sum(i["hours"] for i in items if i["basis"] == "EST")
    if est:
        print(f"\nESTIMATED portion: {est:.1f} GPU-h ({est/max(total,1e-9)*100:.0f}% of total) -- "
              f"all of it is the long-context suites that have never been run. Pilot one before trusting.")

    if a.json:
        json.dump({"per_gpu": [{"gpu": i, "load_h": round(load[i], 3),
                                "units": bins[i]} for i in range(a.gpus) if bins[i]],
                   "total_gpu_h": round(total, 2), "gpus": a.gpus, "cap_h": a.cap_h},
                  open(a.json, "w"), indent=2)
        print(f"\nassignment -> {a.json}")


if __name__ == "__main__":
    main()

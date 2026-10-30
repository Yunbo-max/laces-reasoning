#!/usr/bin/env python3
"""Run the paper's baseline evaluations to completion, one task per GPU.

Rows and columns come from main.tex, not from a wish list:
  tab:capability      MMLU OBQA SIQA StoryC Hella PIQA ARC-e ARC-c WinoG SciQ CQA
  tab:capability-lang LAMBADA SQuAD GSM8K
  tab:longcontext     RACE bAbI-1 4k/8k/16k bAbI-2 4k
  tab:longcontext-suites RULER 4k/16k LB-QA infinityBench MemEval   <- no harness yet
  tab:resources       efficiency      tab:lm  ppl(7)

The generation command is copied VERBATIM from the job that actually succeeded
(laces-fix2-baseline-llama-3.2-3b-1gpu-h100-20260918 / laces-wave2-baseline-qwen2.5-3b),
so this reproduces the published protocol: greedy (temperature 0.0), max_new_tokens 32,
one shared answer-matching rule, 200 samples per task.

    --check     say which baselines exist locally and which datasets have task data.
                Compute nodes have no network, so a model that was never downloaded
                cannot run at all -- this is the gate, and it is answered by reading disk.
    --plan      build the (model x dataset) cells and the wave schedule.
    --submit    emit bodies, dry-run each, then submit keeping <= --in-flight jobs alive.
    --status    summarise job outcomes and which cells have landed.

Usage
  python3 run_baseline_evals.py --check
  python3 run_baseline_evals.py --plan --gpus 50
  python3 run_baseline_evals.py --submit --gpus 50 --in-flight 50 --dry-run-bodies
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
DIFFRWKV = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv")
MODELS_ROOT = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/models")
HF_CACHE = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache")
TASK_DATA = DIFFRWKV / "baseline/Cola-DLM/generate_task_data"
OUT_ROOT = REPO / "results/benchmarks"
LOG_DIR = REPO / "logs/qz"
EVIDENCE = DIFFRWKV / ".omo/evidence/qz-training-skill"

WS = "ws-9dcc0e1f-80a4-4af2-bc2f-0e3527b17e6"  # placeholder, overwritten below
WS = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
SPEC_BY_GPU = {1: ("79fe954a-be92-4772-ac0b-94ad8a79b7bb", 100),
               4: ("8a53ac21-299a-4dee-85e9-9c04a544cf8d", 900),
               8: ("7166bd2e-6cbe-4bd9-be38-762d11003e7f", 1800)}

# --- paper rows: baseline name -> candidate local dirs under MODELS_ROOT ----------
BASELINES = {
    "Qwen2.5-3B":              ["Qwen2.5-3B"],
    "Qwen2.5-1.5B":            ['qwen2.5-1.5b', 'Qwen2.5-1.5B'],
    "Llama-3.2-3B":            ["Llama-3.2-3B", "Llama-3.2-3B-Instruct"],
    "Phi-2":                   ["phi-2", "Phi-2", "phi-2-2.7b"],
    "Gemma-2-2B":              ["gemma-2-2b", "gemma-2-2b-it"],
    "RWKV-7 World3":           ["RWKV7-Goose-World3-2.9B-HF"],
    "RWKV-6 World2.1":         ['rwkv6-world-3b', 'RWKV-6-World-2.1B'],
    # mamba2-2.7b has model_type=None (unloadable); mamba-2.8b is the working checkpoint.
    "Mamba-2":                 ['mamba-2.8b', 'mamba2-2.7b'],
    "LLaDA-8B-Base":           ["LLaDA-8B-Base", "GSAI-ML-LLaDA-8B-Base"],
    "Dream-v0-Base":           ['dream-7b-instruct', 'dreamon-7b'],
    "LLaDA-MoE-7B-A1B":        ['llada-moe-7b-a1b', 'LLaDA-MoE-7B-A1B-Instruct'],
    "OpenDFM CharTool":        ['opendfm-chartool-3b', 'OpenDFM-CharTool'],
    "MDLM-OWT":                ['MDLM-owt', 'mdlm-owt-169m'],
    "SEDD-medium":             ['SEDD-medium', 'sedd-medium'],
    # lowercase copy is complete; "Cola-DLM" is TRUNCATED (DiT short by 460 MB,
    # VAE by 24.8 MB) and dies with SafetensorError: MetadataIncompleteBuffer.
    "CoLA-DLM":                ['cola-dlm', 'Cola-DLM'],
    "ELF-L":                   ['ELF-L-owt-torch', 'ELF-L'],
    "ELF-M":                   ['ELF-M-owt-torch', 'ELF-M-modelscope'],
    "STAR-LDM":                ["STAR-LDM", "star-ldm"],    "GLA-1.3B":               ["fla-hub--gla-1.3B-100B", "gla-1.3B-100B"],
}
# --- our own two models for the same table (relay checkpoints, different harness) ---
# Verified 2026-09-20: champ-dynamic is trajectory_mode=True (uses the trajectory evaluator),
# global-dit-50k is trajectory_mode=False (uses the single-z prefix_cfg evaluator). Running
# either through the other arm yields garbage -- measured: champ via prefix_cfg scored 1.00%
# with 24% degenerate output while its own arm scored 24.67%.
RELAY_MODELS = {
    "champ-dynamic-30k": dict(
        ckpt=str(REPO / "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000"),
        script="run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py", method="traj_cfg"),
    "global-dit-50k": dict(
        ckpt=str(REPO / "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000"),
        script="run_cola_dlm_tasks_prefix_suffix_cfg.py", method="prefix_cfg"),
    # Trained today (job-4d3e47e1, 9h08m, diff 1.26 -> 0.049). Fixed basis + DiT denoiser,
    # trajectory_mode=False so it takes the single-z arm -- the same internally consistent
    # pairing as global-dit-50k. Completes the 2x2 {dynlowrank,fixed} x {dit,birwkv} matrix.
    # 13.3B scale-up, stopped at log-step ~18k; the newest SAVED checkpoint is 15k
    # (save_every_n_steps=5000 -- the 15k->18k steps were never written to disk).
    "13.3B-chunked": dict(
        ckpt=str(REPO / "outputs_dynamic_basis/laces-13.3B-dynlowrank-r32-chunked-birwkv-50000/step_00015000"),
        script="run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py", method="traj_cfg"),
    "13.3B-global": dict(
        ckpt=str(REPO / "outputs_dynamic_basis/laces-13.3B-dynlowrank-r32-global-birwkv-50000/step_00015000"),
        script="run_cola_dlm_tasks_prefix_suffix_cfg.py", method="prefix_cfg"),
    "dit-fixed-50k": dict(
        ckpt=str(REPO / "outputs_dynamic_basis/laces-2.9B-fixed-r32-global-dit768x8-b16-50k/step_00050000"),
        script="run_cola_dlm_tasks_prefix_suffix_cfg.py", method="prefix_cfg"),
}
RELAY_OUT = OUT_ROOT / "evalset"

GROUP = {  # only used for reporting, matches the paper's grouping
    "AR Transformer": ["Qwen2.5-3B", "Qwen2.5-1.5B", "Llama-3.2-3B", "Phi-2", "Gemma-2-2B"],
    "linear-attn/SSM": ["GLA-1.3B", "RWKV-7 World3", "RWKV-6 World2.1", "Mamba-2"],
    "discrete diffusion": ["LLaDA-8B-Base", "Dream-v0-Base", "LLaDA-MoE-7B-A1B",
                           "OpenDFM CharTool", "MDLM-OWT", "SEDD-medium"],
    "continuous latent diffusion": ["CoLA-DLM", "ELF-L", "ELF-M", "STAR-LDM"],
}

# --- paper columns -> task name in the shared generate task data ------------------
# cost in GPU-hours comes from measured QZ run_h (see plan_baseline_evals.py --show-costs)
GEN_TASKS = {  # short-context suite, all through eval_generic_gen_match.py
    "MMLU": ("mmlu", 0.19), "OBQA": ("obqa", 0.19), "SIQA": ("siqa", 0.19),
    "StoryC.": ("story_cloze", 0.19), "Hella.": ("hellaswag", 0.30),
    "PIQA": ("piqa", 0.14), "ARC-e": ("arc_easy", 0.14), "ARC-c": ("arc_challenge", 0.14),
    "WinoG.": ("winogrande", 0.14), "SciQ": ("sciq", 0.14), "CQA": ("commonsenseqa", 0.14),
    "LAMBADA": ("lambada", 0.19), "SQuAD": ("squad", 0.19),
}
# RACE lives in the same task dir; bAbILong has its own harness; the five suites have none.
RACE_TASK = ("race", 0.36)
PPL_CORPORA = "wikitext2,wikitext103,pg19,lm1b,ptb,wmt14,xsum"
# MDLM/SEDD/ELF ship a GPT-2 BPE tokenizer whose fast conversion needs `tiktoken`, which is not
# installed here and cannot be pip-installed on the network-less compute nodes. The tokenizer
# itself is plain GPT-2, and a local copy is on disk, so point at that instead.
G2 = "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/gpt2-medium"
TOK_OVERRIDE = {"MDLM-OWT": G2, "SEDD-medium": G2}
# ELF is deliberately NOT here: its unembed head is 512x32100 = T5 sentencepiece,
# not GPT-2 BPE. Forcing gpt2-medium on it is wrong.
# which diffusion denoising loop each non-causal-LM baseline needs
DIFFUSION_STEPS = {"MDLM-OWT": 128, "SEDD-medium": 128}
# ELF = continuous flow matching over frozen T5-small embeddings; has its own adapter.
ELF_RUNNER = {"ELF-L", "ELF-M"}
DIFFUSION_IMPL = {"MDLM-OWT": "mdlm", "SEDD-medium": "sedd",
                   # Dream declares only AutoConfig + AutoModel (no AutoModelForCausalLM),
                   # so it must go through the diffusion harness, whose loader tries both.
                   "Dream-v0-Base": "hf_diffusion"}
# CoLA is a DiT prior + Text VAE; it has its own adapter.
COLA_ADAPTER = {"CoLA-DLM"}
COLA_STEPS, COLA_GS = 16, 7.0
NO_HARNESS = ["RULER 4k", "RULER 16k", "LB-QA", "infBench", "MemEval"]
BABILONG_LENS = ["4k", "8k", "16k"]


def find_model(name):
    """Case-insensitive, substring-tolerant lookup.

    The first version of this matched exact dir names, so `MDLM-OWT` missed the real
    `MDLM-owt`, `ELF-L` missed `ELF-L-owt-torch`, and OpenDFM missed
    `opendfm-chartool-3b` -- three models were reported MISSING while sitting on disk.
    Never trust a negative from a hand-written candidate list; normalise first.
    """
    for cand in BASELINES[name]:
        p = MODELS_ROOT / cand
        if p.is_dir():
            return p
    keys = [k.lower() for k in BASELINES[name]]
    for p in MODELS_ROOT.iterdir():
        if not p.is_dir():
            continue
        n = p.name.lower()
        if any(k in n or n in k for k in keys):
            return p
    # GLA was fetched into the shared HF cache, not MODELS_ROOT. Without this the 16th
    # model is silently dropped from every plan (found 2026-09-20: count came out 15).
    hub = HF_CACHE / "hub"
    if hub.is_dir():
        for p in hub.glob("models--*"):
            n = p.name.lower().replace("models--", "").replace("--", "/")
            if any(k.replace("/", "-") in n.replace("/", "-") or k in n for k in keys):
                snaps = sorted((p / "snapshots").glob("*"))
                if snaps:
                    return snaps[-1]
    return None


def verify_stage():
    """Static end-to-end check of every dataset's eval, so nothing aborts halfway.

    A half-run eval wastes an hour of H100 and leaves a partial table. The failure modes seen
    in this project were all static and catchable without a GPU: a flag the harness does not
    accept, a task dir that does not exist, a model dir with config but no weights, a corpus
    glob that matches nothing. This checks all of them and prints PASS/FAIL per dataset.
    """
    import ast
    print(f"{'dataset':14s} {'harness':44s} {'flags':6s} {'data':6s} {'verdict'}")
    print("-" * 96)
    harness = REPO / "scripts/eval/eval_generic_gen_match.py"
    # flags this driver passes to the generic harness
    passed = ["--model_dir", "--task_data_dir", "--output_dir", "--tasks",
              "--max_samples", "--max_new_tokens", "--temperature"]
    accepted = set()
    if harness.exists():
        src = harness.read_text()
        accepted = set(re.findall(r'add_argument\(\s*"(--[a-z_]+)"', src))
    bad = [f for f in passed if f not in accepted]
    flags_ok = harness.exists() and not bad

    rows = []
    for label, (task, _h) in list(GEN_TASKS.items()) + [("RACE", RACE_TASK)]:
        df = TASK_DATA / f"{task}.jsonl"
        data_ok = df.exists() and df.stat().st_size > 0
        n = 0
        if data_ok:
            with df.open() as fh:
                n = sum(1 for _ in fh)
        rows.append((label, "eval_generic_gen_match.py", flags_ok, data_ok,
                     f"{n} samples" if data_ok else "MISSING", f"OK ({n})" if (flags_ok and data_ok) else "FAIL"))
    for label in BABILONG_LENS:
        rows.append((f"bAbI-1 {label}", "run_open_context_bench.py --suite babilong", False, False,
                     "no HF-baseline branch", "FAIL (needs wiring)"))
    for s in NO_HARNESS:
        rows.append((s, "(none)", False, False, "not implemented", "FAIL (from scratch)"))
    for label, d in [("wikitext2", "wikitext2_full"), ("wikitext103", "wikitext103"),
                     ("pg19", "pg19"), ("lm1b", "lm1b"), ("ptb", "ptb"),
                     ("wmt14", "wmt14_de_en"), ("xsum", "xsum")]:
        dd = DIFFRWKV / "preprocessed_data" / d
        files = list(dd.glob("**/*")) if dd.is_dir() else []
        nf = sum(1 for p in files if p.is_file())
        rows.append((f"ppl:{label}", "eval_lm_ppl.py", True, nf > 0,
                     f"{nf} files under {d}", f"OK ({nf} files)" if nf else "FAIL (corpus empty)"))

    npass = 0
    for label, h, fl, da, note, verdict in rows:
        ok = verdict.startswith("OK")
        npass += ok
        print(f"{label:14s} {h[:43]:44s} {'ok' if fl else 'NO':6s} {'ok' if da else 'NO':6s} "
              f"{verdict:22s} {note}")
    print(f"\nPASS {npass} / {len(rows)}")
    if not flags_ok:
        print(f"!! harness {harness.name} does not accept: {bad}")
    print("\nNote: this is a static gate. It proves the flags/paths/corpora line up; it does "
          "not prove the model loads on GPU. Run one CPU smoke on a small model for that.")


def check_stage():
    print(f"{'baseline (paper row)':32s} {'group':28s} {'local checkpoint'}")
    print("-" * 96)
    present, missing = [], []
    for grp, names in GROUP.items():
        for n in names:
            p = find_model(n)
            print(f"{n:32s} {grp:28s} {str(p) if p else '** MISSING **'}")
            (present if p else missing).append(n)
    print(f"\npresent {len(present)} / {len(BASELINES)}   missing {len(missing)}")
    if missing:
        print("missing -> " + ", ".join(missing))
        print("\nCompute nodes have no network: these cannot run until downloaded on the")
        print("login node into the shared cache (see the offline-datasets note).")
    print("\n--- task data present? ---")
    ok = miss = 0
    for label, (task, _h) in GEN_TASKS.items():
        f = TASK_DATA / f"{task}.jsonl"
        print(f"  {label:10s} -> {f.name:24s} {'OK' if f.exists() else '** MISSING **'}")
        ok += f.exists(); miss += (not f.exists())
    print(f"  RACE       -> race.jsonl              "
          f"{'OK' if (TASK_DATA/'race.jsonl').exists() else '** MISSING **'}")
    print("\n--- long-context suites: data lives in the HF datasets cache, not preprocessed_data ---")
    suites = {"bAbILong": "datasets--RMT-team--babilong",
              "LB-QA (LongBench)": "datasets--THUDM--LongBench",
              "RULER": "datasets--simonjegou--ruler",
              "MemEval": "datasets--xiaowu0162--longmemeval",
              "infBench": "datasets--xinrongzhang2022--InfiniteBench"}
    for label, d in suites.items():
        p = HF_CACHE / "hub" / d
        print(f"  {label:18s} -> {d:38s} {'OK' if p.is_dir() else '** MISSING **'}")
    print("\n  harness: scripts/eval/run_open_context_bench.py already has "
          "--suite {longbench,babilong,ruler},")
    print("  but it takes --ckpt_dir/--arm (relay models). Baseline (HF) models need a "
          "wiring pass")
    print("  before those cells can run. Data is NOT the blocker; the LongBench default "
          "--longbench_config_dir")
    print("  (/root/benchmarks/LongBench/LongBench/config) does not exist here and must be "
          "pointed at the local copy.")


def cells(only_available=True):
    """(model, dataset) work units. GSM8K uses its own harness and is added separately."""
    out = []
    for gname in GROUP.values():
        for m in gname:
            if only_available and find_model(m) is None:
                continue
            for label, (task, h) in GEN_TASKS.items():
                out.append(dict(model=m, label=label, task=task, kind="gen", hours=h))
            out.append(dict(model=m, label="RACE", task=RACE_TASK[0], kind="gen",
                            hours=RACE_TASK[1]))
    for m in RELAY_MODELS:
        for label, (task, h) in GEN_TASKS.items():
            out.append(dict(model=m, label=label, task=task, kind="relay", hours=h))
        out.append(dict(model=m, label="RACE", task=RACE_TASK[0], kind="relay",
                        hours=RACE_TASK[1]))
    return out


def slug_of(model):
    if model in RELAY_MODELS:
        return "evalset/" + model
    return "baseline-" + model.lower().replace(" ", "-").replace(".", "").replace("/", "-")


def cell_done(cell):
    """True when this cell's jsonl already exists and is non-empty.

    Re-running a landed cell overwrites published numbers and burns GPU for nothing, so
    --skip-existing filters on this. 94 of 168 cells had already landed by 2026-09-20 15:40.
    """
    f = OUT_ROOT / slug_of(cell["model"]) / f"{cell['task']}.jsonl"
    try:
        return f.exists() and f.stat().st_size > 0
    except OSError:
        return False


def build_job(cell_or_cells, max_samples, gpu):
    """One QZ job. Pass a list of cells to run them SEQUENTIALLY on one GPU.

    Submission quota is counted in GPUs requested, not in work done, so one job per cell
    burns one GPU of quota per cell. Measured on 2026-09-20: 134 of 140 one-cell jobs were
    rejected with "总提交配额限制为 430 GPU ... 已使用 431 GPU". Packing N cells into one
    job cuts the quota footprint by N with no change in total compute.
    """
    cells = cell_or_cells if isinstance(cell_or_cells, list) else [cell_or_cells]
    first = cells[0]
    m = first["model"]
    slug = "baseline-" + m.lower().replace(" ", "-").replace(".", "").replace("/", "-")
    out_dir = OUT_ROOT / slug
    spec, shm = SPEC_BY_GPU[gpu]
    if len(cells) == 1:
        name = f"bl-{slug[9:]}-{first['label'].replace(' ','').replace('.','')}"[:60]
    else:
        name = f"bl-{slug[9:]}-pack{len(cells)}"[:60]

    steps = []
    for c in cells:
        task = c["task"]
        log = LOG_DIR / f"{slug.replace('/','-')}-{c['label'].replace(' ','').replace('.','')}.log"
        if c["kind"] == "relay":
            spec_r = RELAY_MODELS[c["model"]]
            runner = (f"python scripts/eval/{spec_r['script']} "
                      f"--ckpt_dir {spec_r['ckpt']} "
                      f"--task_data_dir {TASK_DATA} "
                      f"--output_dir {out_dir} "
                      f"--tasks {task} --method {spec_r['method']} "
                      f"--max_samples {max_samples} --steps 100 --cfg_scale 3.0 "
                      f"--max_new_tokens 32 --temperature 0.0")
        else:
            tok = (f"--tokenizer_dir {TOK_OVERRIDE[c['model']]} "
                   if c["model"] in TOK_OVERRIDE else "")
            # Masked/score-entropy diffusion LMs are not causal-LM shaped, so
            # AutoModelForCausalLM.generate() cannot run them at all (AutoModelForCausalLM
            # rejects MDLMConfig outright). eval_diffusion_lm_gen_match.py already implements
            # their denoising loops, so route those two there.
            if c["model"] in ELF_RUNNER:
                runner = (f"python scripts/eval/eval_elf_gen_match.py "
                          f"--model_dir {find_model(c['model'])} "
                          f"--task_data_dir {TASK_DATA} "
                          f"--output_dir {out_dir} "
                          f"--tasks {task} "
                          f"--max_samples {max_samples} --max_new_tokens 32 "
                          f"--sampling_method sde --num_sampling_steps 64 "
                          f"--sde_gamma 1.0 --self_cond_cfg_scale 3.0")
            elif c["model"] in COLA_ADAPTER:
                runner = (f"python scripts/eval/eval_cola_dlm_gen_match.py "
                          f"--model_dir {find_model(c['model'])} "
                          f"--task_data_dir {TASK_DATA} "
                          f"--output_dir {out_dir} "
                          f"--tasks {task} "
                          f"--max_samples {max_samples} --max_new_tokens 32 "
                          f"--steps {COLA_STEPS} --guidance_scale {COLA_GS} --batch_size 20")
            elif c["model"] in DIFFUSION_IMPL:
                runner = (f"python scripts/eval/eval_diffusion_lm_gen_match.py "
                          f"--impl {DIFFUSION_IMPL[c['model']]} "
                          f"--model_dir {find_model(c['model'])} " + tok +
                          f"--task_data_dir {TASK_DATA} "
                          f"--output_dir {out_dir} "
                          f"--tasks {task} "
                          f"--max_samples {max_samples} --max_new_tokens 32 "
                          f"--steps {DIFFUSION_STEPS.get(c['model'], 64)} --temperature 1.0")
            else:
                runner = (f"python scripts/eval/eval_generic_gen_match.py "
                          f"--model_dir {find_model(c['model'])} " + tok +
                          f"--task_data_dir {TASK_DATA} "
                          f"--output_dir {out_dir} "
                          f"--tasks {task} "
                          f"--max_samples {max_samples} --max_new_tokens 32 --temperature 0.0")
        steps.append(
            f"echo \"=== {c['model']} / {c['label']} ===\"\n"
            f"CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"{runner} > {log} 2>&1 || fail=1\n")
    # one PPL run per model covers all 7 corpora; eval_lm_ppl.py already branches
    # relay (--ckpt_dir) vs HF (--impl hf --model_dir)
    for m in dict.fromkeys(c["model"] for c in cells):
        # eval_lm_ppl.py only knows AutoModelForCausalLM and the relay loader, so it cannot
        # load a masked-diffusion, flow-matching or DiT+VAE checkpoint. Skipping the
        # step keeps a job from being marked job_failed when all 14 cells are good.
        if m in DIFFUSION_IMPL or m in ELF_RUNNER or m in COLA_ADAPTER:
            continue

        if m in RELAY_MODELS:
            r = RELAY_MODELS[m]
            ppl = (f"python scripts/eval/eval_lm_ppl.py --impl relay --ckpt_dir {r['ckpt']} "
                   f"--mode {r['method']} --corpora {PPL_CORPORA} --output {out_dir}/ppl.json")
        else:
            tok = (f"--tokenizer_dir {TOK_OVERRIDE[m]} " if m in TOK_OVERRIDE else "")
            ppl = (f"python scripts/eval/eval_lm_ppl.py --impl hf --model_dir {find_model(m)} "
                   f"{tok}--mode raw --corpora {PPL_CORPORA} --output {out_dir}/ppl.json")
        steps.append(f"echo \"=== {m} / PPL ===\"\n"
                     f"CUDA_VISIBLE_DEVICES=0 {ppl} > {LOG_DIR}/ppl-{slug.replace('/','-')}.log 2>&1 || fail=1\n")
    cmd = (f"set -uo pipefail\nfail=0\n"
           f"mkdir -p {LOG_DIR}\n"
           f"cd {REPO}\n"
           f"export HF_HOME={HF_CACHE}\n"
           # deps the compute-node image may lack (no network there): pre-installed on the
           # shared filesystem on the login node. Verified need: rouge (LongBench summaries +
           # infinityBench long_report), rotary-embedding-torch (CoLA's DiT rope).
           # ${PYTHONPATH:+:$PYTHONPATH} not :$PYTHONPATH: PYTHONPATH is UNSET in the compute
           # container, and under `set -u` a bare $PYTHONPATH aborts the whole shell
           # before mkdir/cd/any log redirect -- which silently killed every job
           # submitted after 16:52:34 with no log at all.
           f"export PYTHONPATH=/inspire/hdd/global_user/zhangjiaquan-253108540222/shared_pylibs${{PYTHONPATH:+:$PYTHONPATH}}\n"
           f"export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
           + "".join(steps) + "exit $fail\n")
    return {"workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
            "name": name,
            "description": f"baseline eval: {len(cells)} cell(s) {[c['label'] for c in cells]} "
                           f"on {m} (paper table cells)",
            "framework": "pytorch", "command": cmd, "max_running_time_ms": "43200000",
            "task_priority": 4,
            "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                                  "instance_count": 1, "spec_id": spec, "shm_gi": shm}]}


def qz_create(body, dry_run):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    fn = EVIDENCE / f"bl_{body['name']}.json"
    fn.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
    if dry_run:
        r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{fn}", "--dry-run"],
                           capture_output=True, text=True, timeout=120)
        if "DRY RUN" not in r.stdout:
            return None, f"dry-run failed: {r.stdout[-200:]}{r.stderr[-200:]}"
    r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{fn}"],
                       capture_output=True, text=True, timeout=180)
    try:
        return json.loads(r.stdout)["Result"]["job_id"], None
    except Exception:
        return None, f"submit failed: {r.stdout[:200]}{r.stderr[:200]}"


def live_count():
    r = subprocess.run(["qz", "train", "ListJobs", "--set", f"workspace_id={WS}",
                        "--set", "PageNumber=1", "--set", "page_size=40"],
                       capture_output=True, text=True, timeout=120)
    try:
        js = json.loads(r.stdout)["Result"]["jobs"]
    except Exception:
        return 0, {}
    alive = [j for j in js if j.get("status") in ("job_running", "job_queuing", "job_creating")
             and (j.get("name") or "").startswith("bl-")]
    return len(alive), {j.get("name"): j.get("status") for j in alive}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--gpus", type=int, default=50)
    ap.add_argument("--in-flight", type=int, default=50)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--gpu", type=int, default=1, choices=sorted(SPEC_BY_GPU))
    ap.add_argument("--cells-per-job", type=int, default=8,
                    help="pack this many cells into one job (sequential on one GPU). "
                         "Submission quota counts GPUs requested, so packing cuts the "
                         "quota footprint by this factor at identical total compute.")
    ap.add_argument("--force-models", default="",
                    help="run these models even if their cells already landed (e.g. to "
                         "re-run the PPL step, which is bundled into the same job)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not re-run a cell whose jsonl already exists")
    ap.add_argument("--exclude-models", default="",
                    help="comma list of models to skip (already submitted in an earlier pass)")
    ap.add_argument("--dry-run-bodies", action="store_true")
    ap.add_argument("--include-missing", action="store_true",
                    help="plan cells for models without a local checkpoint (they will fail)")
    a = ap.parse_args()

    if a.check:
        return check_stage()
    if a.verify:
        return verify_stage()

    cs = cells(only_available=not a.include_missing)
    skip = {m.strip() for m in a.exclude_models.split(',') if m.strip()}
    if skip:
        cs = [c for c in cs if c['model'] not in skip]
        print(f'excluding {sorted(skip)}', flush=True)
    if a.skip_existing:
        before = len(cs)
        force = {m.strip() for m in a.force_models.split(',') if m.strip()}
        cs = [c for c in cs if c['model'] in force or not cell_done(c)]
        print(f'skip-existing: dropped {before-len(cs)} landed cells, {len(cs)} left', flush=True)
    if a.plan or not (a.submit or a.status):
        print(f"runnable baselines: {len({c['model'] for c in cs})}   cells: {len(cs)}")
        tot = sum(c["hours"] for c in cs)
        print(f"total {tot:.1f} GPU-h; on {a.gpus} GPUs ideal {tot/a.gpus:.2f} h")
        waves = [cs[i:i + a.gpus] for i in range(0, len(cs), a.gpus)]
        cum = 0.0
        for k, w in enumerate(waves, 1):
            sp = max((c["hours"] for c in w), default=0)
            cum += sp
            print(f"  wave {k}: {len(w)} tasks, span {sp:.2f} h, cumulative {cum:.2f} h")
        if not a.submit:
            return

    if a.status:
        n, st = live_count()
        print(f"bl-* jobs alive: {n}")
        for k, v in list(st.items())[:20]:
            print(f"  {v:13s} {k}")
        return

    # submit loop: keep <= in_flight bl-* jobs alive; each job carries cells_per_job cells
    queue = [c for c in cs if c["model"] in RELAY_MODELS or find_model(c["model"]) is not None]
    done, quota_blocked = [], 0
    while queue:
        n, _ = live_count()
        if n >= a.in_flight:
            time.sleep(60)
            continue
        # A cell writes into a per-MODEL output dir, so a batch must never span two models
        # (that silently files model B's results under model A). Chunk within the head model.
        head = queue[0]["model"]
        n_take = 0
        for c in queue[:a.cells_per_job]:
            if c["model"] != head:
                break
            n_take += 1
        batch = queue[:n_take]
        del queue[:n_take]
        body = build_job(batch, a.max_samples, a.gpu)
        jid, err = qz_create(body, a.dry_run_bodies)
        labels = ",".join(c["label"] for c in batch)
        print(f"  {'OK  ' if jid else 'FAIL'} {batch[0]['model']:<22s} [{labels}] {jid or err}",
              flush=True)
        done.append((batch, jid, err))
        if err and "配额" in str(err):
            quota_blocked += 1
            if quota_blocked >= 2:
                print(f"\n!! 提交配额被拒 {quota_blocked} 次，停止提交。队列还剩 {len(queue)} 个 cell。",
                      flush=True)
                break
        time.sleep(2)
    ok = sum(1 for _, j, _ in done if j)
    print(f"\nsubmitted {ok} jobs covering "
          f"{sum(len(b) for b, j, _ in done if j)} cells / {len(cs)}; "
          f"failed {sum(1 for _, j, _ in done if not j)}", flush=True)


if __name__ == "__main__":
    main()

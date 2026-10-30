#!/usr/bin/env python3
"""Re-run ONLY the six choice columns whose prompts never showed the options.

Six tasks in the shared task data (arc_easy, arc_challenge, winogrande, sciq,
commonsenseqa, piqa) ship a `choices` field but build their `prompt` without it, so
every model was asked to answer a question whose options it never saw. The four task
harnesses now inline the options (see inline_choices() in scripts/eval/*.py); this
script re-runs the six columns with the fix and writes them to a FRESH directory
(results/benchmarks/fix2choices/<slug>/) so the old and the new numbers can be read
side by side. Nothing here overwrites results/benchmarks/baseline-*/, and the job
names are f2c-* rather than bl-* on purpose: bl-* is another driver's namespace and
its in-flight accounting counts jobs by that prefix.

Why a separate script at all: run_baseline_evals.py --skip-existing would skip
precisely the six stale cells this is meant to replace, and that file is shared.

All six tasks go into ONE harness invocation per model (--tasks a,b,c,...): the
harness already loops over a comma-separated list, so this costs one model load
instead of six.

Usage
  python3 run_fix2_choices_evals.py --list
  python3 run_fix2_choices_evals.py --submit --dry-run
  python3 run_fix2_choices_evals.py --submit
  python3 run_fix2_choices_evals.py --submit --models "Qwen2.5-3B,Phi-2"
  python3 run_fix2_choices_evals.py --status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline_evals import (  # noqa: E402
    COLA_GS, COLA_STEPS, DIFFUSION_IMPL, EVIDENCE, HF_CACHE, IMAGE, LCG, LOG_DIR,
    OUT_ROOT, PROJECT, RELAY_MODELS, REPO, SPEC_BY_GPU, TASK_DATA, TOK_OVERRIDE,
    WS, find_model,
)

SIX = ["arc_easy", "arc_challenge", "winogrande", "sciq", "commonsenseqa", "piqa"]
# CoLA's harness seeds the RNG ONCE per process and then walks the task list, so the noise
# draw a task gets depends on its POSITION in --tasks. The published CoLA run
# (bl-cola-dlm-pack14-full200) used all fourteen tasks in one process; reproducing that
# exact list keeps the six columns on the same RNG stream positions (8..13) as before, so
# before/after differ only by the prompt. CoLA is ~40 s/task, so the eight extra tasks are
# cheap insurance against an argument nobody can check later.
COLA_TASKS = ["mmlu", "obqa", "race", "story_cloze", "hellaswag", "siqa", "lambada",
              "squad", "arc_easy", "arc_challenge", "winogrande", "sciq",
              "commonsenseqa", "piqa"]
OUT_TAG = "fix2choices"
# 1 = the 1-GPU H100 spec (79fe954a); keep the whole sweep on one card per job.
GPU = 1

# The six columns are the only thing this script runs, so every model uses the same
# sample budget and decoding settings as the published protocol.
MAX_SAMPLES = 200
MAX_NEW_TOKENS = 32

# Which runner each system needs. Default is the HF causal-LM harness; the exceptions
# are copied from run_baseline_evals.build_job (which is authoritative, not guessed).
SPECIAL = {
    "CoLA-DLM": ("cola", None),
    "Dream-v0-Base": ("diffusion", "hf_diffusion"),
    "LLaDA-MoE-7B-A1B": ("diffusion", "hf_diffusion"),
}

# Our own rows of tab:capability were measured into results/benchmarks/<variant>/gen/ with
# the same option-less prompts, so they carry the same defect. Two of the four variants are
# already covered by RELAY_MODELS above (champ-dynamic-30k = champ-dynamic, global-dit-50k =
# Global); these are the remaining two, with the checkpoint taken from each variant's own
# gen/run_summary.json rather than guessed.
LOCAL_RELAY = {
    "champ-fixed-512": dict(
        ckpt="/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/outputs_relay/"
             "traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000",
        script="run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py", method="traj_cfg"),
    "global-birwkv-50k": dict(
        ckpt=str(REPO / "outputs_dynamic_basis/"
                 "laces-2.9B-dynlowrank-r32-global-birwkv768x8-b16-50k/step_00050000"),
        script="run_cola_dlm_tasks_prefix_suffix_cfg.py", method="prefix_cfg"),
    # The same-checkpoint control row of tab:capability ("Raw frozen decoder, no plan
    # written"). Its gen/run_summary.json carries the cfg harness's key set (no `method`,
    # no trajectory keys) and the champion ckpt, i.e. the single-z harness with --method
    # raw; its generations share nothing with champ-fixed-512's, so it is a separate run
    # and not a duplicate of the plan-writing arm.
    "raw-rwkv-genthenmatch": dict(
        ckpt="/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/outputs_relay/"
             "traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000",
        script="run_cola_dlm_tasks_prefix_suffix_cfg.py", method="raw"),
}
# Paper rows worth re-running here. mdlm/sedd/rwkv-6/elf are absent on purpose: they
# do not load on this image, so an f2c job for them would burn quota to produce logs.
MODELS = [
    "Qwen2.5-3B", "Qwen2.5-1.5B", "Llama-3.2-3B", "Phi-2", "Gemma-2-2B",
    "GLA-1.3B", "Mamba-2", "RWKV-7 World3",
    "LLaDA-8B-Base", "Dream-v0-Base", "LLaDA-MoE-7B-A1B", "OpenDFM CharTool",
    "CoLA-DLM", "STAR-LDM",
    # our own two relay checkpoints
    "champ-dynamic-30k", "global-dit-50k",
    # the other two of our own four tab:capability variants
    "champ-fixed-512", "global-birwkv-50k",
    # our control row
    "raw-rwkv-genthenmatch",
]
# Already being re-run, right now, by another driver's bl-*-pack14 jobs (checked with
# ListJobs across all 4 pages at 18:08). Excluded from the default pack so the two
# drivers do not pay for the same six columns twice; pass --models to force one.
IN_FLIGHT = {"Mamba-2", "RWKV-7 World3", "LLaDA-8B-Base", "Dream-v0-Base",
             "champ-dynamic-30k", "global-dit-50k"}

DEFAULT_MODELS = [m for m in MODELS if m not in IN_FLIGHT]


def slug_of(name: str) -> str:
    return "baseline-" + name.lower().replace(" ", "-").replace(".", "").replace("/", "-")


def out_dir_of(name: str) -> Path:
    return OUT_ROOT / OUT_TAG / slug_of(name)


def runner_for(name: str) -> str:
    """One python invocation that runs all six columns for this model."""
    out = out_dir_of(name)
    tasks = ",".join(SIX)
    spec_r = RELAY_MODELS.get(name) or LOCAL_RELAY.get(name)
    if spec_r:
        return (f"python scripts/eval/{spec_r['script']} "
                f"--ckpt_dir {spec_r['ckpt']} "
                f"--task_data_dir {TASK_DATA} "
                f"--output_dir {out} "
                f"--tasks {tasks} --method {spec_r['method']} "
                f"--max_samples {MAX_SAMPLES} --steps 100 --cfg_scale 3.0 "
                f"--max_new_tokens {MAX_NEW_TOKENS} --temperature 0.0")
    kind, impl = SPECIAL.get(name, ("hf", None))
    if kind == "cola":
        return (f"python scripts/eval/eval_cola_dlm_gen_match.py "
                f"--model_dir {find_model(name)} "
                f"--task_data_dir {TASK_DATA} "
                f"--output_dir {out} "
                f"--tasks {','.join(COLA_TASKS)} "
                f"--max_samples {MAX_SAMPLES} --max_new_tokens {MAX_NEW_TOKENS} "
                f"--steps {COLA_STEPS} --guidance_scale {COLA_GS} --batch_size 20")
    if kind == "diffusion":
        tok = f"--tokenizer_dir {TOK_OVERRIDE[name]} " if name in TOK_OVERRIDE else ""
        return (f"python scripts/eval/eval_diffusion_lm_gen_match.py "
                f"--impl {impl} --model_dir {find_model(name)} " + tok +
                f"--task_data_dir {TASK_DATA} "
                f"--output_dir {out} "
                f"--tasks {tasks} "
                f"--max_samples {MAX_SAMPLES} --max_new_tokens {MAX_NEW_TOKENS} "
                f"--steps 64 --temperature 1.0")
    tok = f"--tokenizer_dir {TOK_OVERRIDE[name]} " if name in TOK_OVERRIDE else ""
    return (f"python scripts/eval/eval_generic_gen_match.py "
            f"--model_dir {find_model(name)} " + tok +
            f"--task_data_dir {TASK_DATA} "
            f"--output_dir {out} "
            f"--tasks {tasks} "
            f"--max_samples {MAX_SAMPLES} --max_new_tokens {MAX_NEW_TOKENS} "
            f"--temperature 0.0")


def build_job(names: list[str]) -> dict:
    """One QZ job holding several models, run SEQUENTIALLY on one GPU.

    Quota is charged per requested GPU, so packing N models into one job cuts the
    footprint by N with no change in compute.
    """
    spec, shm = SPEC_BY_GPU[GPU]
    steps = []
    for m in names:
        log = LOG_DIR / f"f2c-{slug_of(m)}.log"
        steps.append(
            f"echo \"=== f2c {m} ===\"\n"
            f"CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"{runner_for(m)} > {log} 2>&1 || fail=1\n")
    cmd = (f"set -uo pipefail\nfail=0\n"
           f"mkdir -p {LOG_DIR} {OUT_ROOT / OUT_TAG}\n"
           f"cd {REPO}\n"
           f"export HF_HOME={HF_CACHE}\n"
           # ${PYTHONPATH:+:$PYTHONPATH} not :$PYTHONPATH -- PYTHONPATH is UNSET in the
           # compute container and under `set -u` a bare $PYTHONPATH aborts the shell
           # before any log redirect, losing the whole job silently.
           f"export PYTHONPATH=/inspire/hdd/global_user/zhangjiaquan-253108540222/shared_pylibs${{PYTHONPATH:+:$PYTHONPATH}}\n"
           f"export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
           + "".join(steps) + "exit $fail\n")
    head = slug_of(names[0])
    return {"workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
            "name": f"f2c-{head[9:]}-pack{len(names)}"[:60],
            "description": f"fix2 choices re-run: {names} (6 option-bearing columns) "
                           f"-> results/benchmarks/{OUT_TAG}/",
            "framework": "pytorch", "command": cmd, "max_running_time_ms": "43200000",
            "task_priority": 4,
            "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                                  "instance_count": 1, "spec_id": spec, "shm_gi": shm}]}


def qz_create(body: dict, dry_run: bool) -> tuple[str | None, str | None]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    fn = EVIDENCE / f"f2c_{body['name']}.json"
    fn.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
    if dry_run:
        r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{fn}", "--dry-run"],
                           capture_output=True, text=True, timeout=120)
        if "DRY RUN" not in r.stdout:
            return None, f"dry-run failed: {r.stdout[-300:]}{r.stderr[-300:]}"
        return None, "DRY RUN OK"
    r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{fn}"],
                       capture_output=True, text=True, timeout=180)
    try:
        return json.loads(r.stdout)["Result"]["job_id"], None
    except Exception:
        return None, f"submit failed: {r.stdout[:300]}{r.stderr[:300]}"


def landed(name: str) -> tuple[int, int]:
    """(#columns present with size>0, #columns with exactly 200 rows)."""
    ok = full = 0
    for t in SIX:
        p = out_dir_of(name) / f"{t}.jsonl"
        if p.exists() and p.stat().st_size > 0:
            ok += 1
            if sum(1 for _ in p.open(errors="ignore")) == 200:
                full += 1
    return ok, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--models", default="",
                    help="comma-separated subset; default = the paper rows not already "
                         "being re-run by another driver")
    ap.add_argument("--pack", type=int, default=3, help="models per QZ job")
    ap.add_argument("--skip-existing", action="store_true",
                    help="drop models whose six columns are all already 200 rows")
    a = ap.parse_args()

    names = [m.strip() for m in a.models.split(",") if m.strip()] if a.models else DEFAULT_MODELS
    unknown = [m for m in names if m not in MODELS]
    if unknown:
        raise SystemExit(f"not a paper row in this script: {unknown}\nknown: {MODELS}")

    if a.status:
        print(f"{'model':22s} {'cols':>5s} {'rows=200':>9s}  output dir")
        for m in MODELS:
            ok, full = landed(m)
            print(f"{m:22s} {ok:3d}/6 {full:5d}/6      {out_dir_of(m)}")
        return

    if a.skip_existing:
        before = len(names)
        names = [m for m in names if landed(m)[1] < len(SIX)]
        print(f"skip-existing: dropped {before - len(names)} landed models, {len(names)} left")

    packs = [names[i:i + a.pack] for i in range(0, len(names), a.pack)]
    if a.list or not (a.submit or a.dry_run):
        print(f"{len(names)} models -> {len(packs)} job(s) of <= {a.pack} on {GPU} GPU each")
        print(f"output: {OUT_ROOT / OUT_TAG}/<model>/")
        for k, p in enumerate(packs, 1):
            print(f"  job {k}: {p}")
        def has_local(m):
            if m in RELAY_MODELS or m in LOCAL_RELAY:
                return True
            try:
                return find_model(m) is not None
            except KeyError:      # not a row of run_baseline_evals.BASELINES
                return False
        missing = [m for m in names if not has_local(m)]
        if missing:
            print(f"  ! no local checkpoint for: {missing} (those cells will fail)")
        return

    for k, pack in enumerate(packs, 1):
        body = build_job(pack)
        jid, err = qz_create(body, a.dry_run)
        if a.dry_run:
            print(f"  [{'OK ' if err == 'DRY RUN OK' else 'FAIL'}] job {k} {body['name']}: {err}")
            continue
        print(f"  [{'OK  ' if jid else 'FAIL'}] job {k} {body['name']}: {jid or err}", flush=True)
        if err and "配额" in str(err):
            print("!! quota rejected -- stopping", flush=True)
            break
        time.sleep(2)


if __name__ == "__main__":
    main()

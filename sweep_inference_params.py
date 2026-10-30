#!/usr/bin/env python3
"""Inference hyperparameter sweep for the four completed LACES 2.9B models.

Built on facts verified from the repo + QZ history on 2026-09-20 (not assumed):

  generator  scripts/eval/run_cola_dlm_tasks_prefix_suffix_cfg.py            (--method prefix_cfg|raw)
             scripts/eval/run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py (--method traj_cfg|raw)
  scorer     scripts/eval/score_gen_match.py <variant_dir>   (login node, no GPU)
  --tasks    is COMMA separated (default is ",".join(TASKS_DEFAULT))
  QZ         project-160ccb20 + lcg-71b971a7 + spec_id chosen by GPU count.
             The 09-17 history shows the same batch job_stopped on the old lcg and
             job_succeeded after switching to lcg-71b971a7, so this pairing is the
             one that actually schedules.

WHY temperature matters here (the trap this sweep exists to avoid):
  every one of the 114 parameterised eval jobs in the QZ history ran
  temperature=0.0, i.e. GREEDY, where top_k/top_p have no effect at all.
  So the recorded top_k=50/top_p=0.9 were never exercised. Any sweep of those
  two axes must be anchored at temperature > 0, or the axes are dead.

Usage
-----
  # 1. show the plan + cost (writes job bodies, submits nothing)
  python sweep_inference_params.py --stage repro
  python sweep_inference_params.py --stage ofat  --tasks arc_easy,arc_challenge,sciq
  # 2. submit (needs --submit; --dry-run-bodies runs qz --dry-run on each body first)
  python sweep_inference_params.py --stage ofat --submit --dry-run-bodies --gpu 1
  # 3. score whatever landed, then look at per-axis marginals
  python sweep_inference_params.py --stage score
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
DIFFRWKV = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv")
EVIDENCE = DIFFRWKV / ".omo/evidence/qz-training-skill"
OUT_ROOT = REPO / "results/benchmarks/infer_sweep"
LOG_DIR = REPO / "logs/qz"
TASK_DATA_DIR = DIFFRWKV / "baseline/Cola-DLM/generate_task_data"

# --- QZ standing config (see STANDING_SUBMIT_CONFIG.md) -----------------------
WS = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
SPEC_BY_GPU = {  # verified from qz train GetTrainScheduleConfig for LCG above
    1: ("79fe954a-be92-4772-ac0b-94ad8a79b7bb", 100),
    2: ("26ef0d6e-330d-4650-a18a-7e1fbe8f3717", 400),
    4: ("8a53ac21-299a-4dee-85e9-9c04a544cf8d", 900),
    8: ("7166bd2e-6cbe-4bd9-be38-762d11003e7f", 1800),
}

MODEL_ROOT = REPO / "outputs_dynamic_basis"
MODELS = {
    "global-dit-50k": dict(
        ckpt=str(MODEL_ROOT / "laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000"),
        arms=["prefix_cfg"], consistent=True),
    "global-birwkv-50k": dict(
        ckpt=str(MODEL_ROOT / "laces-2.9B-dynlowrank-r32-global-birwkv768x8-b16-50k/step_00050000"),
        arms=["prefix_cfg"], consistent=False),
    "global-birwkv-fixed-50k": dict(
        ckpt=str(MODEL_ROOT / "laces-2.9B-fixed-r32-global-birwkv768x8-b16-50k/step_00050000"),
        arms=["prefix_cfg"], consistent=False),
    "champ-fixed-26k": dict(
        ckpt=str(DIFFRWKV / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000"),
        # trajectory_mode=True in this checkpoint, so traj_cfg is its real arm. Running it
        # through the single-z prefix_cfg path produced pure garbage (1.00% vs 24.67%), and
        # the historical champ-fixed-512 rows carry method=traj_cfg -- do not use prefix_cfg.
        arms=["traj_cfg"], consistent=True),
}

# --- parameter space ---------------------------------------------------------
# stage "repro" = the exact published setting; it is the positive control.
PUBLISHED = dict(cfg_scale=3.0, temperature=0.0, top_k=50, top_p=0.9,
                 repetition_penalty=1.1, steps=100, max_new_tokens=32)

# stage "ofat" runs one axis at a time from this base. temperature>0 so that
# top_k / top_p are actually live.
BASE = dict(PUBLISHED, temperature=0.5)

AXES = {
    "cfg_scale":          [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "temperature":        [0.0, 0.3, 0.5, 0.7, 1.0],
    "top_k":              [0, 20, 50, 100],
    "top_p":              [0.90, 0.95, 1.00],
    "repetition_penalty": [1.0, 1.1, 1.2, 1.3],
    "steps":              [25, 100, 250, 1000],
    "max_new_tokens":     [16, 32, 64],
}
# only the traj arm of the trajectory evaluator accepts these
TRAJ_AXES = {
    "trajectory_state_blend": [0.3, 0.5, 0.7, 1.0],
    "cond_boundary_scale":    [0.5, 1.0, 1.5],
}
DEFAULTS_TASK_SET = "arc_easy,arc_challenge,sciq"   # 3-task screen for the sweep
FULL_TASK_SET = "piqa,arc_easy,arc_challenge,winogrande,sciq,commonsenseqa"

# measured: full 6-task x 200-sample Fix2 on 1xH100 = 1.1 h
HOURS_PER_FULL_CFG = 1.1


def cfg_name(model, arm, p):
    tag = "-".join(f"{k[:3]}{v}" for k, v in sorted(p.items()) if k in AXES or k in TRAJ_AXES)
    return f"{model}__{arm}__{tag}"


def configs_for(stage, models, factorial=None, axes=None, arms=None):
    """Yield (model, arm, params_dict) tuples."""
    for m in models:
        for arm in MODELS[m]["arms"]:
            if arms and arm not in arms:
                continue
            if stage == "repro":
                yield m, arm, dict(PUBLISHED)
                continue
            pool = dict(AXES)
            if arm == "traj_cfg":
                pool.update(TRAJ_AXES)
            if stage == "ofat":
                for axis in (axes or list(pool)):
                    for v in pool[axis]:
                        p = dict(BASE)
                        p[axis] = v
                        yield m, arm, p
            elif stage == "factorial":
                if not factorial:
                    raise SystemExit("--factorial required for stage=factorial, "
                                     "e.g. --factorial 'cfg_scale=1,3;temperature=0.3,0.7'")
                keys, vals = [], []
                for part in factorial.split(";"):
                    k, v = part.split("=")
                    keys.append(k.strip())
                    vals.append([float(x) if "." in x else int(x) for x in v.split(",")])
                for combo in itertools.product(*vals):
                    yield m, arm, dict(BASE, **dict(zip(keys, combo)))


def build_job(model, arm, p, tasks, max_samples, gpu, tag):
    gen_script = ("run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py"
                  if arm == "traj_cfg" else "run_cola_dlm_tasks_prefix_suffix_cfg.py")
    name = cfg_name(model, arm, p) + (f"__{tag}" if tag else "")
    out_dir = OUT_ROOT / name
    log = LOG_DIR / f"sweep-{name}.log"
    spec, shm = SPEC_BY_GPU[gpu]

    flags = [f"--ckpt_dir {MODELS[model]['ckpt']}",
             f"--task_data_dir {TASK_DATA_DIR}",
             f"--output_dir {out_dir}",
             f"--tasks {tasks}",
             f"--method {arm}",
             f"--max_samples {max_samples}",
             f"--steps {p['steps']}",
             f"--cfg_scale {p['cfg_scale']}",
             f"--max_new_tokens {p['max_new_tokens']}",
             f"--temperature {p['temperature']}",
             f"--top_k {p['top_k']}",
             f"--top_p {p['top_p']}",
             f"--repetition_penalty {p['repetition_penalty']}",
             "--seed 42"]
    if arm == "traj_cfg":
        if p.get("trajectory_state_blend") is not None:
            flags.append(f"--trajectory_state_blend {p['trajectory_state_blend']}")
        if p.get("cond_boundary_scale") is not None:
            flags.append(f"--cond_boundary_scale {p['cond_boundary_scale']}")

    cmd = (f"set -uo pipefail\n"
           f"mkdir -p {LOG_DIR}\n"
           f"mkdir -p {out_dir}\n"
           f"cd {REPO}\n"
           f"export HF_HOME=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache\n"
           f"export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n"
           f"CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
           f"python scripts/eval/{gen_script} " + " ".join(flags) + f" > {log} 2>&1\n")
    return {
        "workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
        "name": name, "description": f"{model} / {arm} inference-param sweep: {p}",
        "framework": "pytorch", "command": cmd, "max_running_time_ms": "7200000",
        "task_priority": 4,
        "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                              "instance_count": 1, "spec_id": spec, "shm_gi": shm}],
    }


def submit(body, dry_run):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    fn = EVIDENCE / f"sweep_{body['name']}.json"
    fn.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
    if dry_run:
        r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{fn}", "--dry-run"],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or "DRY RUN" not in r.stdout:
            return None, f"dry-run failed: {r.stdout[-300:]}{r.stderr[-300:]}"
    r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{fn}"],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        return None, f"submit failed: {r.stdout[-300:]}{r.stderr[-300:]}"
    try:
        return json.loads(r.stdout)["Result"]["job_id"], None
    except Exception:
        return None, f"unparsable response: {r.stdout[:200]}"


def _diag(rows, max_new_tokens):
    """Cheap deterministic pre-screen so human/LLM reading goes to the interesting configs.

    Returns (n, trunc_frac, degen_frac). degen = generation is dominated by one
    repeated 4-gram or by non-word characters (the 'a a a thed the [' failure).
    """
    import re
    n = len(rows) or 1
    trunc = degen = 0
    for r in rows:
        g = (r.get("generate") or "")
        if r.get("generated_tokens") and max_new_tokens and r["generated_tokens"] >= max_new_tokens:
            trunc += 1
        toks = g.split()
        if len(toks) >= 8:
            grams = Counter(tuple(toks[i:i + 4]) for i in range(len(toks) - 3))
            if grams and grams.most_common(1)[0][1] >= 3:
                degen += 1
        elif g.strip() and sum(c.isalnum() for c in g) / max(len(g), 1) < 0.5:
            degen += 1
    return n, trunc / n, degen / n


def dump_stage(per_config, tasks):
    """Print a human-readable digest of every saved config so it can be judged directly.

    Every config dir keeps gen/<task>.jsonl (full prompt/choices/ground_truth/generate)
    plus run_summary.json (ckpt_dir, checkpoint_step, steps, cfg_scale, max_new_tokens,
    temperature, top_k, top_p, repetition_penalty, tasks), and the dir NAME encodes the
    parameters too — so a result is never orphaned from the setting that produced it.
    """
    dirs = sorted(d for d in OUT_ROOT.glob("*") if d.is_dir())
    if not dirs:
        print(f"no variant dirs under {OUT_ROOT}"); return
    for d in dirs:
        rs = next((p for p in (d / "gen" / "run_summary.json", d / "run_summary.json")
                   if p.exists()), None)
        meta = json.loads(rs.read_text()) if rs else {}
        print("=" * 100)
        print(d.name)
        if meta:
            keys = ["ckpt_dir", "checkpoint_step", "steps", "cfg_scale", "temperature",
                    "top_k", "top_p", "repetition_penalty", "max_new_tokens"]
            print("  params: " + "  ".join(f"{k}={meta.get(k)}" for k in keys if k in meta))
        mnt = int(meta.get("max_new_tokens") or 32)
        for task in [t.strip() for t in tasks.split(",")]:
            # the evaluator writes flat into --output_dir; older runs passed ".../gen" as the
            # output_dir, so accept both layouts
            f = next((p for p in (d / "gen" / f"{task}.jsonl", d / f"{task}.jsonl")
                      if p.exists()), None)
            if f is None:
                continue
            with f.open() as fh:
                rows = [json.loads(l) for l in fh if l.strip()]
            n, trunc, degen = _diag(rows, mnt)
            print(f"  [{task}] n={n} truncated={trunc:.0%} degenerate={degen:.0%}")
            for r in rows[:per_config]:
                g = " ".join((r.get("generate") or "").split())[:110]
                print(f"      GT={str(r.get('ground_truth'))[:30]!r:34s} GEN={g!r}")


def score_stage(tasks):
    """Run the unified scorer over every sweep variant dir and report marginals."""
    scorer = REPO / "scripts/eval/score_gen_match.py"
    dirs = sorted(d for d in OUT_ROOT.glob("*") if d.is_dir())
    if not dirs:
        print(f"no variant dirs under {OUT_ROOT}"); return
    rows = []
    for d in dirs:
        r = subprocess.run([sys.executable, str(scorer), str(d)],
                           capture_output=True, text=True, timeout=1800, cwd=str(REPO))
        print(f"--- {d.name}\n{r.stdout.strip()[:400]}")
        rows.append((d.name, r.stdout))
    (OUT_ROOT / "score_raw.txt").write_text("\n".join(f"### {n}\n{o}" for n, o in rows))
    print(f"\nraw scorer output -> {OUT_ROOT/'score_raw.txt'}")
    print("NOTE: score_gen_match.py prints rows, not JSON; parse the file above, "
          "or add a JSON mode to the scorer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["repro", "ofat", "factorial", "score", "dump"])
    ap.add_argument("--dump-per-config", type=int, default=5,
                    help="for --stage dump: sample generations to print per config-task")
    ap.add_argument("--models", default="global-dit-50k,global-birwkv-50k,"
                                       "global-birwkv-fixed-50k,champ-fixed-26k")
    ap.add_argument("--tasks", default=DEFAULTS_TASK_SET)
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--axes", default=None, help="comma list to restrict OFAT axes")
    ap.add_argument("--arms", default=None,
                    help="restrict arms, e.g. 'prefix_cfg' (champ-fixed-26k also has traj_cfg)")
    ap.add_argument("--factorial", default=None, help="e.g. 'cfg_scale=1,3;temperature=0.3,0.7'")
    ap.add_argument("--gpu", type=int, default=1, choices=sorted(SPEC_BY_GPU))
    ap.add_argument("--tag", default="", help="suffix to keep sweeps apart")
    ap.add_argument("--submit", action="store_true", help="actually submit (default: plan only)")
    ap.add_argument("--dry-run-bodies", action="store_true", help="qz --dry-run each body first")
    ap.add_argument("--allow-unfixed-birwkv", action="store_true",
                    help="include the two birwkv models although their eval-side conv_state/ffn_state "
                         "convention is still unverified (train used zeroed, the working-tree fix preserves)")
    ap.add_argument("--force-over", type=int, default=60, help="refuse to submit more than this many jobs")
    a = ap.parse_args()

    if a.stage == "score":
        return score_stage(a.tasks)
    if a.stage == "dump":
        return dump_stage(a.dump_per_config, a.tasks)

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    for m in models:
        if m not in MODELS:
            raise SystemExit(f"unknown model {m!r}; known: {list(MODELS)}")

    gated = [m for m in models if not MODELS[m]["consistent"]]
    if gated and not a.allow_unfixed_birwkv:
        print(f"!! excluding {gated}: their eval still needs the conv_state/ffn_state convention "
              f"settled (see notes). Pass --allow-unfixed-birwkv to include them anyway.\n")
        models = [m for m in models if MODELS[m]["consistent"]]

    axes = [x.strip() for x in a.axes.split(",")] if a.axes else None
    arms = [x.strip() for x in a.arms.split(",")] if a.arms else None
    cfgs = list(configs_for(a.stage, models, a.factorial, axes, arms))
    n_tasks = len(a.tasks.split(","))
    hours = len(cfgs) * HOURS_PER_FULL_CFG * (n_tasks / 6.0) * (a.max_samples / 200.0)

    print(f"stage={a.stage}  models={models}  jobs={len(cfgs)}  "
          f"tasks={n_tasks}  max_samples={a.max_samples}")
    print(f"estimated compute: {hours:.1f} GPU-h on 1xH100 "
          f"({HOURS_PER_FULL_CFG}h for 6 tasks x 200 samples, scaled)\n")
    for m, arm, p in cfgs[:12]:
        print(f"  {cfg_name(m, arm, p)}")
    if len(cfgs) > 12:
        print(f"  ... +{len(cfgs)-12} more")

    if not a.submit:
        print("\n(plan only — pass --submit to actually create jobs)")
        return
    if len(cfgs) > a.force_over:
        raise SystemExit(f"refusing to submit {len(cfgs)} jobs > --force-over {a.force_over}")

    ok, bad = [], []
    for m, arm, p in cfgs:
        body = build_job(m, arm, p, a.tasks, a.max_samples, a.gpu, a.tag)
        jid, err = submit(body, a.dry_run_bodies)
        (ok if jid else bad).append((body["name"], jid or err))
        print(f"  {'OK  ' if jid else 'FAIL'} {body['name']} {jid or err}")
    print(f"\nsubmitted {len(ok)} / {len(cfgs)}; failed {len(bad)}")
    for n, e in bad:
        print(f"  {n}: {e}")


if __name__ == "__main__":
    main()

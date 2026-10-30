#!/usr/bin/env bash
# Wave 2 of the InfiniteBench re-run, split out into its own job.
#
# Why this is separate from run_infinitebench_fixed_sweep.sh: that script runs the 8 baselines
# and then `wait`s before starting our two arms. mamba-2.8b turned out to have no fast kernels
# on this image (mamba_ssm / causal_conv1d / selective_scan_cuda are all absent, and
# transformers' mambapy pscan is gated on `self.training` so it never helps at eval), so it
# falls back to a Python sequential scan at ~3 min/row -- about 12.5 h for 250 rows, which both
# exceeds the 12 h job cap and would hold our own arms hostage behind the `wait`. Our arms are
# the numbers the paper needs, so they get their own job and their own GPUs.
#
# Same protocol as wave 1: identical datasets, samples, token budgets and relay hyperparameters
# recovered from the original run's `config` block.
set -euo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

OUT=$REPO/results/benchmarks/longctx_fix
LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

DATASETS=longbook_qa_eng,longbook_choice_eng,kv_retrieval,code_debug,math_find
DATA=$ROOT/hf_cache/hub/datasets--xinrongzhang2022--InfiniteBench/snapshots/90f0394333616266d9fe85824ceaf505093cbaa5
COMMON="--suite infinitebench --datasets $DATASETS --data_dir $DATA \
  --max_samples 50 --max_input_tokens 32768 --max_new_tokens 128 --seed 42 --device cuda"

run_relay () {  # name, ckpt_dir, arm, expected_step, gpu
  local name=$1 ckpt=$2 arm=$3 step=$4 gpu=$5
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_open_context_bench.py $COMMON \
    --ckpt_dir "$ckpt" --arm "$arm" --expected_step "$step" \
    --plan_steps 100 --cfg_scale 2.0 --blend 0.7 --diffusion_sampler ddpm \
    --output "$OUT/${name}__infinitebench.json" \
    > "$LOGS/ibfix-${name}-infinitebench.log" 2>&1 &
  echo "  [gpu$gpu] $name ($arm) -> $LOGS/ibfix-${name}-infinitebench.log"
}

# champ-dynamic-30k is trajectory_mode=True  -> --arm dynamic (LACESRuntime, chunked)
# global-dit-50k    is trajectory_mode=False -> --arm global  (GlobalRuntime, single-z)
# Running either through the other arm yields garbage; the arms are not interchangeable.
echo "=== WAVE 2: our two arms ==="
run_relay champ-dynamic-30k \
  "$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000" \
  dynamic 30000 0
run_relay global-dit-50k \
  "$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000" \
  global 50000 1
wait
echo "=== WAVE 2 done ==="

grep -h "^\[budget\]" "$LOGS"/ibfix-champ-dynamic-30k-infinitebench.log \
  "$LOGS"/ibfix-global-dit-50k-infinitebench.log || true
echo "=== ALL DONE ==="

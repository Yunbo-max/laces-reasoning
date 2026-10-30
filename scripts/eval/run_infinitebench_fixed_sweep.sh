#!/usr/bin/env bash
# InfiniteBench re-run with the per-model truncation-budget fix + the fla generate fix.
#
# Why this re-run exists (see logs/qz/INFINITEBENCH_FIX_JOBS.md for the full evidence):
#   1. phi-2 scored an exact 0.000 on all five datasets because a 32768-token prompt was fed
#      into a 2048-position table. `effective_input_budget` now clamps per model.
#   2. gla-1.3b and rwkv7-world3 reach `generate` through fla's mixin, which re-feeds the
#      whole sequence every decode step against transformers 5.3.0. `patch_fla_recurrent_generate`
#      fixes it; every previous gla row is invalid.
#   3. The choice scorer missed a glued bare letter ("BQuestion:") and picked the longest
#      matching option instead of the first mentioned.
#
# One model per GPU, 8 GPUs, two waves (10 models). Each model writes its own JSON and its own
# log, so a single failure never blocks the rest and partial results survive a kill.
#
# Hard env requirements (compute nodes have NO network):
#   HF_HOME / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE  -- offline weight+tokenizer resolution
#   PYTHONPATH written with ${PYTHONPATH:+:$PYTHONPATH} -- a bare $PYTHONPATH under `set -u`
#   kills the job with no log at all.
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

# The same five datasets and the same sample/token budgets as the original run, recovered from
# the `config` block of results/benchmarks/longctx/*__infinitebench.json. Only the fix differs.
DATASETS=longbook_qa_eng,longbook_choice_eng,kv_retrieval,code_debug,math_find
DATA=$ROOT/hf_cache/hub/datasets--xinrongzhang2022--InfiniteBench/snapshots/90f0394333616266d9fe85824ceaf505093cbaa5
COMMON="--suite infinitebench --datasets $DATASETS --data_dir $DATA \
  --max_samples 50 --max_input_tokens 32768 --max_new_tokens 128 --seed 42 --device cuda"

run_hf () {   # name, model_dir, gpu
  local name=$1 dir=$2 gpu=$3
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_open_context_bench.py $COMMON \
    --model_dir "$dir" --arm raw \
    --output "$OUT/${name}__infinitebench.json" \
    > "$LOGS/ibfix-${name}-infinitebench.log" 2>&1 &
  echo "  [gpu$gpu] $name -> $LOGS/ibfix-${name}-infinitebench.log"
}

run_relay () {  # name, ckpt_dir, arm, expected_step, gpu
  local name=$1 ckpt=$2 arm=$3 step=$4 gpu=$5
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_open_context_bench.py $COMMON \
    --ckpt_dir "$ckpt" --arm "$arm" --expected_step "$step" \
    --plan_steps 100 --cfg_scale 2.0 --blend 0.7 --diffusion_sampler ddpm \
    --output "$OUT/${name}__infinitebench.json" \
    > "$LOGS/ibfix-${name}-infinitebench.log" 2>&1 &
  echo "  [gpu$gpu] $name ($arm) -> $LOGS/ibfix-${name}-infinitebench.log"
}

echo "=== WAVE 1: 8 baselines (one per GPU) ==="
run_hf qwen2.5-3b    "$ROOT/models/Qwen2.5-3B"   0
run_hf qwen2.5-1.5b  "$ROOT/models/qwen2.5-1.5b" 1
run_hf llama-3.2-3b  "$ROOT/models/Llama-3.2-3B" 2
run_hf phi-2         "$ROOT/models/phi-2"        3
run_hf gemma-2-2b    "$ROOT/models/gemma-2-2b"   4
run_hf gla-1.3b      "$ROOT/hf_cache/hub/models--fla-hub--gla-1.3B-100B/snapshots/46b15820a4df269e99aed9d709e017677c15d24b" 5
run_hf rwkv7-world3  "$ROOT/models/RWKV7-Goose-World3-2.9B-HF" 6
run_hf mamba-2.8b    "$ROOT/models/mamba-2.8b"   7
wait
echo "=== WAVE 1 done ==="

echo "=== WAVE 2: our two arms ==="
# champ-dynamic-30k is trajectory_mode=True  -> --arm dynamic (LACESRuntime, chunked)
# global-dit-50k    is trajectory_mode=False -> --arm global  (GlobalRuntime, single-z)
# Running either through the other arm yields garbage; the arms are not interchangeable.
run_relay champ-dynamic-30k \
  "$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000" \
  dynamic 30000 0
run_relay global-dit-50k \
  "$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000" \
  global 50000 1
wait
echo "=== WAVE 2 done ==="

echo "=== effective budgets actually used ==="
grep -h "^\[budget\]" "$LOGS"/ibfix-*-infinitebench.log || true
echo "=== fla patch status (must be True for gla-1.3b and rwkv7-world3) ==="
grep -H "fla recurrent-generate patch applied" "$LOGS"/ibfix-*-infinitebench.log || true
echo "=== ALL DONE ==="

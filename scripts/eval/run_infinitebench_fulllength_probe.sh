#!/usr/bin/env bash
# Does the recurrent asymmetry actually exist? -- InfiniteBench at (near) full document length.
#
# The main sweep holds --max_input_tokens at 32768 for every model, which preserves the
# original invocation and makes the models comparable. But it does NOT test the claim that
# matters for the paper: recurrent models have no positional ceiling, so they *can* ingest the
# whole ~136k document while phi-2/gemma/Qwen structurally cannot. That asymmetry is only a
# finding if the recurrent models still produce sane output at that length -- an unbounded
# state that silently saturates would look identical to a ceiling, and reporting it as an
# architectural advantage would be wrong.
#
# So: same five-dataset protocol, recurrent models only, budget raised to 131072. 20 samples to
# keep it affordable; this probe answers a yes/no question, it is not the headline table.
#
# CORRECTION, measured after the run (2026-09-21): 131072 is NOT "effectively untruncated", as
# an earlier version of this comment claimed. The real tokenized prompts are much longer than the
# chars/4 estimate that number was chosen from -- longbook_choice_eng averages 228840 tokens and
# code_debug 217836 -- so kv_retrieval and code_debug are still truncated on 20/20 rows even at
# 131072, and only math_find is genuinely complete. The probe is therefore a 4x-more-context
# comparison, not an untruncated one. It still answers the question that matters (does more
# context help?) and the answer is no: gla moves +0.000..+0.050 and rwkv7 COLLAPSES
# (longbook_choice_eng 0.350 -> 0.000, 98/100 rows repetition loops).
#
# Read alongside: logs/qz/INFINITEBENCH_FIX_JOBS.md
set -euo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

OUT=$REPO/results/benchmarks/longctx_full
LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

DATASETS=longbook_qa_eng,longbook_choice_eng,kv_retrieval,code_debug,math_find
DATA=$ROOT/hf_cache/hub/datasets--xinrongzhang2022--InfiniteBench/snapshots/90f0394333616266d9fe85824ceaf505093cbaa5
COMMON="--suite infinitebench --datasets $DATASETS --data_dir $DATA \
  --max_samples 20 --max_input_tokens 131072 --max_new_tokens 128 --seed 42 --device cuda"

run_hf () {   # name, model_dir, gpu
  local name=$1 dir=$2 gpu=$3
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_open_context_bench.py $COMMON \
    --model_dir "$dir" --arm raw \
    --output "$OUT/${name}__infinitebench.json" \
    > "$LOGS/ibfull-${name}-infinitebench.log" 2>&1 &
  echo "  [gpu$gpu] $name -> $LOGS/ibfull-${name}-infinitebench.log"
}

run_relay () {  # name, ckpt_dir, arm, expected_step, gpu
  local name=$1 ckpt=$2 arm=$3 step=$4 gpu=$5
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_open_context_bench.py $COMMON \
    --ckpt_dir "$ckpt" --arm "$arm" --expected_step "$step" \
    --plan_steps 100 --cfg_scale 2.0 --blend 0.7 --diffusion_sampler ddpm \
    --output "$OUT/${name}__infinitebench.json" \
    > "$LOGS/ibfull-${name}-infinitebench.log" 2>&1 &
  echo "  [gpu$gpu] $name ($arm) -> $LOGS/ibfull-${name}-infinitebench.log"
}

# One GPU each -- recurrent models only. Positional models are excluded on purpose: raising
# their budget cannot help, their window is the point of the comparison. mamba-2.8b is excluded
# too, for throughput not principle: this image has no mamba fast kernels, so it runs a Python
# sequential scan at ~3 min/row at 32k, i.e. ~12 min/row here. It cannot finish and its absence
# must be reported as a harness limitation, not as a mamba result.
echo "=== recurrent models at 131072 (effectively untruncated) ==="
run_hf rwkv7-world3 "$ROOT/models/RWKV7-Goose-World3-2.9B-HF" 0
run_hf gla-1.3b     "$ROOT/hf_cache/hub/models--fla-hub--gla-1.3B-100B/snapshots/46b15820a4df269e99aed9d709e017677c15d24b" 1
run_relay champ-dynamic-30k \
  "$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000" \
  dynamic 30000 2
run_relay global-dit-50k \
  "$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000" \
  global 50000 3
wait

echo "=== budgets actually used (all must be 131072 / cli_cap_only) ==="
grep -h "^\[budget\]" "$LOGS"/ibfull-*-infinitebench.log || true
echo "=== OOM check ==="
grep -l -E "CUDA out of memory|OutOfMemoryError" "$LOGS"/ibfull-*-infinitebench.log || echo "no OOM"
echo "=== ALL DONE ==="

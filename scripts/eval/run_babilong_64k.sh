#!/usr/bin/env bash
# bAbILong 64k for tab:longbyLen -- the three rows that show "--" in the 64k column.
#
# WHY THIS SCRIPT AND NOT run_open_context_bench.py:
#   tab:longbyLen's 4k/8k/16k cells were NOT produced by run_open_context_bench.py. They come
#   from run_cola_dlm_tasks_prefix_suffix{,_trajectory}_cfg.py writing
#   results/benchmarks/<arm>/babilong{4k,8k,16k}/babilong_qa{1,2}.jsonl, which are then scored by
#   loose containment. The two harnesses disagree by ~5 points on the SAME predictions
#   (falsify_zero/rwkv7-loop.json self-reports 41.0 at qa1_4k; loose containment on its rows
#   gives 46.0). So a 64k cell produced by the other harness would not be comparable to the
#   4k/8k/16k cells in its own row. This script therefore uses the SAME harness, the SAME
#   generation args, and the SAME checkpoints that produced the published cells.
#
# DECODE SAFETY: generate_raw_answer()/generate_answer_trajectory() are hand-written
#   single-token loops (prefill once with use_cache=True, then feed input_ids=[[next_id]] with
#   past_key_values). They never call HF generate(), so the fla 0.5.0 + transformers 5.3.0
#   recurrent-cache bug (which needs generate()'s _cache_dependant_input_preparation path)
#   cannot reach this family. That is why there is no --decode flag here: the decode is correct
#   by construction, the same property --decode loop buys inside run_open_context_bench.py.
#   Verified by reading the source, not assumed.
#
# TOKEN LENGTHS (measured with the rwkv7 tokenizer on the built inputs, n=100):
#   qa1/64k: min 47072  median 63043  mean 62226  max 64405
#   qa2/64k: min 55189  median 63065  mean 62242  max 64462
#   No truncation anywhere in this harness (tokenizer_any(prompt) has no max_length), and all
#   three models are recurrent with no positional ceiling, so the full 64k is actually consumed.
#
# Usage: run_babilong_64k.sh probe   -> 1 arm x 8 rows, for the cost check
#        run_babilong_64k.sh full    -> 3 arms x 100 rows qa1 (the three table cells)
set -euo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
cd "$REPO"

export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE="${1:-probe}"
TASKDIR="$REPO/task_data/babilong64k"
LOGDIR="$REPO/logs/qz"
mkdir -p "$LOGDIR"

# Inputs must already exist (built on the login node from the offline snapshot; compute has no net).
for t in qa1 qa2; do
  [ -s "$TASKDIR/babilong_$t.jsonl" ] || { echo "FATAL: missing $TASKDIR/babilong_$t.jsonl"; exit 1; }
done

RAW_CKPT=$ROOT/research/DiffRwkv/outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000
GLOBAL_CKPT=$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000
CHAMP_CKPT=$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000

# Generation args copied verbatim from the published run_summary.json files of the 4k/8k/16k cells.
GEN="--steps 100 --cfg_scale 3.0 --max_new_tokens 32 --temperature 0.0 --top_k 50 --top_p 0.9 --repetition_penalty 1.1 --seed 42 --device cuda"

run_raw () {   # frozen RWKV-7 through the write-in path, no plan (the paper's control row)
  local n="$1" out="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py \
    --ckpt_dir "$RAW_CKPT" --task_data_dir "$TASKDIR" --tasks babilong_qa1 \
    --method raw --max_samples "$n" --batch_size 1 $GEN --output_dir "$out"
}
run_global () {  # \method{} Global = prefix/suffix CFG, single global latent
  # NOTE: this script has no --batch_size (unlike the trajectory one); it is single-row only.
  local n="$1" out="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_cola_dlm_tasks_prefix_suffix_cfg.py \
    --ckpt_dir "$GLOBAL_CKPT" --task_data_dir "$TASKDIR" --tasks babilong_qa1 \
    --method prefix_cfg --max_samples "$n" $GEN --output_dir "$out"
}
run_champ () {   # \method{} Chunked = trajectory CFG; blend/horizon/chunk from the published summary
  local n="$1" out="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py \
    --ckpt_dir "$CHAMP_CKPT" --task_data_dir "$TASKDIR" --tasks babilong_qa1 \
    --method traj_cfg --max_samples "$n" --batch_size 1 \
    --trajectory_state_blend 0.7 --cond_boundary_scale 1.0 $GEN --output_dir "$out"
}

if [ "$MODE" = "probe" ]; then
  # Cost check before committing to three cells, per instruction. 8 rows, one arm.
  # NOTE: /usr/bin/time does not exist in the relay2:v2 image -- use SECONDS + torch's own
  # peak-memory report instead (the harness prints prompt_tokens per row).
  echo "[probe] raw arm, 8 rows at ~62k tokens; timing + peak memory"
  start=$SECONDS
  run_raw 8 "$REPO/results/benchmarks/probe64k/raw" 0 > "$LOGDIR/b64k-probe-raw.log" 2>&1 || echo "[probe] arm exited nonzero"
  echo "[probe] elapsed_seconds=$(( SECONDS - start )) for 8 rows"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
  tail -40 "$LOGDIR/b64k-probe-raw.log"
else
  # One arm per GPU; all three in parallel so wall clock = the slowest arm, not the sum.
  run_raw    100 "$REPO/results/benchmarks/raw-rwkv-genthenmatch/babilong64k" 0 > "$LOGDIR/b64k-raw.log"    2>&1 &
  run_global 100 "$REPO/results/benchmarks/global-dit-50k/babilong64k"        1 > "$LOGDIR/b64k-global.log" 2>&1 &
  run_champ  100 "$REPO/results/benchmarks/champ-dynamic/babilong64k"         2 > "$LOGDIR/b64k-champ.log"  2>&1 &
  wait
  echo "[done] all three arms finished; score with score_gen_match.py"
fi

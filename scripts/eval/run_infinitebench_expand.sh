#!/usr/bin/env bash
# InfiniteBench expansion: the 7 tasks the suite has never run.
#
# Priority order is scientific, not alphabetical. `passkey` and `number_string` are
# structurally identical to the ONE saturated cell our method has -- RULER niah_single_1,
# 100.0 at 4k and 16k in all five depth quintiles, "locate one salient anomaly in a uniform
# background" -- but at 125k tokens instead of 16k. They run first and on their own GPUs. The
# other five (longdialogue_qa_eng, longbook_sum_eng, math_calc, code_run, longbook_qa_chn)
# are secondary and share the remaining GPUs.
#
# ---------------------------------------------------------------------------------------
# THREE MEASURED DATA FACTS THAT DETERMINE THIS PROTOCOL (verified on the cached snapshot;
# none of them is inferred)
# ---------------------------------------------------------------------------------------
# 1. passkey/number_string are SORTED BY NEEDLE DEPTH. 590 rows = 59 depth bins x 10 rows;
#    every bin holds one identical depth and the bins strictly increase (checked all 590
#    rows of both files). So `--max_samples 50` -- the protocol the existing 5 cells use --
#    would take the 5 SHALLOWEST bins, needle depth 0.000-0.069: the easiest 8.5% of the
#    benchmark, with the key always inside the first ~8.6k of 125k tokens. A 100.0 there
#    would be worthless as evidence and would look exactly like the win we are hoping for.
#    Hence `--sample_stride 10 --max_samples 59`, which walks all 59 depths, 0.000-0.9999.
#    This is a DEVIATION from the n=50 protocol and is deliberate; n=59 also happens to
#    exceed 50, so it is not a cost saving.
# 2. The context is 125,299 tokens (relay/RWKV tokenizer; 125,307 llama, 125,313 qwen).
#    A 32768 budget would middle-truncate to head 16384 + tail 16384 and the needle would be
#    ABSENT from the input on 43 of the 59 sampled rows -- the task becomes unanswerable by
#    construction and the cell would measure our truncation, not the model. Budget is
#    therefore 131072, which holds all 125.3k tokens: seen_fraction 1.000, needle present on
#    59/59 rows. This is the only budget at which the experiment's question can be asked.
# 3. Baseline positional windows decide who can even be evaluated. At budget 131072:
#      llama-3.2-3b  131072 window -> 130944 budget -> seen 1.000, needle 59/59  COMPARABLE
#      qwen2.5-1.5b  131072 window -> 130944 budget -> seen 1.000, needle 59/59  COMPARABLE
#      qwen2.5-3b     32768 window ->  32640 budget -> seen 0.260, needle 16/59  CEILING .271
#      gemma-2-2b      8192 window ->   8064 budget -> seen 0.064, needle  4/59  CEILING .068
#      phi-2           2048 window ->   1920 budget -> seen 0.015, needle  2/59  CEILING .034
#    The honest headline is NOT "we beat the baselines at 125k". Two windowed baselines reach
#    this length like-for-like and are run here for exactly that reason; the other three are
#    capped by their own architecture and their scores are bounded by arithmetic before any
#    model runs. Those ceilings are reported next to any number we quote.
#
# NOTE ON max_new_tokens: kept at 128 to match the existing 5 cells. Official InfiniteBench
# uses 6 for passkey and 12 for number_string -- far tighter -- so 128 is generous and cannot
# handicap us. It DOES cap math_calc, whose gold is a 14,999-number list against an official
# 30000-token budget: the ordered-prefix ceiling under 128 tokens is ~0.004. math_calc is
# reported as cap-bounded, not as a capability measurement.
set -euo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Our own staging directory. The canonical longctx_fix/ and longctx_full/ files belong to
# other lines (the existing-5-task InfiniteBench cells and the LongBench re-run) and are not
# touched here.
OUT=$REPO/results/benchmarks/infbench_expand
LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

DATA=$ROOT/hf_cache/hub/datasets--xinrongzhang2022--InfiniteBench/snapshots/90f0394333616266d9fe85824ceaf505093cbaa5
NEEDLE=passkey,number_string
SECONDARY=longdialogue_qa_eng,longbook_sum_eng,math_calc,code_run,longbook_qa_chn

# Budget 131072 for every cell so the arms and the two 131072-window baselines are
# protocol-matched. positional_window() clamps the narrow-window baselines automatically and
# records the clamp in the result JSON `config`.
NEEDLE_COMMON="--suite infinitebench --datasets $NEEDLE --data_dir $DATA \
  --sample_stride 10 --max_samples 59 --max_input_tokens 131072 --max_new_tokens 128 \
  --seed 42 --device cuda"
# The secondary five are NOT depth-sorted (verified: context length is not ascending in any
# of the five), so the head-50 protocol of the existing cells applies unchanged.
SECOND_COMMON="--suite infinitebench --datasets $SECONDARY --data_dir $DATA \
  --max_samples 50 --max_input_tokens 131072 --max_new_tokens 128 --seed 42 --device cuda"

# --decode loop for every fla recurrent baseline. Measured (see greedy_loop's docstring):
# rwkv7 scores unpatched 0% / patched-generate 5% / loop 40% on the same prompts, so a
# `generate` number for these models is a harness artefact, not a model result.
run_hf () {   # name, model_dir, gpu, common, tag, extra
  local name=$1 dir=$2 gpu=$3 common=$4 tag=$5; shift 5
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_open_context_bench.py $common "$@" \
    --model_dir "$dir" --arm raw \
    --output "$OUT/${name}__${tag}__infinitebench.json" \
    > "$LOGS/ibexp-${name}-${tag}.log" 2>&1 &
  echo "  [gpu$gpu] $name ($tag) -> $LOGS/ibexp-${name}-${tag}.log"
}

# champ-dynamic-30k is trajectory_mode=True  -> --arm dynamic (LACESRuntime, chunked)
# global-dit-50k    is trajectory_mode=False -> --arm global  (GlobalRuntime, single-z) and
# NEEDS --expected_step 50000 or it dies at the step gate (default 30000).
# Crossing the arms yields ~1% garbage; they are not interchangeable.
run_relay () {  # name, ckpt_dir, arm, expected_step, gpu, common, tag
  local name=$1 ckpt=$2 arm=$3 step=$4 gpu=$5 common=$6 tag=$7
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_open_context_bench.py $common \
    --ckpt_dir "$ckpt" --arm "$arm" --expected_step "$step" \
    --plan_steps 100 --cfg_scale 2.0 --blend 0.7 --diffusion_sampler ddpm \
    --output "$OUT/${name}__${tag}__infinitebench.json" \
    > "$LOGS/ibexp-${name}-${tag}.log" 2>&1 &
  echo "  [gpu$gpu] $name ($arm, $tag) -> $LOGS/ibexp-${name}-${tag}.log"
}

CHAMP=$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000
GLOBAL=$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000
GLA=$ROOT/hf_cache/hub/models--fla-hub--gla-1.3B-100B/snapshots/46b15820a4df269e99aed9d709e017677c15d24b

# Every process writes its own file, so a shard that dies costs one cell, not the run.
# run_open_context_bench writes its result ONCE, after the loop -- there is no incremental
# checkpoint -- so a shard that overruns the job's time cap yields nothing at all. The
# projection below is from MEASURED rates in results/benchmarks/longctx_full at this budget
# (relay 15.7 s/row, rwkv7 10.7, gla 7.0): longest shard is the 5-task secondary at
# 5 x 50 x 15.7s = 1.1 h, well inside the 12 h cap.
echo "=== PRIORITY 1: passkey + number_string, all 59 depths, budget 131072 ==="
run_relay champ-dynamic-30k "$CHAMP"  dynamic 30000 0 "$NEEDLE_COMMON" needle
run_relay global-dit-50k    "$GLOBAL" global  50000 1 "$NEEDLE_COMMON" needle
run_hf    rwkv7-world3  "$ROOT/models/RWKV7-Goose-World3-2.9B-HF" 2 "$NEEDLE_COMMON" needle --decode loop
run_hf    gla-1.3b      "$GLA"                                    3 "$NEEDLE_COMMON" needle --decode loop
# The two baselines whose 131072 window reaches this length like-for-like. These are the only
# honest head-to-head comparisons available at 125k.
run_hf    llama-3.2-3b  "$ROOT/models/Llama-3.2-3B"  4 "$NEEDLE_COMMON" needle
run_hf    qwen2.5-1.5b  "$ROOT/models/qwen2.5-1.5b"  5 "$NEEDLE_COMMON" needle
# One window-limited baseline, run to MEASURE the clamp rather than assert it: qwen2.5-3b's
# 32768 ceiling bounds it at 0.271 on this sample by arithmetic, and the run confirms the
# harness reports that bound instead of silently scoring it as a capability.
run_hf    qwen2.5-3b    "$ROOT/models/Qwen2.5-3B"    6 "$NEEDLE_COMMON" needle
echo "=== PRIORITY 2: the other five tasks, our arms (GPU7 shared, sequential) ==="
(
  CUDA_VISIBLE_DEVICES=7 python3 scripts/eval/run_open_context_bench.py $SECOND_COMMON \
    --ckpt_dir "$CHAMP" --arm dynamic --expected_step 30000 \
    --plan_steps 100 --cfg_scale 2.0 --blend 0.7 --diffusion_sampler ddpm \
    --output "$OUT/champ-dynamic-30k__secondary__infinitebench.json" \
    > "$LOGS/ibexp-champ-dynamic-30k-secondary.log" 2>&1
  CUDA_VISIBLE_DEVICES=7 python3 scripts/eval/run_open_context_bench.py $SECOND_COMMON \
    --ckpt_dir "$GLOBAL" --arm global --expected_step 50000 \
    --plan_steps 100 --cfg_scale 2.0 --blend 0.7 --diffusion_sampler ddpm \
    --output "$OUT/global-dit-50k__secondary__infinitebench.json" \
    > "$LOGS/ibexp-global-dit-50k-secondary.log" 2>&1
) &
echo "  [gpu7] champ-dynamic-30k then global-dit-50k (secondary 5) -> $LOGS/ibexp-*-secondary.log"
wait
echo "=== ALL SHARDS DONE ==="

echo "=== effective budgets actually used (must be 131072 for the recurrent models) ==="
grep -h "^\[budget\]" "$LOGS"/ibexp-*.log || true
echo "=== fla patch status (True expected for rwkv7 and gla) ==="
grep -H "fla recurrent-generate patch applied" "$LOGS"/ibexp-*.log || true
echo "=== files written ==="
ls -la "$OUT" || true

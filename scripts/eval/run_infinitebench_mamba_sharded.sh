#!/usr/bin/env bash
# mamba-2.8b on InfiniteBench, one dataset per GPU, because it cannot finish serially.
#
# Why this job exists. This image has no mamba fast kernels: mamba_ssm, causal_conv1d and
# selective_scan_cuda are all absent, and transformers' mambapy `pscan` fallback is gated on
# `self.training` (modeling_mamba.py:431) so it never engages at eval. mamba therefore runs a
# Python sequential scan over a 32768-token prompt at ~3 min/row. 250 rows serially is ~12.9 h
# against the 12 h job cap -- and run_open_context_bench.py writes its JSON exactly once, at
# the end (single `output_path.write_text` call), so a run that overruns the cap produces
# NOTHING. That is why the main sweep's mamba cell would otherwise be blank rather than partial.
#
# The five datasets are independent, so sharding them one-per-GPU turns ~12.9 h of serial work
# into ~2.6 h wall-clock (50 rows x ~3 min) and every shard writes its own complete file.
# summarize_infinitebench.py is pointed at the merged output; merge_mamba_shards.py stitches
# the five shard files into one mamba-2.8b__infinitebench.json with the same schema.
#
# Everything else is byte-identical to wave 1: same datasets, same 50 samples, same 32768 /
# 128 token budgets, same seed, same data snapshot. Only the work split changes, so the numbers
# remain comparable to every other model in the table.
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

OUT=$REPO/results/benchmarks/longctx_mamba_shards
LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

DATA=$ROOT/hf_cache/hub/datasets--xinrongzhang2022--InfiniteBench/snapshots/90f0394333616266d9fe85824ceaf505093cbaa5
COMMON="--suite infinitebench --data_dir $DATA \
  --max_samples 50 --max_input_tokens 32768 --max_new_tokens 128 --seed 42 --device cuda"

gpu=0
for ds in longbook_qa_eng longbook_choice_eng kv_retrieval code_debug math_find; do
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_open_context_bench.py $COMMON \
    --datasets "$ds" \
    --model_dir "$ROOT/models/mamba-2.8b" --arm raw \
    --output "$OUT/mamba-2.8b__${ds}__infinitebench.json" \
    > "$LOGS/ibmamba-${ds}-infinitebench.log" 2>&1 &
  echo "  [gpu$gpu] mamba-2.8b / $ds -> $LOGS/ibmamba-${ds}-infinitebench.log"
  gpu=$((gpu + 1))
done
wait

echo "=== shards written ==="
ls -la "$OUT"
echo "=== budgets (all must be 32768 / cli_cap_only: mamba is recurrent) ==="
grep -h "^\[budget\]" "$LOGS"/ibmamba-*-infinitebench.log || true
echo "=== OOM check ==="
grep -l -E "CUDA out of memory|OutOfMemoryError" "$LOGS"/ibmamba-*.log || echo "no OOM"
echo "=== ALL DONE ==="

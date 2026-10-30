#!/usr/bin/env bash
# mamba-2.8b on RULER, one task per GPU. Fills the last genuinely unowned cell of
# tab:longcontext-suites.
#
# WHY SHARDED. This image has no mamba fast kernels (mamba_ssm / causal_conv1d /
# selective_scan_cuda all absent, and transformers' mambapy pscan fallback is gated on
# `self.training` so it never engages at eval), so mamba runs a Python sequential scan.
# Measured on this exact model/GPU: 23.5 s/row at 4096 tokens (stanza 32 of job-03d6d491
# managed 133 rows in 3160 s of dedicated 1-GPU time before that job was stopped), which
# cross-checks against its bAbILong run on the same GPU (7894 s / 200 rows at mean 7554
# tokens = 21.4 s/row scaled to 4096). The matched protocol is 6 tasks at 4096 + 3 at 16384
# with 200 rows each = 1800 rows ~ 23.5 GPU-h serially, and run_open_context_bench.py writes
# its JSON exactly once at the very end, so a serial run that overruns the cap yields NOTHING.
# One task per GPU makes the wall clock the slowest single task (a 16384 one, ~5.2 h) and
# every shard writes a complete file of its own.
#
# DECODE: mamba is NOT an fla recurrent model (transformers-native MambaForCausalLM; the loader
# prints "model_type='mamba' is native to transformers; fla not imported" and "fla
# recurrent-generate patch applied: False"). The fla 0.5.0 + transformers 5.3.0 cache bug
# therefore cannot affect it, so the hf default decode is correct here. Only rwkv7/gla/
# gated_deltanet need --decode loop. Verified from the partial run's own log, not assumed.
#
# BUDGET: the partial log's budget line reads
#   effective_input_budget 16384, truncation_reason "cli_cap_only (recurrent model: no
#   positional ceiling)"
# so there is no positional clamp to misreport: mamba sees the full 16384.
#
# PROTOCOL: identical to the 8 models already in the table -- same 9 task/length cells, same
# --max_samples 200, --max_input_tokens 16384, --max_new_tokens 128, --seed 42, same snapshot.
# Only the work split differs, so the numbers stay comparable.
set -euo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

OUT=$REPO/results/benchmarks/longctx_mamba_ruler_shards
LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

DATA=$ROOT/hf_cache/hub/datasets--simonjegou--ruler/snapshots/24adceac8a0e6532936e8d721cd9e9084d2e4686
MODEL=$ROOT/models/mamba-2.8b
COMMON="--suite ruler --data_dir $DATA --model_dir $MODEL --arm raw \
  --max_samples 200 --max_input_tokens 16384 --max_new_tokens 128 --seed 42 --device cuda"

# 9 cells, 8 GPUs: the three 16384 cells are ~4x the cost of a 4096 cell, so give each of them
# its own lane and pair two cheap 4096 cells on the last lane.
run_cell () {   # $1=dataset spec  $2=gpu  $3=tag
  local ds="$1" gpu="$2" tag="$3"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_open_context_bench.py $COMMON \
    --datasets "$ds" --output "$OUT/mamba-2.8b__${tag}__ruler.json" \
    > "$LOGS/rulermamba-${tag}.log" 2>&1
  echo "lane$gpu $tag exit=$?"
}

run_cell 16384/niah_single_1 0 16384-niah_single_1 &
run_cell 16384/vt            1 16384-vt            &
run_cell 16384/cwe           2 16384-cwe           &
run_cell 4096/niah_single_1  3 4096-niah_single_1  &
run_cell 4096/niah_multikey_1 4 4096-niah_multikey_1 &
run_cell 4096/vt             5 4096-vt             &
run_cell 4096/cwe            6 4096-cwe            &
run_cell 4096/fwe,4096/qa_1  7 4096-fwe_qa1        &
wait

echo "=== all lanes done; shard files ==="
ls -l "$OUT"
echo "=== merge (fatal on protocol disagreement) ==="
python3 scripts/eval/merge_mamba_shards.py \
  --shard_dir "$OUT" \
  --glob 'mamba-2.8b__*__ruler.json' \
  --output "$REPO/results/benchmarks/longctx_fix/mamba-2.8b__ruler.json"

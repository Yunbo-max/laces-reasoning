set -euo pipefail
ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
OUT=$REPO/results/benchmarks/longctx_bos; LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"; cd "$REPO"
DATA=$ROOT/hf_cache/hub/datasets--xinrongzhang2022--InfiniteBench/snapshots/90f0394333616266d9fe85824ceaf505093cbaa5
# kv_retrieval is the only column with no guessable null and clean golds; longbook_choice_eng is
# the one choice task whose slice is representative. 50 samples, identical to the main table.
COMMON="--suite infinitebench --datasets kv_retrieval,longbook_choice_eng --data_dir $DATA \
  --max_samples 50 --max_input_tokens 32768 --max_new_tokens 128 --seed 42 --device cuda"
run () { CUDA_VISIBLE_DEVICES=$3 python3 scripts/eval/run_open_context_bench.py $COMMON \
    --model_dir "$2" --arm raw --output "$OUT/$1__infinitebench.json" \
    > "$LOGS/ibbos-$1-infinitebench.log" 2>&1 & echo "  [gpu$3] $1"; }
run gemma-2-2b   "$ROOT/models/gemma-2-2b" 0
run gla-1.3b     "$ROOT/hf_cache/hub/models--fla-hub--gla-1.3B-100B/snapshots/46b15820a4df269e99aed9d709e017677c15d24b" 1
run llama-3.2-3b "$ROOT/models/Llama-3.2-3B" 2
wait
grep -h "^\[budget\]" "$LOGS"/ibbos-*.log || true
echo "=== ALL DONE ==="

#!/usr/bin/env bash
# Wave 2 only: our two relay arms, submitted as their own job so they do not wait behind
# mamba-2.8b (which runs the sequential fallback kernel at ~1 row/min => ~7.5h for 450 rows).
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

DATASETS=hotpotqa,2wikimqa,musique,narrativeqa,multifieldqa_en,passage_retrieval_en,gov_report,qmsum,multi_news
COMMON="--suite longbench --datasets $DATASETS \
  --data_dir $ROOT/longbench_local/extract/data \
  --longbench_config_dir $ROOT/longbench_local/pinned_spec \
  --max_samples 50 --max_input_tokens 32768 --max_new_tokens 512 --seed 42 --device cuda \
  --plan_steps 100 --cfg_scale 2.0 --blend 0.7 --diffusion_sampler ddpm"

# champ-dynamic-30k: trajectory_mode=True  -> --arm dynamic (chunked, LACESRuntime)
# global-dit-50k:    trajectory_mode=False -> --arm global  (single-z, GlobalRuntime)
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/run_open_context_bench.py $COMMON \
  --ckpt_dir "$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000" \
  --arm dynamic --expected_step 30000 \
  --output "$OUT/champ-dynamic-30k__longbench.json" \
  > "$LOGS/lbfix-champ-dynamic-30k-longbench.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 scripts/eval/run_open_context_bench.py $COMMON \
  --ckpt_dir "$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000" \
  --arm global --expected_step 50000 \
  --output "$OUT/global-dit-50k__longbench.json" \
  > "$LOGS/lbfix-global-dit-50k-longbench.log" 2>&1 &

wait
echo "=== OUR ARMS DONE ==="
grep -h "^\[budget\]" "$LOGS"/lbfix-champ-dynamic-30k-longbench.log "$LOGS"/lbfix-global-dit-50k-longbench.log || true

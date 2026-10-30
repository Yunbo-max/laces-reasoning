#!/bin/bash
# MemEval (LongMemEval oracle) re-run across the paper's selected model set.
# One GPU per model, 10 models over 8 GPUs (two GPUs take a second model).
# Fixes relied on: fixed longmemeval_score_approx (answer-span + binary-choice commitment),
# positional_window/effective_input_budget (phi-2's 2048 window), patch_fla_recurrent_generate
# (gla/rwkv7 fla+transformers generate corruption).
set -uo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning
export HF_HOME=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/inspire/hdd/global_user/zhangjiaquan-253108540222/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

DATA=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/datasets--xiaowu0162--longmemeval/snapshots/2ec2a557f339b6c0369619b1ed5793734cc87533
OUT=$REPO/results/benchmarks/longctx
LOGD=$REPO/logs/qz
mkdir -p "$OUT" "$LOGD"
cd "$REPO"

M=/inspire/hdd/global_user/zhangjiaquan-253108540222/models
CK=$REPO/outputs_dynamic_basis

# name|gpu|kind|path|arm|expected_step
JOBS=(
  "qwen2.5-3b|0|hf|$M/Qwen2.5-3B|raw|30000"
  "qwen2.5-1.5b|1|hf|$M/qwen2.5-1.5b|raw|30000"
  "llama-3.2-3b|2|hf|$M/Llama-3.2-3B|raw|30000"
  "phi-2|3|hf|$M/phi-2|raw|30000"
  "gemma-2-2b|4|hf|$M/gemma-2-2b|raw|30000"
  "gla-1.3b|5|hf|/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/models--fla-hub--gla-1.3B-100B/snapshots/46b15820a4df269e99aed9d709e017677c15d24b|raw|30000"
  "rwkv7-world3|6|hf|$M/RWKV7-Goose-World3-2.9B-HF|raw|30000"
  "mamba-2.8b|7|hf|$M/mamba-2.8b|raw|30000"
  "champ-dynamic-30k|0|ckpt|$CK/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000|dynamic|30000"
  "global-dit-50k|1|ckpt|$CK/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000|global|50000"
)

run_one() {
  local name=$1 gpu=$2 kind=$3 path=$4 arm=$5 step=$6
  local flag
  if [ "$kind" = "hf" ]; then flag="--model_dir $path"; else flag="--ckpt_dir $path"; fi
  echo "[launch] $name on GPU $gpu ($kind, arm=$arm)"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/eval/run_open_context_bench.py \
      --suite memeval \
      --datasets longmemeval_oracle \
      --data_dir "$DATA" \
      $flag --arm "$arm" \
      --output "$OUT/${name}__memeval.json" \
      --max_samples 50 \
      --max_input_tokens 131072 \
      --max_new_tokens 128 \
      --expected_step "$step" \
      --seed 42 --device cuda \
      > "$LOGD/memeval-fix-${name}.log" 2>&1
  echo "[done] $name exit=$?"
}

# Wave 1: the 8 single-GPU baselines in parallel.
for spec in "${JOBS[@]:0:8}"; do
  IFS='|' read -r n g k p a s <<< "$spec"
  run_one "$n" "$g" "$k" "$p" "$a" "$s" &
done
wait
echo "=== wave 1 (baselines) complete ==="

# Wave 2: our two relay arms (heavier; run after the baselines free their memory).
for spec in "${JOBS[@]:8:2}"; do
  IFS='|' read -r n g k p a s <<< "$spec"
  run_one "$n" "$g" "$k" "$p" "$a" "$s" &
done
wait
echo "=== wave 2 (relay arms) complete ==="

echo "=== SCORES ==="
python3 - <<'PY'
import json,glob
for f in sorted(glob.glob('/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/results/benchmarks/longctx/*__memeval.json')):
    try:
        d=json.load(open(f)); a=d['aggregates']['longmemeval_oracle']
        print('%-22s n=%d score=%.4f' % (f.split('/')[-1].replace('__memeval.json',''), a['n'], a['score']))
    except Exception as e: print(f, 'UNREADABLE', e)
PY
echo "=== ALL DONE ==="

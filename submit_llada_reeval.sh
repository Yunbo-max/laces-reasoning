#!/bin/bash
# 重新评测LLaDA模型 (forward()已修复)

set -e

TASK_DATA_DIR="baseline/Cola-DLM/generate_task_data"
OUTPUT_BASE="/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/results/difflm_reeval"
TASKS="arc_easy,arc_challenge,winogrande,sciq,commonsenseqa,piqa,mmlu,obqa,race,story_cloze,hellaswag,siqa,lambada,squad"

mkdir -p "$OUTPUT_BASE"

# 3个LLaDA模型待重评
MODELS=(
  "llada-8b:/inspire/hdd/global_user/zhangjiaquan-253108540222/models/LLaDA-8B-Base"
  "llada-moe-7b:/inspire/hdd/global_user/zhangjiaquan-253108540222/models/llada-moe-7b-a1b"
  "dream-7b:dream-7b"  # 用HF model ID
)

for model_spec in "${MODELS[@]}"; do
  IFS=':' read model_name model_path <<< "$model_spec"
  
  echo "Submitting: $model_name (14 tasks)"
  
  # 为LLaDA和Dream-7B提交完整评测
  qz train CreateJob \
    --name "difflm-${model_name}-full-reeval" \
    --command "
      cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning && \
      export HF_HUB_OFFLINE=1 && \
      python scripts/eval/eval_diffusion_lm_gen_match.py \
        --model_dir '$model_path' \
        --impl hf_diffusion \
        --task_data_dir '$TASK_DATA_DIR' \
        --output_dir '$OUTPUT_BASE' \
        --tasks $TASKS \
        --max_samples 1000
    " 2>&1 | grep -E "Created|Error|success" || true
  
  sleep 1
done

echo ""
echo "✅ LLaDA重评任务已提交"

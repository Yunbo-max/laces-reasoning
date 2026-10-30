#!/bin/bash
# 发布2.9B完整8模型矩阵的Chunked模型训练
# 4个Chunked模型: DiT/BiRWKV × Dynamic/Fixed
# 顺序提交：一个一个H100训练（每个 ~50小时）

set -e

cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning

echo "=========================================="
echo "  发布4个2.9B Chunked模型训练任务"
echo "  策略: 顺序提交（1个H100顺序训 200小时总）"
echo "=========================================="
echo ""

# 4个Chunked配置 - 顺序训练
CONFIGS=(
  "chunked-dit-dynamic|dit|dynamic"
  "chunked-dit-fixed|dit|fixed"
  "chunked-birwkv-dynamic|birwkv|dynamic"
  "chunked-birwkv-fixed|birwkv|fixed"
)

for config in "${CONFIGS[@]}"; do
  IFS='|' read model_name denoiser writer_type <<< "$config"

  echo "提交: $model_name (${denoiser} + ${writer_type})"
  echo "  预期时间: ~50小时 单块H100"
  echo ""

  # 构建输出目录名
  output_dir="outputs_${writer_type}_basis/laces-2.9B-${writer_type}-r32-chunked-${denoiser}-50k"

  # 提交训练任务 - 顺序执行
  qz train CreateJob \
    --name "train-2.9B-chunked-${model_name}-50k" \
    --spec lcg-79b2ad0e \
    --command "
      cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning && \
      export HF_HUB_OFFLINE=1 && \
      python scripts/train/train_relay_lm.py \
        --backbone rwkv7-goose-2.9b \
        --denoiser ${denoiser} \
        --writer ${writer_type} \
        --trajectory_mode chunked \
        --trajectory_geometry 64x64 \
        --data_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/fineweb_4096 \
        --output_dir ${output_dir} \
        --steps 50000 \
        --batch_size 16 \
        --learning_rate 2e-4 \
        --log_interval 100
    " 2>&1 | tail -5

  echo ""
done

echo "✅ 4个2.9B Chunked模型训练任务已全部提交（顺序执行）"
echo ""
echo "预期时间: 4个模型 × 50小时 = 200小时 wall-clock (1个H100)"
echo "预计完成: 在9月26-27号左右"
echo ""
echo "监控: qz train ListJobs --workspace-id ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"

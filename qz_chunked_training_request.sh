#!/bin/bash
# 发布4个2.9B Chunked模型训练任务到QZ集群
# 顺序执行：一个一个H100（每个50小时）

set -e

cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning

echo "=========================================="
echo "  发布4个2.9B Chunked模型训练任务"
echo "=========================================="
echo ""

# QZ 全局参数
WORKSPACE_ID="ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT_ID="project-632c8db8-4530-413a-ada5-df91774a7e09"
LOGIC_COMPUTE_GROUP="lcg-79b2ad0e"
FRAMEWORK="pytorch"
DOCKER_IMAGE="docker.sii.shaipower.online/inspire-studio/relay2:v2"
SPEC_ID="1d3cb6ac-6a44-4acf-b8de-7facaedd8330"
TASK_PRIORITY=6

# 4个Chunked配置
CONFIGS=(
  "chunked-dit-dynamic|dit|dynamic"
  "chunked-dit-fixed|dit|fixed"
  "chunked-birwkv-dynamic|birwkv|dynamic"
  "chunked-birwkv-fixed|birwkv|fixed"
)

for config in "${CONFIGS[@]}"; do
  IFS='|' read model_name denoiser writer_type <<< "$config"

  echo "提交: $model_name (${denoiser} + ${writer_type})"

  output_dir="outputs_${writer_type}_basis/laces-2.9B-${writer_type}-r32-chunked-${denoiser}-50k"

  # 使用 jq 生成正确的 JSON
  temp_request="/tmp/qz_${model_name}_request.json"

  jq -n \
    --arg cmd "cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning && export HF_HUB_OFFLINE=1 && python scripts/train/train_relay_lm.py --backbone rwkv7-goose-2.9b --denoiser ${denoiser} --writer ${writer_type} --trajectory_mode chunked --trajectory_geometry 64x64 --data_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/fineweb_4096 --output_dir ${output_dir} --steps 50000 --batch_size 16 --learning_rate 2e-4 --log_interval 100" \
    --arg spec_id "$SPEC_ID" \
    --arg img "$DOCKER_IMAGE" \
    --arg model "$model_name" \
    --arg wid "$WORKSPACE_ID" \
    --arg pid "$PROJECT_ID" \
    --arg lcg "$LOGIC_COMPUTE_GROUP" \
    '{
      "auto_fault_tolerance": true,
      "command": ("bash -lc " | @json) + ($cmd | @json),
      "description": ("Train 2.9B chunked " + $model + " to 50k steps"),
      "framework": "pytorch",
      "framework_config": [
        {
          "image": $img,
          "image_type": "SOURCE_PRIVATE",
          "instance_count": 1,
          "shm_gi": 900,
          "spec_id": $spec_id
        }
      ],
      "fault_tolerance_max_retry": 3,
      "fault_tolerance_retry_interval_sec": 300,
      "logic_compute_group_id": $lcg,
      "max_running_time_ms": "604800000",
      "name": ("train-2.9B-chunked-" + $model + "-50k"),
      "project_id": $pid,
      "task_priority": 6,
      "tb_summary_path": "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/tb_logs",
      "workspace_id": $wid
    }' > "$temp_request"

  # 提交到 QZ
  qz train CreateJob --data @"$temp_request" 2>&1 | tail -5
  echo ""

  rm -f "$temp_request"
done

echo "✅ 4个2.9B Chunked模型训练任务已全部提交（顺序执行）"
echo ""
echo "预期时间: 4个模型 × 50小时 = 200小时 wall-clock"
echo "预计完成: 在9月26-27号左右"
echo ""
echo "监控: qz train ListJobs --workspace-id ${WORKSPACE_ID}"

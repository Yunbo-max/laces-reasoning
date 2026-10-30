#!/bin/bash
# Parameter sweep for global-birwkv-50k (垃圾文本修复)
# 用法: bash submit_birwkv_param_sweep.sh
#
# 目标: 找到最优 cfg_scale × temperature 组合
# 数据: 3个smoke tasks (arc_easy, sciq, piqa)
# 总计: 20个配置 × 3任务 = 60 evals
# 预期时间: ~2小时 (1×H100)

set -e

CKPT_DIR="/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-birwkv768x8-b16-50k"
OUTPUT_BASE="/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/results/infer_sweep_birwkv"
SMOKE_TASKS="arc_easy,sciq,piqa"

mkdir -p "$OUTPUT_BASE"

CFG_SCALES=(0.5 1.0 1.5 2.0 3.0)
TEMPERATURES=(0.0 0.3 0.5 0.7)

echo "=== Submitting parameter sweep for global-birwkv-50k ==="
echo "Checkpoint: $CKPT_DIR"
echo "Output: $OUTPUT_BASE"
echo "Tasks: $SMOKE_TASKS"
echo ""

JOB_IDS=()

for cfg_scale in "${CFG_SCALES[@]}"; do
  for temperature in "${TEMPERATURES[@]}"; do
    CONFIG_NAME="cfg${cfg_scale}_temp${temperature}"
    OUTPUT_FILE="$OUTPUT_BASE/${CONFIG_NAME}.json"

    echo "Submitting: $CONFIG_NAME ..."

    # Create submit command
    CMD="
      cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning && \
      export HF_HUB_OFFLINE=1 && \
      python scripts/eval/run_cola_dlm_tasks_prefix_suffix_cfg.py \
        --ckpt_dir '$CKPT_DIR' \
        --cfg_scale $cfg_scale \
        --temperature $temperature \
        --tasks $SMOKE_TASKS \
        --output_dir '$OUTPUT_BASE' \
        --method prefix_cfg \
        --steps 100 \
        --max_new_tokens 32 \
        --seed 42
    "

    # Submit to QZ (顺序，1×H100)
    qz train CreateJob \
      --name "birwkv-param-sweep-${CONFIG_NAME}" \
      --command "$CMD" \
      2>&1 | tee -a "$OUTPUT_BASE/submit.log" || true

    echo "  ✓ Submitted"

    # 小延迟避免QZ API限流
    sleep 1
  done
done

echo ""
echo "=== All 20 jobs submitted ==="
echo "Monitor with: qz train ListJobs --workspace-id ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6 --page-size 50"
echo ""
echo "Next step: 等待2小时完成后，运行 analyze_sweep_results.py 找最优参数"

#!/bin/bash
# 综合执行计划：修复所有失败任务并优化参数
# 用户指令：一个模型一个H100，顺序提交
# 状态: 准备就绪

set -e

cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning

echo "=========================================="
echo "LACES 论文 — 失败任务修复与优化"
echo "=========================================="
echo ""
echo "✅ 已完成:"
echo "  1. LLaDA forward()修复 (HF cache两个版本)"
echo "  2. LLaDA-MOE forward()修复"
echo "  3. birwkv参数搜索脚本 (submit_birwkv_param_sweep.sh)"
echo "  4. 结果分析脚本 (analyze_sweep_results.py)"
echo ""
echo "📋 执行序列:"
echo ""
echo "[T=0h] 阶段1: 参数优化 global-birwkv-50k"
echo "  - 提交20个配置 × 3任务 smoke test"
echo "  - 预期: ~2小时完成"
echo "  - 命令: bash submit_birwkv_param_sweep.sh"
echo ""
echo "[T=2h] 分析参数搜索结果"
echo "  - 找出最优(cfg_scale, temperature)"
echo "  - 命令: python analyze_sweep_results.py"
echo ""
echo "[T=2.5h] 用最优参数重评14任务"
echo "  - 预期: ~5小时完成"
echo "  - 输出: global-birwkv-50k最终分数"
echo ""
echo "[T=8h] 阶段2: 重新评测LLaDA模型"
echo "  - LLaDA forward()已修复，可重新提交"
echo "  - 3个模型 × 14任务 = 42 evals"
echo "  - 预期: ~7小时完成"
echo ""
echo "[T=15h] 等待13.3B训练完成"
echo "  - 当前进度: 8% (ETA +14h = T+14h)"
echo ""
echo "[T=29h] 阶段3: 评测13.3B模型"
echo "  - 4个模型 × 14任务 = 56 evals"
echo "  - 预期: ~7小时"
echo ""
echo "[T=36h] 汇总所有结果到论文表格"
echo "  - Fill main table (Fix2结果 + birwkv优化 + 13.3B scaling)"
echo "  - Fill appendix (PPL表格待traj_cfg)"
echo ""
echo "=========================================="
echo ""
echo "🚀 现在启动阶段1?"
read -p "输入 'yes' 继续: " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "启动参数搜索..."
bash submit_birwkv_param_sweep.sh

echo ""
echo "✅ 所有任务已提交"
echo "下一步: 等待2小时，然后运行 python analyze_sweep_results.py"

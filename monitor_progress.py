#!/usr/bin/env python3
"""
监控和汇总LACES实验进度
"""
import subprocess
import json
from pathlib import Path
from datetime import datetime

def check_sweep_results():
    """检查参数扫描结果"""
    sweep_dir = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/results/infer_sweep_birwkv")

    if not sweep_dir.exists():
        return {"status": "pending", "count": 0}

    json_files = list(sweep_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name not in ["submit.log", "analysis.json"]]

    return {"status": "in_progress", "count": len(json_files), "dir": str(sweep_dir)}

def check_llada_results():
    """检查LLaDA重评结果"""
    output_dir = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/results/difflm_reeval")

    if not output_dir.exists():
        return {"status": "pending", "count": 0}

    result_files = list(output_dir.glob("*.jsonl"))
    return {"status": "in_progress" if result_files else "pending", "count": len(result_files)}

def main():
    print("\n" + "="*80)
    print("  LaLaLanD ICLR-27 实验进度监控")
    print("="*80)
    print(f"  时间: {datetime.now().isoformat()}")
    print("="*80 + "\n")

    print("📊 实验阶段状态:\n")

    print("🔵 阶段1: global-birwkv-50k 参数搜索")
    sweep_status = check_sweep_results()
    print(f"   状态: {sweep_status['status']}")
    if sweep_status['count'] > 0:
        print(f"   完成: {sweep_status['count']}/20 配置")
    else:
        print("   提交时间: 刚刚 (~等待2小时)")
    print()

    print("🟢 阶段2: LLaDA模型重评 (forward()已修复)")
    llada_status = check_llada_results()
    print(f"   状态: {llada_status['status']}")
    if llada_status['count'] > 0:
        print(f"   完成: {llada_status['count']}/42 任务")
    else:
        print("   提交时间: 刚刚")
    print()

    print("🟡 阶段3: 13.3B模型训练 (进度8%)")
    print("   ETA: +14小时 (明天~1400 UTC)")
    print()

    print("="*80)
    print("  下一步操作")
    print("="*80)
    print("""
1. 等待2小时，参数扫描完成后:
   python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/analyze_sweep_results.py

2. 用最优参数重评14任务 (准备后续提交)

3. 监控任务进度:
   qz train ListJobs --workspace-id ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6 --page-size 20

4. 查看具体任务日志:
   qz train GetJobLog --workspace-id ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6 --job-id <job-id>
    """)

if __name__ == "__main__":
    main()

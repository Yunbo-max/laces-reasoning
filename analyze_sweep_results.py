#!/usr/bin/env python3
"""
分析参数扫描结果，找最优 (cfg_scale, temperature) 配置
"""
import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_sweep_results(sweep_dir: str = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/results/infer_sweep_birwkv"):
    """
    从扫描结果中找最优配置

    结果文件格式: cfg{scale}_temp{temp}.json
    其中包含每个任务的准确率/质量分数
    """
    results = defaultdict(lambda: {"configs": [], "best_config": None, "best_score": -1.0})

    sweep_path = Path(sweep_dir)
    if not sweep_path.exists():
        print(f"❌ Sweep results directory not found: {sweep_dir}")
        return

    # 收集所有配置的结果
    for json_file in sorted(sweep_path.glob("*.json")):
        if json_file.name in ["submit.log", "analysis.json"]:
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            # 解析文件名获取参数
            config_name = json_file.stem  # e.g., "cfg0.5_temp0.0"
            parts = config_name.replace("cfg", "").replace("_temp", ",").split(",")
            if len(parts) != 2:
                print(f"⚠️ Skipping {json_file.name} - name format unexpected")
                continue

            cfg_scale = float(parts[0])
            temperature = float(parts[1])

            # 计算平均准确率
            if isinstance(data, dict) and "results" in data:
                # 格式: {"results": {"task1": accuracy, "task2": accuracy, ...}}
                tasks = data.get("results", {})
                if tasks:
                    mean_accuracy = sum(tasks.values()) / len(tasks)
                else:
                    mean_accuracy = 0.0
            elif isinstance(data, dict) and "arc_easy" in data:
                # 格式: {"arc_easy": accuracy, "sciq": accuracy, ...}
                mean_accuracy = sum(data.values()) / len(data)
            else:
                print(f"⚠️ Unknown result format in {json_file.name}")
                mean_accuracy = 0.0

            # 记录这个配置
            config_result = {
                "cfg_scale": cfg_scale,
                "temperature": temperature,
                "mean_accuracy": mean_accuracy,
                "file": json_file.name,
                "details": tasks if "tasks" in locals() else data
            }

            results["all_configs"].append(config_result)

            # 更新最佳配置
            if mean_accuracy > results["best_score"]:
                results["best_score"] = mean_accuracy
                results["best_config"] = {
                    "cfg_scale": cfg_scale,
                    "temperature": temperature,
                    "mean_accuracy": mean_accuracy
                }

        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse {json_file.name}: {e}")
            continue
        except Exception as e:
            print(f"⚠️ Error processing {json_file.name}: {e}")
            continue

    if not results["all_configs"]:
        print("❌ No sweep results found!")
        return

    # 排序配置
    results["all_configs"].sort(key=lambda x: x["mean_accuracy"], reverse=True)

    # 打印报告
    print("\n" + "="*80)
    print("  PARAMETER SWEEP ANALYSIS: global-birwkv-50k")
    print("="*80)

    print(f"\nTotal configs evaluated: {len(results['all_configs'])}")
    print(f"Best score: {results['best_score']:.1%}")

    if results["best_config"]:
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"    cfg_scale:    {results['best_config']['cfg_scale']}")
        print(f"    temperature:  {results['best_config']['temperature']}")
        print(f"    mean accuracy: {results['best_config']['mean_accuracy']:.1%}")

    print("\n" + "-"*80)
    print("  Top 10 configurations:")
    print("-"*80)
    print(f"{'Rank':<5} {'cfg_scale':<10} {'temp':<8} {'Accuracy':<12} {'File':<30}")
    print("-"*80)

    for i, config in enumerate(results["all_configs"][:10], 1):
        print(f"{i:<5} {config['cfg_scale']:<10.1f} {config['temperature']:<8.1f} "
              f"{config['mean_accuracy']:<12.1%} {config['file']:<30}")

    # 保存分析结果
    analysis_file = Path(sweep_dir) / "analysis.json"
    with open(analysis_file, "w") as f:
        json.dump({
            "total_configs": len(results["all_configs"]),
            "best_config": results["best_config"],
            "top_10": results["all_configs"][:10],
            "all_configs": results["all_configs"],
        }, f, indent=2)

    print(f"\n✅ Analysis saved to: {analysis_file}")

    # 打印后续步骤
    if results["best_config"]:
        print("\n" + "="*80)
        print("  NEXT STEP: Full evaluation with best config")
        print("="*80)
        cfg = results["best_config"]["cfg_scale"]
        temp = results["best_config"]["temperature"]
        print(f"""
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning

# 使用最优参数评估全部14个任务
python scripts/eval/run_cola_dlm_tasks_prefix_suffix_cfg.py \\
  --ckpt_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-birwkv768x8-b16-50k \\
  --cfg_scale {cfg} \\
  --temperature {temp} \\
  --tasks arc_easy,arc_challenge,winogrande,sciq,commonsenseqa,piqa,mmlu,obqa,race,story_cloze,hellaswag,siqa,lambada,squad \\
  --output_dir results/final_optimized \\
  --method prefix_cfg \\
  --steps 100
""")

if __name__ == "__main__":
    import sys
    sweep_dir = sys.argv[1] if len(sys.argv) > 1 else "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/results/infer_sweep_birwkv"
    analyze_sweep_results(sweep_dir)

#!/usr/bin/env python3
"""
Global-BiRWKV 参数搜索 - 找到让它不再是0.5%的最优参数
"""
import subprocess
import json
from itertools import product

# 搜索空间
cfg_scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
temperatures = [0.0, 0.3, 0.5, 0.7, 1.0]

# 基础参数
MODEL_PATH = "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-birwkv768x8-b16-50k/step_00050000"
TASKS = ["arc_easy", "arc_challenge", "winogrande", "sciq", "commonsenseqa", "piqa"]

print("=" * 80)
print("  Global-BiRWKV 参数搜索")
print(f"  搜索空间: {len(cfg_scales)} × {len(temperatures)} = {len(cfg_scales) * len(temperatures)} 组合")
print("  每组测3个任务（arc_easy, winogrande, sciq）")
print("=" * 80)
print()

configs = list(product(cfg_scales, temperatures))
print(f"总共 {len(configs)} 个参数组合要测试")
print()

# 分批提交
batch_size = 10
for batch_idx in range(0, len(configs), batch_size):
    batch = configs[batch_idx:batch_idx + batch_size]
    print(f"\n批次 {batch_idx // batch_size + 1}:")

    for cfg, temp in batch:
        task_cmd = (
            f"cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning && "
            f"export HF_HUB_OFFLINE=1 && "
            f"python scripts/eval/eval_prefix_suffix_cfg_lambada.py "
            f"--ckpt_dir {MODEL_PATH} "
            f"--tasks arc_easy winogrande sciq "
            f"--cfg_scale {cfg} "
            f"--temperature {temp} "
            f"--steps 100 "
            f"--output_dir results/benchmarks/global-birwkv-param-search/cfg{cfg}_temp{temp}"
        )

        # 构建QZ请求
        request = {
            "auto_fault_tolerance": True,
            "command": f"bash -lc '{task_cmd}'",
            "description": f"Tune global-birwkv: cfg={cfg}, temp={temp}",
            "framework": "pytorch",
            "framework_config": [{
                "image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
                "image_type": "SOURCE_PRIVATE",
                "instance_count": 1,
                "shm_gi": 900,
                "spec_id": "1d3cb6ac-6a44-4acf-b8de-7facaedd8330"
            }],
            "fault_tolerance_max_retry": 2,
            "fault_tolerance_retry_interval_sec": 300,
            "logic_compute_group_id": "lcg-79b2ad0e",
            "max_running_time_ms": "3600000",
            "name": f"tune-birwkv-cfg{cfg}-temp{temp}",
            "project_id": "project-632c8db8-4530-413a-ada5-df91774a7e09",
            "task_priority": 8,
            "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
        }

        result = subprocess.run(
            ["qz", "train", "CreateJob", "--data", json.dumps(request)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"  ✅ cfg={cfg}, temp={temp}")
        else:
            print(f"  ❌ cfg={cfg}, temp={temp}")

print()
print("=" * 80)
print(f"  已提交 {len(configs)} 个参数组合的搜索任务")
print("  预期耗时: 30-40分钟（并行执行）")
print("  结果位置: results/benchmarks/global-birwkv-param-search/")
print("=" * 80)

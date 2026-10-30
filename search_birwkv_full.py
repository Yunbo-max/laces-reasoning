#!/usr/bin/env python3
"""
Global-BiRWKV 全面参数搜索 - 1个H100，大搜索空间找最优
"""
import subprocess
import json
from itertools import product

print("=" * 100)
print("  Global-BiRWKV 全面参数搜索")
print("=" * 100)
print()

# 大搜索空间
cfg_scales = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

configs = list(product(cfg_scales, temperatures))
print(f"搜索空间: {len(cfg_scales)} × {len(temperatures)} = {len(configs)} 个参数组合")
print(f"测试任务: arc_easy, winogrande, sciq (3个)")
print()

MODEL_PATH = "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-birwkv768x8-b16-50k/step_00050000"

submitted = 0
for cfg, temp in configs:
    task_cmd = (
        f"cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning && "
        f"export HF_HUB_OFFLINE=1 && "
        f"python scripts/eval/eval_prefix_suffix_cfg_lambada.py "
        f"--ckpt_dir {MODEL_PATH} "
        f"--tasks arc_easy winogrande sciq "
        f"--cfg_scale {cfg} "
        f"--temperature {temp} "
        f"--steps 100 "
        f"--output_dir results/benchmarks/global-birwkv-fullsearch/cfg{cfg:.1f}_temp{temp:.1f}"
    )

    request = {
        "auto_fault_tolerance": True,
        "command": f"bash -lc '{task_cmd}'",
        "description": f"Full search: birwkv cfg={cfg:.1f} temp={temp:.1f}",
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
        "name": f"birwkv-fullsearch-cfg{cfg:.1f}-temp{temp:.1f}",
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
        submitted += 1
        print(f"✅ cfg={cfg:.1f}, temp={temp:.1f}")
    else:
        print(f"❌ cfg={cfg:.1f}, temp={temp:.1f}")

print()
print("=" * 100)
print(f"Global-BiRWKV: {submitted}/{len(configs)} 任务提交成功")
print(f"预期耗时: ~5小时（1个H100顺序执行 {len(configs)} 个任务）")
print(f"结果目录: results/benchmarks/global-birwkv-fullsearch/")
print("=" * 100)

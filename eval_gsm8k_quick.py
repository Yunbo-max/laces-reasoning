#!/usr/bin/env python3
"""
用1个H100立刻测GSM8K: raw-rwkv-2.9b 和 champ-dynamic
"""
import subprocess
import json

print("=" * 100)
print("  GSM8K + C-Eval 快速测试 - 2个模型")
print("=" * 100)
print()

models = {
    "raw-rwkv-2.9b": {
        "ckpt": "outputs_relay/singlez-2.9B-s2-finetune/step_00050000",  # raw model path
        "name": "raw-rwkv"
    },
    "champ-dynamic": {
        "ckpt": "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000",
        "name": "champ-dynamic"
    }
}

submitted = 0

for model_name, info in models.items():
    print(f"提交: {model_name}")

    ckpt_path = info["ckpt"]
    output_name = info["name"]

    # GSM8K + C-Eval eval命令
    task_cmd = (
        f"cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning && "
        f"export HF_HUB_OFFLINE=1 && "
        f"python scripts/eval/eval_prefix_suffix_cfg_lambada.py "
        f"--ckpt_dir {ckpt_path} "
        f"--tasks gsm8k c_eval "
        f"--cfg_scale 3.0 "
        f"--temperature 0.0 "
        f"--steps 100 "
        f"--output_dir results/benchmarks/{output_name}-gsm8k-ceval"
    )

    request = {
        "auto_fault_tolerance": True,
        "command": f"bash -lc '{task_cmd}'",
        "description": f"GSM8K: {model_name}",
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
        "name": f"eval-gsm8k-{output_name}",
        "project_id": "project-632c8db8-4530-413a-ada5-df91774a7e09",
        "task_priority": 9,
        "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
    }

    result = subprocess.run(
        ["qz", "train", "CreateJob", "--data", json.dumps(request)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"  ✅ 提交成功")
        submitted += 1
    else:
        print(f"  ❌ 提交失败")

print()
print("=" * 100)
print(f"✅ {submitted}/2 评测任务已提交")
print("预期耗时: 2-3小时 (1个H100，GSM8K+C-Eval)")
print("结果目录: results/benchmarks/{raw-rwkv,champ-dynamic}-gsm8k-ceval/")
print("=" * 100)

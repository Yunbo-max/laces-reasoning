#!/usr/bin/env python3
"""
提交 DiT + fixed 50k 训练任务到QZ
"""
import subprocess
import json

print("=" * 90)
print("  提交 DiT + fixed 50k 训练")
print("=" * 90)
print()

output_dir = "outputs_fixed_basis/laces-2.9B-fixed-r32-global-dit-b16-50k"

train_cmd = (
    "cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning && "
    "export HF_HUB_OFFLINE=1 && "
    "python scripts/train/train_relay_lm.py "
    "--backbone rwkv7-goose-2.9b "
    "--denoiser dit "
    "--writer fixed "
    "--trajectory_mode global "
    "--data_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/fineweb_4096 "
    f"--output_dir {output_dir} "
    "--steps 50000 "
    "--batch_size 16 "
    "--learning_rate 2e-4 "
    "--log_interval 100"
)

request = {
    "auto_fault_tolerance": True,
    "command": f"bash -lc '{train_cmd}'",
    "description": "Train DiT + fixed 50k for 2.9B model matrix",
    "framework": "pytorch",
    "framework_config": [{
        "image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
        "image_type": "SOURCE_PRIVATE",
        "instance_count": 1,
        "shm_gi": 900,
        "spec_id": "1d3cb6ac-6a44-4acf-b8de-7facaedd8330"
    }],
    "fault_tolerance_max_retry": 3,
    "fault_tolerance_retry_interval_sec": 300,
    "logic_compute_group_id": "lcg-79b2ad0e",
    "max_running_time_ms": "604800000",
    "name": "train-2.9B-dit-fixed-50k",
    "project_id": "project-632c8db8-4530-413a-ada5-df91774a7e09",
    "task_priority": 6,
    "tb_summary_path": "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/tb_logs",
    "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
}

result = subprocess.run(
    ["qz", "train", "CreateJob", "--data", json.dumps(request)],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("✅ DiT + fixed 50k 训练任务提交成功")
    print()
    print(f"输出目录: {output_dir}")
    print("预期耗时: ~9小时 (1个H100)")
    print("预计完成: 今晚/明天早上")
else:
    print("❌ 提交失败")
    print(result.stderr)

print()
print("=" * 90)

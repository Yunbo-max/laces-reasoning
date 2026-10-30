#!/usr/bin/env python3
"""
发布4个2.9B Chunked模型训练到QZ - 4个GPU并行执行
"""
import subprocess
import json
import sys

WORKSPACE_ID = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT_ID = "project-632c8db8-4530-413a-ada5-df91774a7e09"
LOGIC_COMPUTE_GROUP = "lcg-79b2ad0e"
DOCKER_IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
SPEC_ID = "1d3cb6ac-6a44-4acf-b8de-7facaedd8330"

configs = [
    ("chunked-dit-dynamic", "dit", "dynamic"),
    ("chunked-dit-fixed", "dit", "fixed"),
    ("chunked-birwkv-dynamic", "birwkv", "dynamic"),
    ("chunked-birwkv-fixed", "birwkv", "fixed"),
]

print("=" * 60)
print("  发布4个2.9B Chunked模型训练任务")
print("  配置: 4个GPU并行执行")
print("  预期耗时: ~9小时 (单块H100)")
print("=" * 60)
print()

submitted_count = 0
failed_count = 0

for model_name, denoiser, writer_type in configs:
    print(f"提交: {model_name} ({denoiser} + {writer_type})")

    output_dir = f"outputs_{writer_type}_basis/laces-2.9B-{writer_type}-r32-chunked-{denoiser}-50k"

    # 训练命令
    cmd = (
        "cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning && "
        "export HF_HUB_OFFLINE=1 && "
        f"python scripts/train/train_relay_lm.py "
        f"--backbone rwkv7-goose-2.9b "
        f"--denoiser {denoiser} "
        f"--writer {writer_type} "
        f"--trajectory_mode chunked "
        f"--trajectory_geometry 64x64 "
        f"--data_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/fineweb_4096 "
        f"--output_dir {output_dir} "
        f"--steps 50000 "
        f"--batch_size 16 "
        f"--learning_rate 2e-4 "
        f"--log_interval 100"
    )

    # 构建请求JSON
    request = {
        "auto_fault_tolerance": True,
        "command": f"bash -lc '{cmd}'",
        "description": f"Train 2.9B chunked {model_name} to 50k steps (GPU parallel)",
        "framework": "pytorch",
        "framework_config": [
            {
                "image": DOCKER_IMAGE,
                "image_type": "SOURCE_PRIVATE",
                "instance_count": 1,
                "shm_gi": 900,
                "spec_id": SPEC_ID
            }
        ],
        "fault_tolerance_max_retry": 3,
        "fault_tolerance_retry_interval_sec": 300,
        "logic_compute_group_id": LOGIC_COMPUTE_GROUP,
        "max_running_time_ms": "604800000",
        "name": f"train-2.9B-chunked-{model_name}-50k",
        "project_id": PROJECT_ID,
        "task_priority": 6,
        "tb_summary_path": "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/tb_logs",
        "workspace_id": WORKSPACE_ID
    }

    # 提交到QZ
    result = subprocess.run(
        ["qz", "train", "CreateJob", "--data", json.dumps(request)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"  ✅ 提交成功")
        submitted_count += 1
        # 尝试解析输出获取job ID
        if result.stdout:
            try:
                out_json = json.loads(result.stdout)
                if "id" in out_json:
                    print(f"     Job ID: {out_json['id']}")
            except:
                pass
    else:
        print(f"  ❌ 提交失败")
        if result.stderr:
            print(f"     错误: {result.stderr[:200]}")
        failed_count += 1

    print()

print("=" * 60)
print(f"  结果: {submitted_count}/4 任务提交成功")
if failed_count > 0:
    print(f"  失败: {failed_count} 个任务")
print()
print("  预期耗时: ~9小时 (4个GPU并行)")
print("  预计完成: 在9月20日晚间左右")
print()
print(f"  监控: qz train ListJobs --workspace-id {WORKSPACE_ID}")
print("=" * 60)

sys.exit(0 if failed_count == 0 else 1)

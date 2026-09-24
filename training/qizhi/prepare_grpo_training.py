#!/usr/bin/env python3
"""
Prepares and optionally submits a Qizhi training job for LACES GRPO post-training.

This script generates the JSON request body for a `qz train CreateJob` command
to run GRPO on a pre-trained LACES model using a specific reasoning dataset.

Example (Dry Run):
  python training/qizhi/prepare_grpo_training.py --dataset deepmath --dry-run
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Qizhi Job Configuration ---
QZ_WORKSPACE_ID = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
QZ_PROJECT_ID = "project-632c8db8-4530-413a-ada5-df91774a7e09"
QZ_LOGIC_COMPUTE_GROUP_ID = "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7"
QZ_IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
QZ_H100_8_CARD_SPEC_ID = "8a53ac21-299a-4dee-85e9-9c04a544cf8d" # 8-card H200 spec

# --- Paths ---
# This checkpoint is the starting point for GRPO fine-tuning.
BASE_CHECKPOINT_DIR = REPO_ROOT / "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000"

# Datasets for GRPO, assuming they are prepared by `prepare_reasoning_datasets.sh`
GRPO_DATASETS = {
    "deepmath": REPO_ROOT / "data/deepmath_block_grpo",
    "gsm8k": REPO_ROOT / "data/gsm8k_block_grpo",
    "math": REPO_ROOT / "data/math_block_grpo",
}

def get_training_command(dataset: str, args: argparse.Namespace) -> str:
    """Constructs the `torchrun` command for GRPO training."""
    if dataset not in GRPO_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    data_dir = GRPO_DATASETS[dataset]
    output_dir = REPO_ROOT / f"outputs_grpo/laces-2.9B-{dataset}-from-30k"
    nproc_per_node = 8

    # Arguments for laces_posttrain.run_math_block_grpo
    grpo_args = [
        f"--data {data_dir}",
        f"--ckpt-dir {BASE_CHECKPOINT_DIR}",
        f"--output {output_dir}",
        f"--mode train",
        f"--steps {args.steps}",
        f"--lr {args.lr}",
        f"--group-size {args.group_size}",
        f"--kl-coef {args.kl_coef}",
        f"--save-every {args.save_every}",
        f"--eval-every {args.eval_every}",
        # Add other GRPO hyperparameters as needed
    ]

    # Using DDP via torchrun
    torchrun_cmd = f"""\
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nnodes=1 --nproc_per_node={nproc_per_node} --node_rank=0 --master_addr=127.0.0.1 --master_port=29502 \
    -m laces_posttrain.run_math_block_grpo \
    {' '.join(grpo_args)}
    """

    full_command = f"cd {str(REPO_ROOT)} && {torchrun_cmd}"
    return ' '.join(full_command.split())

def main():
    p = argparse.ArgumentParser(description="Prepare Qizhi job for LACES GRPO training.")
    p.add_argument("--dataset", required=True, choices=GRPO_DATASETS.keys(), help="Dataset for GRPO training.")
    p.add_argument("--dry-run", action="store_true", help="Print the qz command instead of the JSON body.")
    p.add_argument("--steps", type=int, default=10000, help="Number of training steps.")
    p.add_argument("--lr", type=float, default=1e-7, help="Learning rate.")
    p.add_argument("--group-size", type=int, default=8, help="GRPO group size.")
    p.add_argument("--kl-coef", type=float, default=0.02, help="KL coefficient.")
    p.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps.")
    p.add_argument("--eval-every", type=int, default=1000, help="Evaluate every N steps.")
    args = p.parse_args()

    username = os.getenv('USER', 'opencode-user')
    job_name = f"laces-grpo-{args.dataset}-{username}"
    description = f"LACES GRPO post-training on '{args.dataset}' from dynamic-basis 30k checkpoint."
    command = get_training_command(args.dataset, args)

    job_data = {
        "workspace_id": QZ_WORKSPACE_ID,
        "project_id": QZ_PROJECT_ID,
        "logic_compute_group_id": QZ_LOGIC_COMPUTE_GROUP_ID,
        "name": job_name,
        "description": description,
        "framework": "pytorch",
        "command": command,
        "max_running_time_ms": str(4 * 24 * 60 * 60 * 1000),  # 4 days
        "task_priority": 5,
        "framework_config": [
            {
                "image": QZ_IMAGE,
                "image_type": "SOURCE_PRIVATE",
                "instance_count": 1,
                "spec_id": QZ_H100_8_CARD_SPEC_ID,
                "shm_gi": 1800,
            }
        ],
    }

    if args.dry_run:
        json_str_escaped = json.dumps(json.dumps(job_data))
        print(f"qz train CreateJob --dry-run --data {json_str_escaped}")
    else:
        print(json.dumps(job_data, indent=2))

if __name__ == "__main__":
    main()

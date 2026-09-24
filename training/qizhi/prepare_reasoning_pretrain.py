#!/usr/bin/env python3
"""
Prepares and optionally submits a Qizhi training job for the LACES reasoning-aware pretraining.

This script generates the JSON request body for a `qz train CreateJob` command.
It encapsulates the logic for dataset paths, training stages (P0, P1, P2), and resource configuration.

Example (Dry Run):
  python training/qizhi/prepare_reasoning_pretrain.py --stage p0 --dry-run

Example (Live Submission):
  python training/qizhi/prepare_reasoning_pretrain.py --stage p1 | xargs -0 qz train CreateJob --data
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

def get_abs_path(p):
    return str(Path(p).resolve())

# --- Qizhi Job Configuration ---
# These are based on previous successful job submissions in the project.
# TODO: Potentially move these to a shared YAML config.
QZ_WORKSPACE_ID = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
QZ_PROJECT_ID = "project-632c8db8-4530-413a-ada5-df91774a7e09"
QZ_LOGIC_COMPUTE_GROUP_ID = "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7"
QZ_IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"

# H100 Specs
QZ_H100_8_CARD_SPEC_ID = "8a53ac21-299a-4dee-85e9-9c04a544cf8d" # 8-card H200 spec # 8-card 80G H100
QZ_H100_1_CARD_SPEC_ID = "..." # TODO: Find a 1-card H100 spec ID if needed

# --- Dataset Configuration ---
# Placeholder paths. These should point to the actual preprocessed data on the cluster.
DATASET_PATHS = {
    "fineweb_edu": get_abs_path("preprocessed_data/fineweb_edu_packed_full"),
    "openwebmath": get_abs_path("preprocessed_data/openwebmath_packed_full"),
    "proofpile2": get_abs_path("preprocessed_data/proofpile2_packed_full"),
    "code": get_abs_path("preprocessed_data/starcoder_packed_full"),
    "openthoughts": get_abs_path("preprocessed_data/openthoughts_packed_full"),
    "science": get_abs_path("preprocessed_data/science_texts_packed_full"),
}

# --- Model & Checkpoint Configuration ---
# Path to the base RWKV model used for the frozen backbone.
BASE_RWKV_PATH = get_abs_path("models/RWKV7-Goose-World3-2.9B-HF-fla-v2")

def get_training_command(stage: str, args: argparse.Namespace) -> str:
    """Constructs the `torchrun` command for a given training stage."""

    # Shared settings
    run_name_prefix = f"laces-2.9B-reasoning-pretrain-{stage}"
    config_name = "rwkv_relay_2.9B_dynamic_basis_dynlowrank_r32"
    nproc_per_node = 8
    batch_size = 4 # Per-GPU batch size

    # Stage-specific logic
    if stage == "p0": # S0 behavioral VAE
        # TODO: This requires a new training mode in `train_state_hijacking_dit.py`
        # to implement the `L_behavior` loss. For now, we stub it out.
        print("WARNING: Stage p0 (behavioral VAE) is not fully implemented in the trainer.", file=sys.stderr)
        print("WARNING: Generating a placeholder command.", file=sys.stderr)
        stage_specific_args = [
            "training.stage=0",
            "+training.behavioral_vae_loss=true", # Hypothetical flag
            f"loss.text_loss_weight={args.lambda_beh}",
            f"logging.run_name={run_name_prefix}-beh{args.lambda_beh}",
        ]
        # Dataset mixture for P0
        # For simplicity, we just point to a single dir. A real implementation
        # would use a dataloader that mixes these sources according to ratios.
        token_dir = DATASET_PATHS["fineweb_edu"]

    elif stage == "p1": # S1 dynamic memory writer
        stage_specific_args = ["training.stage=1"]
        token_dir = DATASET_PATHS["fineweb_edu"]

    elif stage == "p2": # S2 trajectory prior
        stage_specific_args = ["training.stage=2"]
        token_dir = DATASET_PATHS["openwebmath"] # More reasoning-focused

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Assemble the full torchrun command
    torchrun_cmd = f"""\
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nnodes=1 --nproc_per_node={nproc_per_node} --node_rank=0 --master_addr=127.0.0.1 --master_port=29501 \
    train_state_hijacking_dit.py \
    --config-name {config_name} \
    model.rwkv_local_path={BASE_RWKV_PATH} \
    data.token_dir={token_dir} \
    training.train_batch_size={batch_size} \
    {' '.join(stage_specific_args)}
    """

    # Command to be executed on the Qizhi node
    full_command = f"cd {get_abs_path('.')} && {torchrun_cmd}"
    return ' '.join(full_command.split())

def main():
    p = argparse.ArgumentParser(description="Prepare Qizhi job for LACES reasoning pretraining.")
    p.add_argument("--stage", required=True, choices=["p0", "p1", "p2"], help="Training stage to prepare.")
    p.add_argument("--dry-run", action="store_true", help="Print the qz command instead of the JSON body.")
    p.add_argument("--lambda_beh", type=float, default=0.5, help="Weight for the behavioral loss in Stage P0.")
    args = p.parse_args()

    username = os.getenv('USER', 'opencode-user')
    job_name = f"laces-reasoning-pretrain-{args.stage}-{username}"
    description = f"Reasoning-aware LACES pretraining, Stage {args.stage.upper()}. Datasets from arXiv:2609.13356v1 proposal."
    command = get_training_command(args.stage, args)

    job_data = {
        "workspace_id": QZ_WORKSPACE_ID,
        "project_id": QZ_PROJECT_ID,
        "logic_compute_group_id": QZ_LOGIC_COMPUTE_GROUP_ID,
        "name": job_name,
        "description": description,
        "framework": "pytorch",
        "command": command,
        "max_running_time_ms": str(2 * 24 * 60 * 60 * 1000),  # 2 days
        "task_priority": 6,
        "framework_config": [
            {
                "image": QZ_IMAGE,
                "image_type": "SOURCE_PRIVATE",
                "instance_count": 1,
                "spec_id": QZ_H200_1_CARD_SPEC_ID,
                "shm_gi": 900,
            }
        ],
    }

    if args.dry_run:
        # In dry-run, escape the JSON for shell command
        json_str_escaped = json.dumps(json.dumps(job_data))
        print(f"qz train CreateJob --dry-run --data {json_str_escaped}")
    else:
        # In live mode, print the raw JSON for piping to `qz`
        print(json.dumps(job_data, indent=2))

if __name__ == "__main__":
    main()

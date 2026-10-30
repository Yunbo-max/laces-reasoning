#!/usr/bin/env python3
"""Prepare and (optionally) submit the GLOBAL dynamic-basis training job on Qizhi.

This is the single-z counterpart of the dynamic-basis chunked recipe: same
dynlowrank S1 writer (rank 32), same joint S1+S2 stage-2 co-adaptation, but
trajectory_mode is off so one global latent z drives the writer.

  config : configs/rwkv_relay_2.9B_dynamic_basis_dynlowrank_r32_global.yaml
  data   : DiffRwkv/preprocessed_data/owt_rwkv_tokens/train  (512-token OWT shards)
  model  : /inspire/.../models/RWKV7-Goose-World3-2.9B-HF    (frozen backbone)

Resource: ONE H100 (80GB), 200GiB host memory, 20 CPU.
Set --nproc 8 to use the 8-card H100 node instead.

Examples
--------
  python training/qizhi/prepare_global_basis_training.py --dry-run
  python training/qizhi/prepare_global_basis_training.py --batch 24 > /tmp/job.json
  qz train CreateJob --data @/tmp/job.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Qizhi configuration -----------------------------------------------------
QZ_WORKSPACE_ID = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
QZ_PROJECT_ID = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"  # 面向新一代编程智能体的AI系统和Infra研究
QZ_LOGIC_COMPUTE_GROUP_ID = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
QZ_IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"

# 1-card H100 (80GB) spec, resolved for the LCG above via
#   qz resource-price GetLogicComputeGroupResourceSpecPrices
# (gpu_count=1, memory_size_gib=200, cpu_count=20).
# NOTE: spec_id overrides gpu_count -- do not also pass a gpu_count field.
QZ_H100_1_CARD_SPEC_ID = "79fe954a-be92-4772-ac0b-94ad8a79b7bb"
# 8-card H100 spec, kept for reference:
QZ_H100_8_CARD_SPEC_ID = "7166bd2e-6cbe-4bd9-be38-762d11003e7f"

CONFIG_NAME = "rwkv_relay_2.9B_dynamic_basis_dynlowrank_r32_global"
RUN_NAME = "laces-2.9B-dynlowrank-r32-global-joint-b32-50k"

# Defaults -- override from the CLI.
DEFAULT_STEPS = 50000
DEFAULT_BATCH = 16          # single-card H100 80GB. Reference per-GPU batch was 4
                            # (x8 cards = global 32); 16 = 4x that. Raise to 32 if the
                            # log shows headroom, drop to 8 if it OOMs.
DEFAULT_RESUME = (
    "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000"
)
DEFAULT_MAX_HOURS = 72


def run_name_for(args: argparse.Namespace) -> str:
    return (
        f"laces-2.9B-{args.s1_writer}-r32-global-{args.global_denoiser}"
        f"{args.dit_hidden}x{args.dit_depth}-b{args.batch}-{args.steps // 1000}k"
    )


def build_command(args: argparse.Namespace) -> str:
    overrides = [
        f"model.rwkv_local_path={REPO_ROOT / 'models/RWKV7-Goose-World3-2.9B-HF-fla-v2'}",
        f"model.dit_hidden={args.dit_hidden}",
        f"model.dit_depth={args.dit_depth}",
        f"model.dit_num_heads={args.dit_heads}",
        f"model.global_denoiser_type={args.global_denoiser}",
        f"model.s1_writer_type={args.s1_writer}",
        f"training.require_s1_writer_grad={str(args.require_s1_writer_grad).lower()}",
        f"training.num_train_steps={args.steps}",
        f"training.train_batch_size={args.batch}",
        f"training.save_every_n_steps={args.save_every}",
        f"logging.run_name={run_name_for(args)}",
    ]
    if args.resume:
        overrides.append(f"training.resume={args.resume}")

    if args.nproc <= 1:
        launcher = f"CUDA_VISIBLE_DEVICES={args.gpu} python -u "
    else:
        launcher = (
            "CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(args.nproc)) + " "
            f"torchrun --nnodes=1 --nproc_per_node={args.nproc} --node_rank=0 "
            f"--master_addr={args.master_addr} --master_port={args.master_port} "
        )
    inner = (
        f"mkdir -p {REPO_ROOT}/validation_logs_qz && "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        + launcher
        + "train_state_hijacking_dit.py "
        f"--config-name {CONFIG_NAME} "
        + " ".join(overrides)
    )
    return f"cd {REPO_ROOT} && {inner} > {REPO_ROOT}/validation_logs_qz/{run_name_for(args)}.log 2>&1"


def build_job(args: argparse.Namespace) -> dict:
    return {
        "workspace_id": QZ_WORKSPACE_ID,
        "project_id": QZ_PROJECT_ID,
        "logic_compute_group_id": QZ_LOGIC_COMPUTE_GROUP_ID,
        "name": f"global-{args.s1_writer}-{args.global_denoiser}{args.dit_hidden}x{args.dit_depth}-b{args.batch}-{args.steps // 1000}k",
        "description": (
            "GLOBAL single-z training with the dynamic-basis (dynlowrank rank-32) S1 writer; "
            "joint S1+S2 stage-2 co-adaptation on OWT 512-token windows, frozen 2.9B RWKV-7 renderer."
        ),
        "framework": "pytorch",
        "command": build_command(args),
        "max_running_time_ms": str(int(args.max_hours * 3600 * 1000)),
        "task_priority": 6,
        "framework_config": [
            {
                "image": QZ_IMAGE,
                "image_type": "SOURCE_PRIVATE",
                "instance_count": 1,
                "spec_id": QZ_H100_1_CARD_SPEC_ID if args.nproc <= 1 else QZ_H100_8_CARD_SPEC_ID,
                "shm_gi": 900,
            }
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                   help="Per-step batch. Raise until the H100 is full.")
    p.add_argument("--resume", default=DEFAULT_RESUME,
                   help="S0+S1 checkpoint dir. Empty trains from scratch.")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--dit-hidden", type=int, default=768, help="Global S2 DiT width.")
    p.add_argument("--dit-depth", type=int, default=8, help="Global S2 DiT depth.")
    p.add_argument("--dit-heads", type=int, default=4, help="Global S2 DiT heads.")
    p.add_argument("--global-denoiser", default="dit", choices=("dit", "rwkv", "birwkv"),
                   help="Denoiser driving the single global latent.")
    p.add_argument("--s1-writer", default="dynlowrank", choices=("dynlowrank", "fixed", "mixture"),
                   help="S1 state writer: dynlowrank (dynamic basis) or fixed (static basis).")
    p.add_argument("--require-s1-writer-grad", default=True,
                   type=lambda v: str(v).lower() not in ("0", "false", "no"),
                   help="Guard that the dynlowrank writer receives nonzero gradient. "
                        "Must be false for the fixed writer, which the trainer forbids.")
    p.add_argument("--gpu", default="0")
    p.add_argument("--nproc", type=int, default=1,
                   help="GPUs to use: 1 = single card (plain python); >1 = torchrun DDP.")
    p.add_argument("--master-addr", default="127.0.0.1")
    p.add_argument("--master-port", default="29501")
    p.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the qz command instead of the JSON body.")
    args = p.parse_args()

    job = build_job(args)

    if args.dry_run:
        print(f"qz train CreateJob --dry-run --data {json.dumps(json.dumps(job))}")
    else:
        print(json.dumps(job, indent=2))


if __name__ == "__main__":
    main()

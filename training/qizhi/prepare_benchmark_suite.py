#!/usr/bin/env python3
"""Prepare a Qizhi evaluation job for the locally runnable LACES benchmark subset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex

REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFRWKV_ROOT = REPO_ROOT.parent / "DiffRwkv"
PROJECT_WORKDIR = Path("/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222")

QZ_WORKSPACE_ID = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
QZ_PROJECT_ID = "project-632c8db8-4530-413a-ada5-df91774a7e09"
QZ_LOGIC_COMPUTE_GROUP_ID = "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7"
QZ_IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
QZ_SPEC_ID = "7166bd2e-6cbe-4bd9-be38-762d11003e7f"

CHECKPOINT_PATHS = {
    "dynamic-basis": REPO_ROOT / "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000",
    "fixed-basis": DIFFRWKV_ROOT / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000",
    "raw-rwkv": DIFFRWKV_ROOT / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000",
}

# Only keep benchmark groups whose entrypoints and known data roots are present.
BENCHMARK_CONFIGS = {
    "mmlu_pro": {
        "script": "python -m laces_posttrain.run",
        "output_subdir": "mmlu_pro",
        "args": [
            "--mode eval",
            "--data data/mmlu_pro_eval",
            "--split dev",
            "--s2-checkpoint {ckpt}",
        ],
    },
    "gsm8k": {
        "script": "python scripts/eval/eval_chain_reasoning_numeric.py",
        "output_subdir": "gsm8k",
        "args": [
            "--ckpt_dir {ckpt}",
            "--task gsm8k",
            "--n 100",
            "--output {output}",
        ],
    },
    "cola8": {
        "script": "python scripts/eval/run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py",
        "output_subdir": "cola8",
        "args": [
            "--ckpt_dir {ckpt}",
            "--task_data_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/baseline/Cola-DLM/generate_task_data",
            "--output_dir {output}",
            "--tasks mmlu,obqa,race,squad,story_cloze",
            "--method traj_cfg",
            "--max_samples 100",
            "--steps 100",
            "--cfg_scale 3.0",
            "--max_new_tokens 32",
            "--temperature 0.0",
        ],
    },
    "mcpack": {
        "output_subdir": "mcpack",
        "commands": [
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_multichoice.py --ckpt_dir {ckpt} --task_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/mmlu/validation --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/mmlu.json",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_multichoice.py --ckpt_dir {ckpt} --task_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/openbookqa/validation --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/openbookqa.json",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_multichoice.py --ckpt_dir {ckpt} --task_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/race/validation --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/race.json",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_multichoice.py --ckpt_dir {ckpt} --task_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/siqa/validation --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/siqa.json",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_multichoice.py --ckpt_dir {ckpt} --task_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/story_cloze/validation --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/story_cloze.json",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_multichoice.py --ckpt_dir {ckpt} --task_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/piqa/validation --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/piqa.json",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_multichoice.py --ckpt_dir {ckpt} --task_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/hellaswag/validation --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/hellaswag.json",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_multichoice.py --ckpt_dir {ckpt} --task_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/arc_easy/validation --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/arc_easy.json",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_multichoice.py --ckpt_dir {ckpt} --task_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/arc_challenge/validation --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/arc_challenge.json",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_multichoice.py --ckpt_dir {ckpt} --task_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/winogrande/validation --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/winogrande.json",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_multichoice.py --ckpt_dir {ckpt} --task_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/sciq/validation --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/sciq.json",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_lambada.py --ckpt_dir {ckpt} --data_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/lambada/test --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/lambada.json"
        ]
    },
    "paperpack": {
        "output_subdir": "paperpack",
        "commands": [
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_prefix_suffix_cfg_multichoice.py --ckpt_dir {ckpt} --task_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/mmlu_pro/test --max_samples 200 --steps 100 --scales 1 --device cuda --output {output}/mmlu_pro.json",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_state_hijack_target_ppl_matrix.py --ckpt_dir {ckpt} --data_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/xsum/validation --output {output}/xsum_target_ppl.json --max_samples 200 --device cuda --dtype bf16 --conditions raw,ddpm100",
            "python /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/scripts/eval/eval_state_hijack_target_ppl_matrix.py --ckpt_dir {ckpt} --data_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/wmt14_de_en/validation --output {output}/wmt14_de_en_target_ppl.json --max_samples 200 --device cuda --dtype bf16 --conditions raw,ddpm100"
        ]
    },
    "babilong": {
        "script": "python scripts/eval/run_open_context_bench.py",
        "output_subdir": "babilong",
        "args": [
            "--suite babilong",
            "--datasets qa1,qa2",
            "--data_dir /tmp/babilong_eval",
            "--ckpt_dir {ckpt}",
            "--arm {arm}",
            "--max_samples 20",
            "--output {output}",
        ],
    },
}


def arm_for_model(model_type: str) -> str:
    return "raw" if model_type == "raw-rwkv" else "dynamic"


def build_command(model_type: str, benchmarks: list[str]) -> str:
    ckpt = CHECKPOINT_PATHS[model_type].resolve()
    commands: list[str] = []
    arm = arm_for_model(model_type)
    for benchmark in benchmarks:
        cfg = BENCHMARK_CONFIGS[benchmark]
        output = f"results/benchmarks/{model_type}/{cfg['output_subdir']}"
        if "commands" in cfg:
            commands.extend(cmd.format(ckpt=ckpt, output=output, arm=arm) for cmd in cfg["commands"])
            continue
        formatted_args = [arg.format(ckpt=ckpt, output=output, arm=arm) for arg in cfg["args"]]
        commands.append(f"{cfg['script']} {' '.join(formatted_args)}")
    log_dir = PROJECT_WORKDIR / "qz_eval_logs"
    log_path = log_dir / f"{model_type}_{'_'.join(benchmarks)}.log"
    inner = (
        f"set -euxo pipefail; "
        f"mkdir -p {shlex.quote(str(log_dir))}; "
        f"exec > {shlex.quote(str(log_path))} 2>&1; "
        f"cd {shlex.quote(str(REPO_ROOT))}; "
        f"export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; "
        f"{' && '.join(commands)}"
    )
    return f"bash -lc {shlex.quote(inner)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", required=True, choices=CHECKPOINT_PATHS.keys())
    parser.add_argument("--benchmark", required=True, nargs="+", choices=BENCHMARK_CONFIGS.keys())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_name = "_".join(args.benchmark)
    job_name = f"laces-benchmark-{args.model_type}-{benchmark_name}-{os.getenv('USER', 'opencode-user')}"
    job = {
        "workspace_id": QZ_WORKSPACE_ID,
        "project_id": QZ_PROJECT_ID,
        "logic_compute_group_id": QZ_LOGIC_COMPUTE_GROUP_ID,
        "name": job_name,
        "description": f"Runnable benchmark subset {', '.join(args.benchmark)} on {args.model_type}",
        "framework": "pytorch",
        "command": build_command(args.model_type, args.benchmark),
        "max_running_time_ms": str(12 * 60 * 60 * 1000),
        "task_priority": 5,
        "framework_config": [
            {
                "image": QZ_IMAGE,
                "image_type": "SOURCE_PRIVATE",
                "instance_count": 1,
                "spec_id": QZ_SPEC_ID,
                "shm_gi": 900,
            }
        ],
    }
    if args.dry_run:
        print(f"qz train CreateJob --dry-run --data {json.dumps(json.dumps(job))}")
    else:
        print(json.dumps(job, indent=2))


if __name__ == "__main__":
    main()

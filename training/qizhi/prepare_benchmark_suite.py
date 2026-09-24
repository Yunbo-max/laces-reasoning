#!/usr/bin/env python3
"""
Prepares and optionally submits a Qizhi evaluation job for comparing LACES models.

This script generates the JSON request body for a `qz train CreateJob` command
to run a suite of benchmarks (MMLU-Pro, GPQA, etc.) on an H100 node.

It supports comparing:
1. Fixed-basis LACES (the previous champion)
2. Dynamic-basis LACES (the new model)
3. Raw RWKV (the frozen backbone)

Example (Dry Run):
  python training/qizhi/prepare_benchmark_suite.py --model-type dynamic-basis --benchmark mmlu_pro --dry-run

Example (Live Submission):
  python training/qizhi/prepare_benchmark_suite.py --model-type fixed-basis --benchmark gsm8k | xargs -0 qz train CreateJob --data
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
QZ_H100_1_CARD_SPEC_ID = "7166bd2e-6cbe-4bd9-be38-762d11003e7f" # 1-card H200 spec

# --- Checkpoint Paths ---
# These paths must be accessible from the Qizhi execution environment.
CHECKPOINT_PATHS = {
    "dynamic-basis": REPO_ROOT / "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000",
    # This is the champion from the original DiffRwkv repo. We assume it's available.
    "fixed-basis": REPO_ROOT.parent / "DiffRwkv/outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000",
    "raw-rwkv": REPO_ROOT / "models/RWKV7-Goose-World3-2.9B-HF-fla-v2",
}

# --- Benchmark Configuration ---
# Maps benchmark names to the scripts and arguments needed to run them.
BENCHMARK_CONFIGS = {
    "mmlu_pro": {
        "script": "laces_posttrain.run",
        "args": {
            "--mode": "eval",
            "--data-dir": "data/mmlu_pro_eval", # Assumes prepared data
        },
    },
    },
    "mmlu_redux": {
        "script": "scripts/eval/eval_mmlu_redux.py",
        "args": {
            "--split": "test",
        },
    },
    "gpqa_diamond": {
        "script": "scripts/eval/eval_gpqa.py",
        "args": {
            "--dataset": "gpqa_diamond",
        },
    },
    "c_eval": {
        "script": "scripts/eval/eval_ceval.py",
        "args": {
            "--split": "val",
        },
    },
    "gsm8k": {
        "script": "scripts/eval/eval_chain_reasoning_numeric.py",
        "args": {
            "--dataset": "gsm8k",
        },
    },
    "lambada": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "lambada_openai",
        },
    },
    "piqa": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "piqa",
        },
    },
    "hellaswag": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "hellaswag",
        },
    },
    "winogrande": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "winogrande",
        },
    },
    "arc_easy": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "arc_easy",
        },
    },
    "arc_challenge": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "arc_challenge",
        },
    },
    "glue": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "glue",
        },
    },
    "sciq": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "sciq",
        },
    },
    "squad": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "squad2",
        },
    },
    "storycloze": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "storycloze_2016",
        },
    },
    "ifeval": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "ifeval",
        },
    },
    "bfcl": {
        "script": "scripts/eval/eval_bfcl.py",
        "args": {},
    },
    "truthfulqa": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "truthfulqa_mc",
        },
    },
    "known_unknowns": {
        "script": "scripts/eval/eval_lm_harness.py",
        "args": {
            "--tasks": "bigbench_known_unknowns",
        },
    },
    "babilong": {
        "script": "scripts/eval/eval_babilong.py",
        "args": {},
    },
}

def get_eval_command(model_type: str, benchmarks: list[str], args: argparse.Namespace) -> str:
    """Constructs a single command to run a sequence of benchmarks."""
    if model_type not in CHECKPOINT_PATHS:
        raise ValueError(f"Unknown model type: {model_type}")

    all_eval_cmds = []
    for benchmark in benchmarks:
        if benchmark not in BENCHMARK_CONFIGS:
            print(f"Warning: Unknown benchmark '{benchmark}', skipping.", file=sys.stderr)
            continue

        ckpt_path = CHECKPOINT_PATHS[model_type]
        bench_cfg = BENCHMARK_CONFIGS[benchmark]

        if bench_cfg["script"].startswith("laces_posttrain"):
            entry_cmd = f"python -m {bench_cfg['script']}"
        else:
            entry_cmd = f"python {bench_cfg['script']}"

        output_dir = f"results/benchmarks/{model_type}/{benchmark}"

        eval_args = [
            f"--ckpt-dir {ckpt_path}",
            f"--output-dir {output_dir}",
            f"--device cuda",
        ]
        for arg, val in bench_cfg["args"].items():
            eval_args.append(f"{arg} {val}")
        
        all_eval_cmds.append(f"{entry_cmd} {' '.join(eval_args)}")

    # Chain all benchmark commands with &&
    chained_cmd = " && ".join(all_eval_cmds)
    full_command = f"cd {str(REPO_ROOT)} && {chained_cmd}"
    return ' '.join(full_command.split())

def main():
    p = argparse.ArgumentParser(description="Prepare Qizhi benchmark evaluation job.")
    p.add_argument("--model-type", required=True, choices=CHECKPOINT_PATHS.keys(), help="Model to evaluate.")
    p.add_argument("--benchmark", required=True, nargs='+', choices=BENCHMARK_CONFIGS.keys(), help="Benchmark(s) to run.")
    p.add_argument("--dry-run", action="store_true", help="Print the qz command instead of the JSON body.")
    args = p.parse_args()

    # Create a composite name for the benchmarks
    benchmark_name = "_".join(args.benchmark)
    job_name = f"laces-benchmark-{args.model_type}-{benchmark_name}-{os.getenv('USER', 'opencode-user')}"
    description = f"Benchmark(s) '{', '.join(args.benchmark)}' on model '{args.model_type}' on H100."
    command = get_eval_command(args.model_type, args.benchmark, args)

    job_data = {
        "workspace_id": QZ_WORKSPACE_ID,
        "project_id": QZ_PROJECT_ID,
        "logic_compute_group_id": QZ_LOGIC_COMPUTE_GROUP_ID,
        "name": job_name,
        "description": description,
        "framework": "pytorch",
        "command": command,
        "max_running_time_ms": str(12 * 60 * 60 * 1000),  # 12 hours
        "task_priority": 5,
        "framework_config": [
            {
                "image": QZ_IMAGE,
                "image_type": "SOURCE_PRIVATE",
                "instance_count": 1,
                "spec_id": QZ_H100_1_CARD_SPEC_ID,
                "shm_gi": 900, # Half of an 8-card node
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

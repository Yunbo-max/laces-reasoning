#!/usr/bin/env python3
"""Submit the RULER expansion (13 tasks x 3 lengths) as one 8-GPU QZ job.

One job, not 30: the driver `scripts/eval/run_ruler_expand.sh` shards the cells across the
8 GPUs of a single container. Submitting per-cell would pay the model-load cost (a 2.9B
relay checkpoint plus its frozen backbone) 30 times instead of 8.

GetJobLog is broken on this cluster, so the driver redirects everything to logs/qz/ and
this script records the returned job_id in logs/qz/RULER_EXPAND_JOBS.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
EVIDENCE = REPO / ".omo/evidence/qz-training-skill"
LEDGER = REPO / "logs/qz/RULER_EXPAND_JOBS.md"

WS = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
SPEC8, SHM8 = "7166bd2e-6cbe-4bd9-be38-762d11003e7f", 1800

NAME = "ruler-expand-13task"


def body() -> dict:
    # The driver already exports every offline variable itself; the wrapper only has to
    # reach it. `bash` (not `source`) so a driver failure cannot take down the outer shell
    # before the tail of the log is printed.
    cmd = (
        "set -uo pipefail\n"
        f"cd {REPO}\n"
        f"mkdir -p {REPO}/logs/qz {REPO}/results/benchmarks/ruler_expand\n"
        f"bash {REPO}/scripts/eval/run_ruler_expand.sh "
        f"> {REPO}/logs/qz/ruler-expand-driver.log 2>&1 || fail=1\n"
        f"tail -80 {REPO}/logs/qz/ruler-expand-driver.log\n"
        "exit ${fail:-0}\n"
    )
    return {
        "workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
        "name": NAME,
        "description": "RULER 13-task x 3-length expansion (binding-capacity curve), n=100",
        "framework": "pytorch", "command": cmd,
        "max_running_time_ms": "28800000",  # 8 h; longest shard is projected ~2.5 h
        "task_priority": 4,
        "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                              "instance_count": 1, "spec_id": SPEC8, "shm_gi": SHM8}],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    payload = EVIDENCE / "ruler_expand_13task.json"
    payload.write_text(json.dumps(body(), indent=2, ensure_ascii=False) + "\n")
    print(f"[payload] {payload}")

    if args.dry_run:
        r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{payload}", "--dry-run"],
                           capture_output=True, text=True, timeout=120)
        print(r.stdout[-1500:] or r.stderr[-1500:])
        print("DRY-RUN OK" if "DRY RUN" in r.stdout else "DRY-RUN FAILED")
        return
    if not args.submit:
        print("pass --dry-run or --submit")
        return

    r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{payload}"],
                       capture_output=True, text=True, timeout=300)
    try:
        job_id = json.loads(r.stdout)["Result"]["job_id"]
    except Exception:
        print("SUBMIT FAILED\n", r.stdout[:2000], r.stderr[:2000])
        return
    print(f"job_id={job_id}")
    # Record at submit time -- ListJobs paging is broken, so an unrecorded job_id is lost.
    with LEDGER.open("a") as fh:
        fh.write(f"| `{job_id}` | {NAME} | 8 GPU | submitted |\n")


if __name__ == "__main__":
    main()

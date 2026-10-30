#!/usr/bin/env python3
"""Submit the InfiniteBench 7-task expansion as one 8-GPU QZ job.

One job, not eight: the driver `scripts/eval/run_infinitebench_expand.sh` shards the cells
across the 8 GPUs of a single container, so the 2.9B relay checkpoint and its frozen backbone
are loaded once per GPU rather than once per cell.

GetJobLog is broken on this cluster, so the driver redirects everything to logs/qz/ and this
script records the returned job_id in logs/qz/INFBENCH_EXPAND_JOBS.md at submit time.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
EVIDENCE = REPO / ".omo/evidence/qz-training-skill"
LEDGER = REPO / "logs/qz/INFBENCH_EXPAND_JOBS.md"

WS = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
SPEC8, SHM8 = "7166bd2e-6cbe-4bd9-be38-762d11003e7f", 1800

NAME = "ibexp-infinitebench-7task"

# The ladder control reuses this submitter rather than forking it, so the workspace/spec/offline
# wrapper cannot drift between the main run and its own control.
JOBS = {
    "expand": {
        "name": NAME,
        "driver": "scripts/eval/run_infinitebench_expand.sh",
        "log": "logs/qz/ibexp-driver.log",
        "payload": "infbench_expand_7task.json",
        "description": ("InfiniteBench 7 unrun tasks. passkey/number_string at budget 131072 "
                        "with --sample_stride 10 (files are depth-sorted; head-50 would be the "
                        "5 shallowest of 59 depth bins), n=59, both arms + 4 baselines; the "
                        "other 5 tasks n=50 on our two arms."),
        "scope": ("7 tasks; needle pair n=59 stride-10 budget 131072 on 2 arms + rwkv7/gla/"
                  "llama-3.2-3b/qwen2.5-1.5b/qwen2.5-3b; secondary 5 n=50 on 2 arms"),
    },
    "ladder": {
        "name": "ibexp-passkey-length-ladder",
        "driver": "scripts/eval/run_infinitebench_ladder.sh",
        "log": "logs/qz/ibexp-ladder-driver.log",
        "payload": "infbench_ladder_control.json",
        "description": ("Length-ladder CONTROL for the passkey/number_string result: same rows, "
                        "same bare-'Answer:' prompt, same scorer, contexts rebuilt to ~3.9k and "
                        "~15.7k tokens with the needle held at its original depth. Separates "
                        "length from prompt format, which the RULER ladder cannot do because "
                        "RULER ends in a completion cue."),
        "scope": ("passkey+number_string n=59 at 4k and 16k, needle in-input 59/59; 2 arms x 2 "
                  "rungs + rwkv7-world3 x 2 rungs + llama-3.2-3b at 4k"),
    },
}


def body(job: dict) -> dict:
    # `set -uo pipefail` WITHOUT -e: the driver's own failure must not abort this wrapper
    # before the log tail is printed, and GetJobLog cannot be used to recover it.
    # The driver exports every offline variable itself (including the mandatory
    # ${PYTHONPATH:+:$PYTHONPATH} guard -- a bare $PYTHONPATH under set -u kills the job with
    # no log at all), so the wrapper only has to reach it.
    cmd = (
        "set -uo pipefail\n"
        f"cd {REPO}\n"
        f"mkdir -p {REPO}/logs/qz {REPO}/results/benchmarks/infbench_expand\n"
        f"bash {REPO}/{job['driver']} "
        f"> {REPO}/{job['log']} 2>&1 || fail=1\n"
        f"tail -120 {REPO}/{job['log']}\n"
        "exit ${fail:-0}\n"
    )
    return {
        "workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
        "name": job["name"],
        "description": job["description"],
        "framework": "pytorch", "command": cmd,
        # 12 h. Longest shard projected 1.1 h from MEASURED rates at this budget
        # (results/benchmarks/longctx_full: relay 15.7 s/row, rwkv7 10.7, gla 7.0), so this is
        # ~10x headroom. The margin matters: run_open_context_bench writes its result ONCE
        # after the loop, so a shard that overruns the cap yields NOTHING, not a partial file.
        "max_running_time_ms": "43200000",
        "task_priority": 4,
        "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                              "instance_count": 1, "spec_id": SPEC8, "shm_gi": SHM8}],
    }


def record(job_id: str, job: dict) -> None:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text(
            "# InfiniteBench expansion job ledger (the 7 unrun tasks)\n\n"
            "Recorded at submit time because `GetJobLog` is broken on this cluster: without\n"
            "the job_id here a running job cannot be found again.\n\n"
            "| submitted (UTC) | job_id | name | scope |\n|---|---|---|---|\n")
    with LEDGER.open("a") as handle:
        handle.write(f"| {stamp} | `{job_id}` | {job['name']} | {job['scope']} |\n")
    print(f"[ledger] {LEDGER}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--job", choices=sorted(JOBS), default="expand")
    args = ap.parse_args()

    job = JOBS[args.job]
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    payload = EVIDENCE / job["payload"]
    payload.write_text(json.dumps(body(job), indent=2, ensure_ascii=False) + "\n")
    print(f"[payload] {payload}")

    if args.dry_run:
        result = subprocess.run(
            ["qz", "train", "CreateJob", "--data", f"@{payload}", "--dry-run"],
            capture_output=True, text=True, timeout=180)
        print(result.stdout[-2000:] or result.stderr[-2000:])
        print("DRY-RUN OK" if "DRY RUN" in result.stdout else "DRY-RUN FAILED")
        return 0
    if not args.submit:
        print("pass --dry-run or --submit")
        return 1

    result = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{payload}"],
                            capture_output=True, text=True, timeout=300)
    print(result.stdout[-2000:] or result.stderr[-2000:])
    try:
        job_id = json.loads(result.stdout)["Result"]["job_id"]
    except Exception:
        print("could not parse job_id -- RECORD IT BY HAND in the ledger")
        return 1
    print(f"[job] {job_id}")
    record(job_id, job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

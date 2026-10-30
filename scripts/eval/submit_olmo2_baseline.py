#!/usr/bin/env python3
"""Submit the OLMo-2-0425-1B baseline evaluation as one 1-GPU QZ job.

One GPU is enough: the measurable long-context surface for this model is 2 bAbILong cells
(n=100) plus 5 RULER 4096 cells (n=200), all at a <=4k budget, and the 14 short-context
generate-then-match cells are 200 items x 32 new tokens. Nothing here needs the 8-GPU spec.

Why so few long-context cells: OLMo-2 is a 4096-position family (verified on 1B/7B/13B),
so bAbILong 8k/16k, every RULER 16384 cell, all of LongBench and all of InfiniteBench are
out of window. scripts/eval/audit_olmo2_window.py measured the exact seen_fraction each
cell would have, with OLMo-2's own tokenizer, before any GPU time was spent; the numbers are
in results/benchmarks/longctx/olmo-2-1b__window_audit.json and the driver's header lists
which cells were dropped and why.

GetJobLog is broken on this cluster, so the driver redirects everything to logs/qz/ and this
script records the returned job_id in logs/qz/OPENTRAIN_BASELINES_LEDGER.md at submit time,
appending under the "### OLMo-2" heading only (three other agents write their own sections
of the same file concurrently).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import re
import subprocess
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
EVIDENCE = REPO / ".omo/evidence/qz-training-skill"
LEDGER = REPO / "logs/qz/OPENTRAIN_BASELINES_LEDGER.md"

WS = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
SPEC1, SHM1 = "79fe954a-be92-4772-ac0b-94ad8a79b7bb", 100

NAME = "olmo2-1b-baseline"
HEADING = "### OLMo-2"
LOG = "logs/qz/olmo2-driver.log"


def body() -> dict:
    # `set -uo pipefail` WITHOUT -e: the driver's own failure must not abort this wrapper
    # before the log tail is printed, and GetJobLog cannot be used to recover it.
    # The driver exports every offline variable itself (including the mandatory
    # ${PYTHONPATH:+:$PYTHONPATH} guard -- a bare $PYTHONPATH under set -u kills the job
    # with no log at all), so the wrapper only has to reach it.
    cmd = (
        "set -uo pipefail\n"
        f"cd {REPO}\n"
        f"mkdir -p {REPO}/logs/qz {REPO}/results/benchmarks/longctx "
        f"{REPO}/results/benchmarks/baseline-olmo-2-1b {REPO}/results/benchmarks/ppl16\n"
        f"bash {REPO}/scripts/eval/run_olmo2_baseline.sh "
        f"> {REPO}/{LOG} 2>&1 || fail=1\n"
        f"tail -150 {REPO}/{LOG}\n"
        "exit ${fail:-0}\n"
    )
    return {
        "workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
        "name": NAME,
        "description": ("OLMo-2-0425-1B (fully-open-data baseline, rev a1847dff). bAbILong "
                        "qa1/4k+qa2/4k n=100; RULER 4096 niah_single_1/niah_multikey_1/vt/"
                        "qa_1/cwe n=200; 14-task generate-then-match n=200 @32 new tokens; "
                        "7-corpus PPL. 4096-position model: bAbILong 8k/16k, RULER 16384, "
                        "LongBench and InfiniteBench are out of window and deliberately not "
                        "run (see olmo-2-1b__window_audit.json)."),
        "framework": "pytorch", "command": cmd,
        # 8 h. The long-context part is ~1400 rows at <=4k, and the 1B model runs far faster
        # than the 2.9B relay arms measured at 7.3 s/sample, so this is large headroom.
        # The margin matters: run_open_context_bench writes its result ONCE after the loop,
        # so a shard that overruns the cap yields NOTHING, not a partial file.
        "max_running_time_ms": "28800000",
        "task_priority": 4,
        "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                              "instance_count": 1, "spec_id": SPEC1, "shm_gi": SHM1}],
    }


def record(job_id: str) -> None:
    """Append one line under `### OLMo-2`, leaving every other section byte-identical.

    Three other agents are appending to their own headings in this same file, so this must
    never rewrite the whole document: it splices a single line in after the heading and
    writes the rest back unchanged.
    """
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    line = f"OLMo-2 | all (longctx+gen+ppl) | `{job_id}` | {stamp} | `{LOG}`\n"
    # Inserting after the heading is collision-safe by POSITION, but read-modify-write is still
    # a race: another agent's append between our read and our write would be overwritten (a
    # concurrent edit to this file was in fact observed today). An advisory lock on a sidecar
    # closes that window for every writer that takes it. The lock file is separate from the
    # ledger so flock never truncates or creates content in the ledger itself.
    lockpath = LEDGER.with_suffix(LEDGER.suffix + ".lock")
    with open(lockpath, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            text = LEDGER.read_text()
            if HEADING not in text:
                raise SystemExit(f"FATAL: {HEADING!r} not found in {LEDGER}; refusing to guess")
            # Insert immediately after the heading line, so concurrent appends by other agents
            # to their own headings cannot collide with this one.
            pattern = re.compile(rf"^{re.escape(HEADING)}[ \t]*$", re.M)
            match = pattern.search(text)
            end = match.end()
            updated = text[:end] + "\n\n" + line.rstrip("\n") + text[end:]
            # Guard against a partial write leaving the shared ledger truncated: write a temp
            # file beside it and rename, which is atomic within a directory.
            tmp = LEDGER.with_suffix(LEDGER.suffix + ".tmp-olmo2")
            tmp.write_text(updated)
            tmp.replace(LEDGER)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    print(f"[ledger] {LEDGER}\n[line] {line.strip()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    payload = EVIDENCE / "olmo2_1b_baseline.json"
    payload.write_text(json.dumps(body(), indent=2, ensure_ascii=False) + "\n")
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
    record(job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

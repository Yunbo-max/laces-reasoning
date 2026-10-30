#!/usr/bin/env python3
"""Submit the pythia-2.8b benchmark suite to QZ as several independent shards.

Sharded on purpose. The brief records a 14 h job that died unexplained on this cluster, so
each shard is a separate job with its own result file and its own log on disk; one shard
dying never costs the others. `GetJobLog` is broken here, so every step redirects to disk.

Shard order is priority order, not file order: the short-context capability suite and PPL go
first because pythia-2.8b is in-window there and it is the only genuinely size-matched
comparison against our 2.9 B renderer. The long-context shards are expected to come back
out-of-window and exist to produce the truncation audit.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222"
REPO = f"{ROOT}/research/laces-reasoning"
EVID = Path(f"{REPO}/.omo/evidence/pythia-2.8b")
LEDGER = Path(f"{REPO}/logs/qz/OPENTRAIN_BASELINES_LEDGER.md")

WS = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
SPEC_1GPU, SHM_1GPU = "79fe954a-be92-4772-ac0b-94ad8a79b7bb", 100

REV = "2a259cdd96a4beb1cdf467512e3904197345f6a9"
MODEL = f"{ROOT}/hf_cache/hub/models--EleutherAI--pythia-2.8b/snapshots/{REV}"
BENCH = "scripts/eval/run_open_context_bench.py"
TASKDATA = f"{ROOT}/research/DiffRwkv/baseline/Cola-DLM/generate_task_data"
LONGCTX = f"{REPO}/results/benchmarks/longctx"
INWIN = f"{REPO}/results/benchmarks/longctx_inwindow"
LOGD = f"{REPO}/logs/qz"

BABI = (f"{ROOT}/hf_cache/hub/datasets--RMT-team--babilong/snapshots/"
        "ee0d588794c7ac098062ee0d247c733d62e94fe2/data")
RULER = (f"{ROOT}/hf_cache/hub/datasets--simonjegou--ruler/snapshots/"
         "24adceac8a0e6532936e8d721cd9e9084d2e4686")
LB = f"{ROOT}/longbench_local/extract/data"
LBSPEC = f"{ROOT}/longbench_local/pinned_spec"
INF = (f"{ROOT}/hf_cache/hub/datasets--xinrongzhang2022--InfiniteBench/snapshots/"
       "90f0394333616266d9fe85824ceaf505093cbaa5")
MEM = (f"{ROOT}/hf_cache/hub/datasets--xiaowu0162--longmemeval/snapshots/"
       "2ec2a557f339b6c0369619b1ed5793734cc87533")

# The `${PYTHONPATH:+:$PYTHONPATH}` form is mandatory: PYTHONPATH is unset in the compute
# container and a bare `$PYTHONPATH` under `set -u` aborts the shell before any log redirect,
# killing the job with no log at all.
PREAMBLE = f"""set -uo pipefail
fail=0
mkdir -p {LOGD} {LONGCTX} {INWIN} {REPO}/results/benchmarks/ppl16 \\
         {REPO}/results/benchmarks/baseline-pythia-28b \\
         {REPO}/results/benchmarks/fix2choices/baseline-pythia-28b
cd {REPO}
export HF_HOME={ROOT}/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH={ROOT}/shared_pylibs${{PYTHONPATH:+:$PYTHONPATH}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
"""


def step(cmd: str, log: str) -> str:
    return (f'echo "=== {log} === $(date -u)"\n'
            f"CUDA_VISIBLE_DEVICES=0 {cmd} > {LOGD}/{log}.log 2>&1 || fail=1\n")


def genmatch(outdir: str, tasks: str, log: str) -> str:
    return step(
        f"python3 scripts/eval/eval_generic_gen_match.py "
        f"--model_dir {MODEL} --task_data_dir {TASKDATA} --output_dir {outdir} "
        f"--tasks {tasks} --max_samples 200 --max_new_tokens 32 --temperature 0.0", log)


def bench(suite: str, datasets: str, data_dir: str, out: str, log: str,
          max_samples: int, mit: int, mnt: int, extra: str = "") -> str:
    return step(
        f"python3 -u {BENCH} --suite {suite} --datasets {datasets} --data_dir {data_dir} "
        f"{extra}--model_dir {MODEL} --arm raw --output {out} "
        f"--max_samples {max_samples} --max_input_tokens {mit} --max_new_tokens {mnt} "
        f"--seed 42 --device cuda", log)


# --- shards, in priority order ----------------------------------------------------------
# A: in-window and size-matched -> the numbers that matter. B: in-window length sweep.
# C/D/E: briefed grid, expected out-of-window; run for the truncation audit.
SHARDS: dict[str, dict] = {
    "A-short-ppl": dict(
        desc="pythia-2.8b short-context capability (14 columns) + PPL 7 corpora; all in-window",
        hours=6,
        steps=[
            genmatch(f"{REPO}/results/benchmarks/baseline-pythia-28b",
                     "mmlu,obqa,race,squad,story_cloze,hellaswag,lambada,siqa",
                     "pythia28-genmatch-main"),
            genmatch(f"{REPO}/results/benchmarks/fix2choices/baseline-pythia-28b",
                     "arc_easy,arc_challenge,winogrande,sciq,commonsenseqa,piqa",
                     "pythia28-genmatch-fix2"),
            # eval_lm_ppl.py is run UNMODIFIED: its dtype probe silently falls back to
            # bfloat16 (os.path.join with no module-level `import os`, NameError swallowed by
            # `except Exception: pass`). All 8 existing ppl16 HF rows carry that same
            # fallback, so patching it here would make pythia the only fp16 row in a bf16
            # table. Comparability beats local correctness; the bug is in the ledger.
            step(f"python3 scripts/eval/eval_lm_ppl.py --impl hf --model_dir {MODEL} "
                 f"--mode raw --corpora wikitext2,wikitext103,pg19,lm1b,ptb,wmt14,xsum "
                 f"--max_files 100 --output {REPO}/results/benchmarks/ppl16/pythia-2.8b.json",
                 "ppl16-pythia-2.8b"),
        ]),
    "B-babilong-inwindow": dict(
        desc="pythia-2.8b bAbILong qa1/qa2 at 0k/1k/2k -- the lengths that fit its 2048 window",
        hours=4,
        steps=[bench("babilong", "qa1/0k,qa1/1k,qa1/2k,qa2/0k,qa2/1k,qa2/2k", BABI,
                     f"{INWIN}/pythia-2.8b__babilong_inwindow.json",
                     "pythia28-babilong-inwindow", 200, 16384, 32)]),
    "C-babilong-briefed": dict(
        desc="pythia-2.8b bAbILong briefed grid 4k/8k/16k -- expected OUT OF WINDOW, run for audit",
        hours=4,
        steps=[bench("babilong", "qa1/4k,qa1/8k,qa1/16k,qa2/4k", BABI,
                     f"{LONGCTX}/pythia-2.8b__babilong.json",
                     "pythia28-babilong", 200, 16384, 32)]),
    "D-ruler": dict(
        desc="pythia-2.8b RULER 4096+16384 -- expected OUT OF WINDOW (min shipped length 4096)",
        hours=6,
        steps=[bench("ruler",
                     "4096/niah_single_1,4096/niah_multikey_1,4096/vt,4096/cwe,4096/fwe,"
                     "4096/qa_1,16384/niah_single_1,16384/vt,16384/cwe", RULER,
                     f"{LONGCTX}/pythia-2.8b__ruler.json", "pythia28-ruler",
                     200, 16384, 128)]),
    "E-lb-inf-mem": dict(
        desc="pythia-2.8b LongBench + InfiniteBench + MemEval -- expected OUT OF WINDOW",
        hours=6,
        steps=[
            bench("longbench", "narrativeqa,qasper,multifieldqa_en,hotpotqa", LB,
                  f"{LONGCTX}/pythia-2.8b__longbench.json", "pythia28-longbench",
                  50, 32768, 512, extra=f"--longbench_config_dir {LBSPEC} "),
            bench("infinitebench",
                  "longbook_qa_eng,longbook_choice_eng,kv_retrieval,code_debug,math_find", INF,
                  f"{LONGCTX}/pythia-2.8b__infinitebench.json", "pythia28-infinitebench",
                  50, 32768, 128),
            # MemEval's 50-row slice has a 0.220 random floor and known scorer defects; run
            # for completeness, reported as uninformative at or below that floor.
            bench("memeval", "longmemeval_oracle", MEM,
                  f"{LONGCTX}/pythia-2.8b__memeval.json", "pythia28-memeval",
                  50, 131072, 128),
        ]),
}


def body(key: str) -> dict:
    sh = SHARDS[key]
    return {
        "workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
        "name": f"pythia28-{key}"[:60],
        "description": sh["desc"],
        "framework": "pytorch",
        "command": PREAMBLE + "".join(sh["steps"]) + f'echo "DONE fail=$fail $(date -u)"\nexit $fail\n',
        "max_running_time_ms": str(sh["hours"] * 3600 * 1000),
        "task_priority": 4,
        "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                              "instance_count": 1, "spec_id": SPEC_1GPU, "shm_gi": SHM_1GPU}],
    }


def append_ledger(rows: list[str]) -> None:
    """Append job rows under the '### Pythia' table only.

    Three other agents write their own sections of this file concurrently, so this does a
    targeted insert at the end of the Pythia table rather than rewriting the file.
    """
    text = LEDGER.read_text()
    marker = "| family | suite | job_id | submitted UTC | log path |\n|---|---|---|---|---|\n"
    idx = text.index(marker) + len(marker)
    LEDGER.write_text(text[:idx] + "".join(rows) + text[idx:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    keys = list(SHARDS) if a.shards == "all" else a.shards.split(",")
    EVID.mkdir(parents=True, exist_ok=True)

    rows, failed = [], []
    for k in keys:
        b = body(k)
        fn = EVID / f"job_{b['name']}.json"
        fn.write_text(json.dumps(b, indent=2, ensure_ascii=False) + "\n")
        r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{fn}", "--dry-run"],
                           capture_output=True, text=True, timeout=180)
        if "DRY RUN" not in r.stdout:
            print(f"[{k}] DRY-RUN FAILED\n{r.stdout[-800:]}\n{r.stderr[-400:]}")
            failed.append(k)
            continue
        print(f"[{k}] dry-run ok  body={fn}")
        if a.dry_run:
            continue
        r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{fn}"],
                           capture_output=True, text=True, timeout=300)
        try:
            jid = json.loads(r.stdout)["Result"]["job_id"]
        except Exception:
            print(f"[{k}] SUBMIT FAILED\n{r.stdout[:600]}\n{r.stderr[:400]}")
            failed.append(k)
            continue
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"[{k}] job_id={jid}")
        rows.append(f"| Pythia | {k} | `{jid}` | {ts} | `{LOGD}/pythia28-*.log` |\n")
        # Record every job id the moment it exists: ListJobs paging permanently drops old
        # jobs, so an id that is not written down here is unrecoverable.
        append_ledger([rows[-1]])

    if rows:
        print(f"\nrecorded {len(rows)} job id(s) in {LEDGER}")
    if failed:
        print(f"FAILED shards: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

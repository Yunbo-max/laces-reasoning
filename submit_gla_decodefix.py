#!/usr/bin/env python3
"""Audit + rebuild the gla-1.3b long-context cells that were decoded with `--decode hf`.

Owned defect class: `--decode hf` is measurably wrong for fla recurrent models. rwkv7 was
quarantined; gla-1.3b is the same architecture family (`model_type: gla`, needs
`import fla.models` to load, and `patch_fla_recurrent_generate` DOES fire on it) and was not.

Three jobs:

  gla-decode-fingerprint-bos   20-row hf-vs-loop first-token/full-sequence agreement at the
                               CANONICAL BOS setting. The existing gla-fingerprint.json was
                               taken with add_bos=False, but every post-07:30 canonical cell
                               prepends BOS, so that fingerprint does not describe the
                               published rows. Cheap (20 rows x 3 decodes).
  gla-loopfix-ruler            RULER, its own single-GPU job. Measured 3.38 GPU*h on hf and
                               the loop path is slower per token, so it must not share a lane
                               with the cheap cells.
  gla-loopfix-cheap            babilong + memeval + infinitebench, one lane each on a 4-GPU
                               spec. Measured hf costs 0.16 / 0.09 / 1.89 GPU*h.

Protocol per suite is READ OUT of each existing canonical file's `config`, never guessed:
  babilong      --max_samples 200 --max_input_tokens 16384  --max_new_tokens 32
  ruler         --max_samples 200 --max_input_tokens 16384  --max_new_tokens 128
  longbench     --max_samples 50  --max_input_tokens 32768  --max_new_tokens 512   (NOT run
                here: another agent's job-f3bfc9e4 / job-ff9b1460 already produced loop rows)
  infinitebench --max_samples 50  --max_input_tokens 32768  --max_new_tokens 128
  memeval       --max_samples 50  --max_input_tokens 131072 --max_new_tokens 128

Everything writes to results/benchmarks/longctx_decodefix/ (a staging dir this line owns).
NOTHING is copied to canonical by the job -- the copy is a separate, verified step, because a
concurrently running job I do not own (job-f175107f lc-200-postfix-fla) also `cp -f`s into
results/benchmarks/longctx/ with the hf default.

  python3 submit_gla_decodefix.py --plan
  python3 submit_gla_decodefix.py --submit --dry-run
  python3 submit_gla_decodefix.py --submit
  python3 submit_gla_decodefix.py --status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
HF_CACHE = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache")
SHARED = "/inspire/hdd/global_user/zhangjiaquan-253108540222/shared_pylibs"
LOGS = REPO / "logs/qz"
STAGE = REPO / "results/benchmarks/longctx_decodefix"
FZ = REPO / "results/benchmarks/falsify_zero"
EVID = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv"
            "/.omo/evidence/qz-training-skill")

WS = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
SPEC1 = ("79fe954a-be92-4772-ac0b-94ad8a79b7bb", 100)    # 1 GPU
SPEC4 = ("8a53ac21-299a-4dee-85e9-9c04a544cf8d", 900)    # 4 GPU

GLA = str(HF_CACHE / "hub/models--fla-hub--gla-1.3B-100B/snapshots"
          "/46b15820a4df269e99aed9d709e017677c15d24b")
BABI = str(HF_CACHE / "hub/datasets--RMT-team--babilong/snapshots"
           "/ee0d588794c7ac098062ee0d247c733d62e94fe2/data")
RULER = str(HF_CACHE / "hub/datasets--simonjegou--ruler/snapshots"
            "/24adceac8a0e6532936e8d721cd9e9084d2e4686")
INFB = str(HF_CACHE / "hub/datasets--xinrongzhang2022--InfiniteBench/snapshots"
           "/90f0394333616266d9fe85824ceaf505093cbaa5")
MEMEV = str(HF_CACHE / "hub/datasets--xiaowu0162--longmemeval/snapshots"
            "/2ec2a557f339b6c0369619b1ed5793734cc87533")
# LongBench is NOT in hf_cache -- it was extracted to its own tree. Both path and dataset list
# are read out of the existing rwkv7 longbench result's `config`, never guessed.
LONGB = "/inspire/hdd/global_user/zhangjiaquan-253108540222/longbench_local/extract/data"
LONGB_SETS = ("hotpotqa,2wikimqa,musique,narrativeqa,multifieldqa_en,passage_retrieval_en,"
              "gov_report,qmsum,multi_news")

# The ${PYTHONPATH:+:$PYTHONPATH} guard is mandatory: PYTHONPATH is UNSET in this image and
# under `set -u` a bare $PYTHONPATH aborts the shell before any log redirect exists, killing
# the job with NO log at all.
PREAMBLE = (
    "set -uo pipefail\n"
    f"mkdir -p {LOGS} {STAGE} {FZ}\n"
    f"cd {REPO}\n"
    f"export HF_HOME={HF_CACHE}\n"
    "export HF_HUB_OFFLINE=1\n"
    "export TRANSFORMERS_OFFLINE=1\n"
    f"export PYTHONPATH={SHARED}${{PYTHONPATH:+:$PYTHONPATH}}\n"
    "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
)

BENCH = "python scripts/eval/run_open_context_bench.py"
BABI_SETS = "qa1/4k,qa1/8k,qa1/16k,qa2/4k"
RULER_SETS = ("4096/niah_single_1,4096/niah_multikey_1,4096/vt,4096/cwe,4096/fwe,"
              "4096/qa_1,16384/niah_single_1,16384/vt,16384/cwe")
INFB_SETS = "longbook_qa_eng,longbook_choice_eng,kv_retrieval,code_debug,math_find"


def cell(suite: str, data_dir: str, sets: str, ms: int, mit: int, mnt: int) -> str:
    """One `--decode loop` cell, logged to its own file, never `cp`d to canonical."""
    log = LOGS / f"dfix-gla-1.3b-{suite}.log"
    out = STAGE / f"gla-1.3b__{suite}.json"
    return (f"  {BENCH} --suite {suite} --datasets {sets} --data_dir {data_dir} "
            f"--model_dir {GLA} --decode loop "
            f"--max_samples {ms} --max_input_tokens {mit} --max_new_tokens {mnt} "
            f"--output {out} > {log} 2>&1; echo \"{suite} exit=$?\"\n")


def fingerprint_body() -> dict:
    """20-row hf-vs-loop agreement at the canonical BOS setting.

    `--add_bos` is the whole point of re-taking this: falsify_zero/gla-fingerprint.json was
    measured with add_bos=False (its row 0 prefix is 3977 tokens), while gla-loop-bos.json and
    the canonical cells are at 3978 with BOS. An agreement number from the wrong BOS setting
    cannot arbitrate the published rows.
    """
    log = LOGS / "dfix-gla-fingerprint-bos.log"
    out = FZ / "gla-fingerprint-bos.json"
    cmd = PREAMBLE + (
        "echo '=== gla-1.3b hf-vs-loop fingerprint, add_bos ON (canonical setting) ==='\n"
        "CUDA_VISIBLE_DEVICES=0 python scripts/eval/falsify_zero_baselines.py "
        f"--mode fingerprint --model_dir {GLA} --data_dir {BABI} "
        f"--dataset qa1/4k --budget_int 16384 --max_samples 20 --max_new_tokens 32 "
        f"--add_bos --output {out} > {log} 2>&1; echo \"fingerprint exit=$?\"\n"
        f"tail -40 {log}\n"
    )
    spec, shm = SPEC1
    return {"workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
            "name": "gla-decode-fingerprint-bos",
            "description": (
                "gla-1.3b is an fla recurrent model (model_type=gla, patch_fla_recurrent_"
                "generate fires) whose five canonical long-context cells carry NO decode key, "
                "i.e. they ran the hf default that is known-wrong for this family. This is the "
                "cheap arbitration before spending GPU on full re-runs: 20 bAbILong qa1/4k rows "
                "x {unpatched generate, patched generate, greedy loop}, agreement reported per "
                "prefix length. Re-taken with --add_bos because the existing gla fingerprint "
                "used add_bos=False while the published cells prepend BOS."),
            "framework": "pytorch", "command": cmd,
            "max_running_time_ms": "7200000", "task_priority": 4,
            "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                                  "instance_count": 1, "spec_id": spec, "shm_gi": shm}]}


def ruler_body() -> dict:
    """RULER alone on 1 GPU: measured 3.38 GPU*h with hf, and loop is slower per token."""
    cmd = PREAMBLE + (
        "echo '=== gla-1.3b RULER, --decode loop, 200 samples ==='\n"
        "( export CUDA_VISIBLE_DEVICES=0\n"
        + cell("ruler", RULER, RULER_SETS, 200, 16384, 128) +
        ") &\nwait\necho '=== RULER LOOP COMPLETE ==='\nexit 0\n"
    )
    spec, shm = SPEC1
    return {"workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
            "name": "gla-loopfix-ruler",
            "description": (
                "Rebuild gla-1.3b RULER with --decode loop. The canonical cell "
                "(results/benchmarks/longctx/gla-1.3b__ruler.json) has no decode key and was "
                "produced by the hf default, which is measurably wrong for fla recurrent "
                "models. Same protocol as the existing cell so the rows stay comparable: 9 "
                "datasets, --max_samples 200 --max_input_tokens 16384 --max_new_tokens 128, "
                "seed 42. Own single-GPU job because hf already cost 3.38 GPU*h here and the "
                "loop path is slower per token. Writes ONLY to longctx_decodefix/."),
            "framework": "pytorch", "command": cmd,
            "max_running_time_ms": "86400000", "task_priority": 4,
            "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                                  "instance_count": 1, "spec_id": spec, "shm_gi": shm}]}


def cheap_body() -> dict:
    """babilong + memeval + infinitebench, one lane each on a 4-GPU spec.

    Each lane gets its OWN CUDA_VISIBLE_DEVICES; three lanes on a 4-GPU spec, never more
    lanes than GPUs -- a 2-lane pack on the 1-GPU spec already lost a lane once to
    "No CUDA GPUs are available".
    """
    lanes = [
        ("babilong", cell("babilong", BABI, BABI_SETS, 200, 16384, 32)),
        ("memeval", cell("memeval", MEMEV, "longmemeval_oracle", 50, 131072, 128)),
        ("infinitebench", cell("infinitebench", INFB, INFB_SETS, 50, 32768, 128)),
    ]
    body = PREAMBLE + "echo '=== gla-1.3b loop decode: 3 cells on 3 lanes ==='\n"
    for gpu, (name, c) in enumerate(lanes):
        body += f"# ---- lane {gpu}: {name} ----\n( export CUDA_VISIBLE_DEVICES={gpu}\n{c}) &\n"
    body += "wait\necho '=== CHEAP LOOP CELLS COMPLETE ==='\nexit 0\n"
    spec, shm = SPEC4
    return {"workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
            "name": "gla-loopfix-cheap",
            "description": (
                "Rebuild gla-1.3b babilong/memeval/infinitebench with --decode loop. All three "
                "canonical cells have no decode key and ran the hf default, which is wrong for "
                "fla recurrent models. Protocol read out of each existing file's config so the "
                "cells stay comparable: babilong 200/16384/32, memeval 50/131072/128, "
                "infinitebench 50/32768/128, seed 42. Note babilong's splits only contain 100 "
                "rows each, so n=100 regardless of --max_samples 200. Writes ONLY to "
                "longctx_decodefix/."),
            "framework": "pytorch", "command": body,
            "max_running_time_ms": "86400000", "task_priority": 4,
            "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                                  "instance_count": 1, "spec_id": spec, "shm_gi": shm}]}


def rwkv7_babilong_body() -> dict:
    """Task 4: rebuild rwkv7's babilong cell through the REAL harness with --decode loop.

    Preferred over copying falsify_zero/rwkv7-loop.json, because that file came from the probe
    script (`mode: rerun`) and so has no `decode_path` / `decode_is_trustworthy` / `audit` /
    `seen_fraction` keys, and a different row schema. Running the harness gives a cell that is
    structurally identical to every other published cell AND carries the trust booleans.

    Affordable: the probe's 400-row loop decode took 835 s = 0.23 GPU*h, essentially the same
    as the quarantined hf run (0.24 GPU*h).

    `--max_samples 200` is passed to match the other cells' config, but bAbILong's splits
    contain exactly 100 rows each (verified by reading the JSON), so the honest published n is
    100 for every dataset. The number is NOT relabelled.
    """
    log = LOGS / "dfix-rwkv7-world3-babilong.log"
    out = STAGE / "rwkv7-world3__babilong.json"
    rwkv7 = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/models"
             "/RWKV7-Goose-World3-2.9B-HF")
    cmd = PREAMBLE + (
        "echo '=== rwkv7-world3 bAbILong, --decode loop (rebuilds the quarantined cell) ==='\n"
        "( export CUDA_VISIBLE_DEVICES=0\n"
        f"  {BENCH} --suite babilong --datasets {BABI_SETS} --data_dir {BABI} "
        f"--model_dir {rwkv7} --decode loop "
        f"--max_samples 200 --max_input_tokens 16384 --max_new_tokens 32 "
        f"--output {out} > {log} 2>&1; echo \"babilong exit=$?\"\n"
        ") &\nwait\necho '=== RWKV7 BABILONG LOOP COMPLETE ==='\nexit 0\n"
    )
    spec, shm = SPEC1
    return {"workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
            "name": "rwkv7-loopfix-babilong",
            "description": (
                "Rebuild the quarantined rwkv7-world3 bAbILong cell properly, through "
                "run_open_context_bench.py with --decode loop, so the published cell carries "
                "decode_path=loop and decode_is_trustworthy=True instead of only existing as a "
                "quarantine README plus a probe-script JSON. Reference numbers to reproduce "
                "(falsify_zero/rwkv7-loop.json, decode=loop, n=100): qa1_4k 41.0, qa1_8k 18.0, "
                "qa1_16k 17.0, qa2_4k 20.0. bAbILong splits hold only 100 rows, so n=100 "
                "regardless of --max_samples 200. Writes ONLY to longctx_decodefix/."),
            "framework": "pytorch", "command": cmd,
            "max_running_time_ms": "21600000", "task_priority": 4,
            "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                                  "instance_count": 1, "spec_id": spec, "shm_gi": shm}]}


def rwkv7_rest_body() -> dict:
    """Task 5: rwkv7-world3 ruler + infinitebench + longbench with --decode loop.

    The task brief said to attempt these only if step 3 left budget, and to report the measured
    cost otherwise. It left plenty: the hf-decode cost estimate (~4.6 GPU*h for ruler) was wrong
    for the LOOP path. Measured on the gla loop ruler job while it ran, by counting score rows
    over a 90 s window: 0.78 s/row => 1800 rows = 0.39 GPU*h, roughly 9x cheaper than the hf run
    it replaces (3.38 GPU*h). The loop is slower per *token* but the hf path was re-feeding the
    whole prefix every step, which is quadratic -- so for long prefixes the loop is far FASTER,
    not slower. Scaling 0.39 h by rwkv7's 2.9B/1.3B parameter ratio gives well under 1 GPU*h.

    longbench is included because a loop-decoded rwkv7 longbench ALREADY exists
    (longctx_fix/rwkv7-world3-loop__longbench.json, macro 0.0955 vs the hf 0.0291 -- a 3.28x
    understatement of a baseline, 0/450 predictions byte-identical). That file is trustworthy on
    its numbers but predates the flag, so it carries no decode_path/decode_is_trustworthy. Re-run
    it here so the published cell is self-describing rather than needing a footnote.

    Three lanes on the 4-GPU spec, one GPU each.
    """
    rwkv7 = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/models"
             "/RWKV7-Goose-World3-2.9B-HF")

    def rw_cell(suite, data_dir, sets, ms, mit, mnt):
        log = LOGS / f"dfix-rwkv7-world3-{suite}.log"
        out = STAGE / f"rwkv7-world3__{suite}.json"
        return (f"  {BENCH} --suite {suite} --datasets {sets} --data_dir {data_dir} "
                f"--model_dir {rwkv7} --decode loop "
                f"--max_samples {ms} --max_input_tokens {mit} --max_new_tokens {mnt} "
                f"--output {out} > {log} 2>&1; echo \"{suite} exit=$?\"\n")

    lanes = [
        ("ruler", rw_cell("ruler", RULER, RULER_SETS, 200, 16384, 128)),
        ("infinitebench", rw_cell("infinitebench", INFB, INFB_SETS, 50, 32768, 128)),
        ("longbench", rw_cell("longbench", LONGB, LONGB_SETS, 50, 32768, 512)),
    ]
    body = PREAMBLE + "echo '=== rwkv7-world3 loop decode: ruler + infinitebench + longbench ==='\n"
    for gpu, (name, c) in enumerate(lanes):
        body += f"# ---- lane {gpu}: {name} ----\n( export CUDA_VISIBLE_DEVICES={gpu}\n{c}) &\n"
    body += "wait\necho '=== RWKV7 REST LOOP CELLS COMPLETE ==='\nexit 0\n"
    spec, shm = SPEC4
    return {"workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
            "name": "rwkv7-loopfix-rest",
            "description": (
                "rwkv7-world3 ruler + infinitebench + longbench with --decode loop. These cells "
                "are absent from results/benchmarks/longctx/ entirely; rwkv7 is the model where "
                "the fla decode defect is most severe (first generated token already wrong, so "
                "patch_fla_recurrent_generate is necessary but not sufficient), so they must "
                "never be produced with the hf default. Protocol matches every other canonical "
                "cell: ruler 200/16384/128, infinitebench 50/32768/128, longbench 50/32768/512, "
                "seed 42. Cost basis: the loop path was MEASURED at 0.78 s/row on gla ruler "
                "(0.39 GPU*h for 1800 rows), ~9x cheaper than the 3.38 GPU*h hf run, because the "
                "hf defect re-fed the whole prefix every step. Writes ONLY to longctx_decodefix/."),
            "framework": "pytorch", "command": body,
            "max_running_time_ms": "86400000", "task_priority": 4,
            "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                                  "instance_count": 1, "spec_id": spec, "shm_gi": shm}]}


def rwkv7_memeval_body() -> dict:
    """rwkv7 memeval with --decode loop. Found during the full canonical audit.

    `results/benchmarks/longctx/rwkv7-world3__memeval.json` is the LAST fla-recurrent canonical
    cell still on the hf path (audit of all 45 canonical cells: every other model is attention
    (gemma2/llama/qwen2/phi) or native-transformers mamba, neither of which is affected).

    Its correction is large and already estimable without GPU: a companion file
    `longctx/rwkv7-world3__memeval_loopdecode.json` holds 50 loop-decoded predictions as a bare
    {gold, pred} list -- no scores, no config, so it cannot be published as a cell. Scoring those
    predictions with the harness's own `longmemeval_score_approx` gives **0.16 vs the published
    0.04 (4x), with 0/50 predictions byte-identical**. This job produces the real cell so the
    number is backed by a harness run rather than by my offline rescoring of someone's probe dump.

    Cheap: gla memeval on the loop path measured 0.075 GPU*h.
    """
    rwkv7 = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/models"
             "/RWKV7-Goose-World3-2.9B-HF")
    log = LOGS / "dfix-rwkv7-world3-memeval.log"
    out = STAGE / "rwkv7-world3__memeval.json"
    cmd = PREAMBLE + (
        "echo '=== rwkv7-world3 memeval, --decode loop ==='\n"
        "( export CUDA_VISIBLE_DEVICES=0\n"
        f"  {BENCH} --suite memeval --datasets longmemeval_oracle --data_dir {MEMEV} "
        f"--model_dir {rwkv7} --decode loop "
        f"--max_samples 50 --max_input_tokens 131072 --max_new_tokens 128 "
        f"--output {out} > {log} 2>&1; echo \"memeval exit=$?\"\n"
        ") &\nwait\necho '=== RWKV7 MEMEVAL LOOP COMPLETE ==='\nexit 0\n"
    )
    spec, shm = SPEC1
    return {"workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
            "name": "rwkv7-loopfix-memeval",
            "description": (
                "rwkv7-world3 memeval with --decode loop. This is the last fla-recurrent cell in "
                "results/benchmarks/longctx/ still produced by the hf default (verified by "
                "auditing the model_type of every canonical cell). Offline scoring of an existing "
                "loop-decode prediction dump with the harness's own longmemeval_score_approx "
                "gives 0.16 against the published hf 0.04, with 0/50 predictions byte-identical, "
                "so the hf number understates this baseline ~4x. Protocol matches the published "
                "cell exactly: 50/131072/128, seed 42. Writes ONLY to longctx_decodefix/."),
            "framework": "pytorch", "command": cmd,
            "max_running_time_ms": "21600000", "task_priority": 4,
            "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                                  "instance_count": 1, "spec_id": spec, "shm_gi": shm}]}


def gla_longbench_body() -> dict:
    """The one cell in this defect class never rebuilt: canonical gla longbench is UNPATCHED hf.

    Every other gla cell ran *patched* hf, where the residual decode error is small. This one
    predates `patch_fla_recurrent_generate` entirely (`lc-gla-1.3b-longbench.log` has no
    "patch applied" line), so it carries the raw defect: every decode step re-fed the whole
    prefix into an already-advanced recurrent state. It is the single worst-provenance cell I own.

    Loop rows for it DO exist, from another agent's jobs job-f3bfc9e4 (no BOS, macro 0.0735) and
    job-ff9b1460 (BOS, 0.0749), and I used them for the 2x2 axis decomposition. But neither is
    publishable as a canonical cell: both predate the transparency fields, so `decode_path` and
    `decode_is_trustworthy` are **absent**, not False. Per the rule this line keeps re-learning,
    a missing key is "unknown, go read the submit command", never a value -- and a canonical cell
    whose decode provenance cannot be gated on is exactly what this whole defect class was about.
    So the fix is to re-run it through the current harness rather than promote an ungated file.

    Protocol is read out of the existing canonical cell's `config`, not guessed: 50 / 32768 / 512,
    the 9 LongBench datasets in that file's own order, no BOS (matching the canonical cell; the
    BOS axis was separately measured at +0.0013 and is not what this job is testing).

    Cost: MEASURED 0.691 GPU*h for the same 450 rows on the loop path (job-f3bfc9e4's
    elapsed_seconds), so a 1-GPU spec is right. Expectation recorded BEFORE the run: macro within
    noise of 0.0735. Publishing it moves the canonical number 0.0726 -> ~0.0735, i.e. it barely
    changes the score; the point is provenance, not the delta.
    """
    log = LOGS / "dfix-gla-1.3b-longbench.log"
    out = STAGE / "gla-1.3b__longbench.json"
    cmd = PREAMBLE + (
        "echo '=== gla-1.3b longbench, --decode loop (canonical cell is UNPATCHED hf) ==='\n"
        "( export CUDA_VISIBLE_DEVICES=0\n"
        f"  {BENCH} --suite longbench --datasets {LONGB_SETS} --data_dir {LONGB} "
        f"--model_dir {GLA} --decode loop "
        f"--max_samples 50 --max_input_tokens 32768 --max_new_tokens 512 "
        f"--output {out} > {log} 2>&1; echo \"longbench exit=$?\"\n"
        ") &\nwait\necho '=== GLA LONGBENCH LOOP COMPLETE ==='\nexit 0\n"
    )
    spec, shm = SPEC1
    return {"workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
            "name": "gla-loopfix-longbench",
            "description": (
                "gla-1.3b longbench with --decode loop. The canonical cell for this suite is the "
                "only one in the affected set produced by UNPATCHED hf generate (its log has no "
                "'patch applied' line), i.e. the raw fla decode defect where every step re-fed the "
                "whole prefix; measured prediction agreement against the loop is 134/450 vs "
                "354/450 for patched hf. Loop rows from earlier jobs exist but lack the "
                "decode_path / decode_is_trustworthy fields, so they cannot be gated and are not "
                "promotable -- hence a fresh run through the current harness. Protocol matches the "
                "canonical cell exactly: 50/32768/512, 9 datasets, no BOS, seed 42. Cost basis: "
                "0.691 GPU*h measured for the same 450 rows on the loop path. Writes ONLY to "
                "longctx_decodefix/."),
            "framework": "pytorch", "command": cmd,
            "max_running_time_ms": "21600000", "task_priority": 4,
            "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                                  "instance_count": 1, "spec_id": spec, "shm_gi": shm}]}


def eosfix_longbench_body() -> dict:
    """The stop token was never honoured: rebuild BOTH cells it hit -- a baseline AND our own arm.

    Found 2026-09-21 while checking a peer's never-EOS table, where hf rwkv7 free-ran 425/450 but
    my loop cell was 450/450 pinned -- same model, same BOS, so the asymmetry had to be my own
    decode path. `greedy_loop` (baselines) and both relay runtimes (our arms) were handed only
    `tokenizer.eos_token_id`. For RWKV7-Goose-World3-2.9B-HF -- which is the baseline *and* the
    frozen decoder inside our relay checkpoints -- the three declared sources disagree:

        tokenizer.eos_token_id = 65530   <- what was used; never appears in any generation
        generation_config      = 0       <- '<|rwkv_tokenizer_end_of_text|>', the real one
        config.json            = 2       <- '!'

    The tokenizer's `eos_token` is the string '\\n\\n', which resolves to an id the model never
    emits, so the stop check could not fire and every row ran to its full budget.

    MEASURED (id 0 occurrences inside published generations; macro from rescoring the truncated
    prefix with the harness's own metric_for):

        cell                     rows  id0   rows changed      macro           delta
        rwkv7-world3 (baseline)   450  141   82 (67u/15d)   0.0955 -> 0.1197  +0.0242
        global-dit-50k (OUR ARM)  450   79   58 (55u/ 3d)   0.0955 -> 0.1198  +0.0243
        champ-dynamic-30k (ours)  450    0    0             0.0901 -> 0.0901  +0.0000

    DIRECTION -- the reason both cells are in one job. The defect hit a competitor and our own arm
    by almost exactly the same amount, so the LongBench comparison barely moves: 0.0955 vs 0.0955
    before, ~0.1197 vs ~0.1198 after. Fixing it hands us nothing. Re-running only the baseline
    would be a biased half-fix, and re-running only our arm would be worse; both go together.

    NOT re-run, and why. champ-dynamic already stops on its own (33 distinct generation lengths,
    0 rows containing id 0) and is unaffected. gla is unaffected for a different reason: its
    tokenizer's '</s>' resolves to id 2, matching its config and generation_config, so its loop
    had the right id all along -- and it emits it in 0/2950 rows. The other four suites are
    unaffected because their metrics are containment/recall, where trailing junk cannot remove a
    match; verified empirically rather than assumed, by rescoring every row of both arms: RULER
    0/1800 changed (despite id 0 in 390 global-dit / 671 rwkv7 rows), bAbILong 0/400,
    InfiniteBench 0/250. So this is a 2-cell job, not a 10-cell one.

    Why re-run rather than publish the offline rescore: truncating an existing prediction at id 0
    is exact for greedy decode only up to that point -- the model would have spent the rest of its
    budget differently had it stopped. The rescore is a faithful preview, not the cell.

    Protocol is read out of each published cell's own `config`: 50 / 32768 / 512, seed 42, the 9
    LongBench datasets, same data_dir and pinned_spec. Baseline keeps `--arm raw --decode loop`;
    our arm keeps `--arm global --expected_step 50000`. Cost: MEASURED from the published cells'
    own elapsed_seconds -- 1.262 GPU*h (rwkv7) + 1.437 GPU*h (global-dit) = 2.699 GPU*h, two lanes
    of a 4-GPU spec, so wall time is the larger of the two. Expected in both logs:
    `[load] resolved stop ids: [0, 2, 65530]`. Writes ONLY to longctx_eosfix/.
    """
    stage = REPO / "results/benchmarks/longctx_eosfix"
    common = (f"--suite longbench --datasets {LONGB_SETS} --data_dir {LONGB} "
              f"--longbench_config_dir /inspire/hdd/global_user/zhangjiaquan-253108540222"
              f"/longbench_local/pinned_spec "
              f"--max_samples 50 --max_input_tokens 32768 --max_new_tokens 512 --seed 42")
    ckpt = (REPO / "outputs_dynamic_basis"
            "/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000")
    body = PREAMBLE + f"mkdir -p {stage}\n"
    body += "echo '=== EOS-fix rebuild: rwkv7 baseline + global-dit (our arm) ==='\n"
    body += (
        "( export CUDA_VISIBLE_DEVICES=0\n"
        f"  {BENCH} {common} --arm raw --decode loop "
        f"--model_dir /inspire/hdd/global_user/zhangjiaquan-253108540222"
        f"/models/RWKV7-Goose-World3-2.9B-HF "
        f"--output {stage}/rwkv7-world3__longbench.json "
        f"> {LOGS}/eosfix-rwkv7-world3-longbench.log 2>&1; echo \"rwkv7 exit=$?\"\n"
        ") &\n"
        "( export CUDA_VISIBLE_DEVICES=1\n"
        f"  {BENCH} {common} --arm global --expected_step 50000 --ckpt_dir {ckpt} "
        f"--output {stage}/global-dit-50k__longbench.json "
        f"> {LOGS}/eosfix-global-dit-50k-longbench.log 2>&1; echo \"global exit=$?\"\n"
        ") &\n"
        "wait\necho '=== EOS-FIX REBUILD COMPLETE ==='\n"
        f"grep -h 'resolved stop ids' {LOGS}/eosfix-*.log || echo 'MISSING stop-ids banner'\n"
        "exit 0\n"
    )
    spec, shm = SPEC4
    return {"workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
            "name": "eosfix-longbench-rwkv7-globaldit",
            "description": (
                "Rebuild the two LongBench cells hit by a stop-token resolution defect in MY OWN "
                "published cells. greedy_loop and both relay runtimes were passed only "
                "tokenizer.eos_token_id; for RWKV7-Goose-World3 that is 65530, an id the model "
                "never emits, while the real end-of-text is 0 (generation_config) and config.json "
                "says 2. So the stop check never fired and all 450 rows ran to full budget. "
                "Measured: id 0 sits inside 141/450 rwkv7 and 79/450 global-dit generations; "
                "rescoring the truncated prefix moves macro 0.0955->0.1197 (baseline) and "
                "0.0955->0.1198 (our own arm). Both cells are in one job deliberately: the defect "
                "hit competitor and our arm by the same amount, so fixing only one would be a "
                "biased half-fix. champ-dynamic (0 rows affected) and gla (eos resolves correctly "
                "to id 2) need no rebuild, and RULER/bAbILong/InfiniteBench are provably immune "
                "(containment metrics; verified 0/1800, 0/400, 0/250 rows change). Protocol copied "
                "from each published cell's config: 50/32768/512, seed 42. Cost basis measured "
                "from those cells' elapsed_seconds: 1.262 + 1.437 GPU*h on two lanes. Writes ONLY "
                "to longctx_eosfix/; promotion to canonical is a separate gated step."),
            "framework": "pytorch", "command": body,
            "max_running_time_ms": "21600000", "task_priority": 4,
            "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                                  "instance_count": 1, "spec_id": spec, "shm_gi": shm}]}


JOBS = {"fingerprint": fingerprint_body, "ruler": ruler_body, "cheap": cheap_body,
        "rwkv7babilong": rwkv7_babilong_body, "rwkv7rest": rwkv7_rest_body,
        "rwkv7memeval": rwkv7_memeval_body, "glalongbench": gla_longbench_body,
        "eosfix": eosfix_longbench_body}


def submit(key: str, dry: bool):
    EVID.mkdir(parents=True, exist_ok=True)
    b = JOBS[key]()
    fn = EVID / f"dfix_{key}.json"
    fn.write_text(json.dumps(b, indent=2, ensure_ascii=False) + "\n")
    args = ["qz", "train", "CreateJob", "--data", f"@{fn}"] + (["--dry-run"] if dry else [])
    r = subprocess.run(args, capture_output=True, text=True, timeout=180)
    if dry:
        ok = "DRY RUN" in r.stdout
        return ("DRYRUN-OK" if ok else None), (None if ok else f"{r.stdout[-400:]}{r.stderr[-400:]}")
    try:
        return json.loads(r.stdout)["Result"]["job_id"], None
    except Exception:
        return None, f"{r.stdout[:400]}{r.stderr[:400]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    keys = [k.strip() for k in a.only.split(",") if k.strip()] or list(JOBS)

    if a.plan:
        for k in keys:
            print(f"\n===== {k} =====\n{JOBS[k]()['command']}")
        return
    if a.status:
        for f in sorted(STAGE.glob("*.json")) + [FZ / "gla-fingerprint-bos.json"]:
            if not f.exists():
                print(f"  MISSING {f.name}")
                continue
            try:
                d = json.loads(f.read_text())
            except Exception as e:
                print(f"  UNPARSEABLE {f.name} ({type(e).__name__}: partial write?)")
                continue
            print(f"  {f.name:34s} decode_path={d.get('decode_path')} "
                  f"trust={d.get('decode_is_trustworthy')} "
                  f"agg={json.dumps(d.get('aggregates') or d.get('scores_pct'))[:150]}")
        return
    if not a.submit:
        print("nothing to do; pass --plan, --submit or --status")
        return
    for k in keys:
        jid, err = submit(k, a.dry_run)
        print(f"  {'OK  ' if jid else 'FAIL'} {k:12s} {jid or err}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()

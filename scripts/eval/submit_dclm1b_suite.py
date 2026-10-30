#!/usr/bin/env python3
"""Submit the DCLM-1B (TRI-ML/DCLM-1B) benchmark suite to QZ as several independent shards.

Sharded on purpose. The brief records a 14 h job that died unexplained on this cluster, so
each shard is a separate job with its own result file and its own log on disk; one shard
dying never costs the others. `GetJobLog` is broken here, so every step redirects to disk.

Shard order is priority order, not file order, and for THIS family the ordering is not a
nicety. The tokenizer-only pre-flight audit (scripts/eval/audit_dclm_window.py) measured all
23 briefed long-context cells against DCLM's real 2048-token window and **0 of 23 are
measurable** -- every row of every cell truncates, seen_fraction 0.0090 to 0.6881. So shard A
(short-context capability + PPL, both fully in-window) carries essentially all of this
family's reportable content, shard B adds the in-window bAbILong lengths, and shards C/D/E
exist to PRODUCE THE TRUNCATION AUDIT -- the per-row input_tokens /
prompt_tokens_untruncated / seen_fraction that prove unmeasurability rather than asserting
it -- not to harvest scores.

ARCHITECTURE NOTE: `model_type: openlm` is not a transformers architecture. It is loaded via
mlfoundations' `open_lm`, whose inference subtree lives in $ROOT/shared_pylibs/open_lm, plus a
deliberate `xformers` stub (no PyPI xformers wheel is ABI-compatible with this image's
torch 2.8.0a0+...nv25.06; DCLM's config selects torch_attn/swiglu_torch so no xformers kernel
is ever reached). Both are already installed on the shared filesystem from the login node --
a job that tried to pip-install at runtime would hang forever with no log. The PYTHONPATH
export in the preamble is therefore LOAD-BEARING: without it nothing about this model loads.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222"
REPO = f"{ROOT}/research/laces-reasoning"
EVID = Path(f"{REPO}/.omo/evidence/dclm-1b")
LEDGER = Path(f"{REPO}/logs/qz/OPENTRAIN_BASELINES_LEDGER.md")

WS = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
SPEC_1GPU, SHM_1GPU = "79fe954a-be92-4772-ac0b-94ad8a79b7bb", 100

REV = "f43c272dab1b06bc5fdd6bbb5819f8a30ef2833b"
MODEL = f"{ROOT}/hf_cache/hub/models--TRI-ML--DCLM-1B/snapshots/{REV}"
BENCH = "scripts/eval/run_open_context_bench.py"
TASKDATA = f"{ROOT}/research/DiffRwkv/baseline/Cola-DLM/generate_task_data"
LONGCTX = f"{REPO}/results/benchmarks/longctx"
INWIN = f"{REPO}/results/benchmarks/longctx_inwindow"
LOGD = f"{REPO}/logs/qz"
GM_MAIN = f"{REPO}/results/benchmarks/baseline-dclm-1b"
GM_FIX2 = f"{REPO}/results/benchmarks/fix2choices/baseline-dclm-1b"

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
# killing the job with no log at all. For this family it is doubly load-bearing -- it is the
# directory `open_lm` and the xformers stub live in.
PREAMBLE = f"""set -uo pipefail
fail=0
mkdir -p {LOGD} {LONGCTX} {INWIN} {REPO}/results/benchmarks/ppl16 \\
         {GM_MAIN} {GM_FIX2}
cd {REPO}
export HF_HOME={ROOT}/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH={ROOT}/shared_pylibs${{PYTHONPATH:+:$PYTHONPATH}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# Fail fast and loudly if the custom architecture cannot be loaded at all, BEFORE spending the
# shard's GPU hours on steps that would each die the same way with a per-step log nobody
# diffs. The parameter count is not a guess: 1,439,893,504 is derived from config arithmetic
# (vocab*dim*2 non-tied + 24*(3*dim^2 + dim^2 + 3*hidden*dim + 2*dim LN + 2*dim qk_norm) + dim,
# hidden=5632) and it equals index total_size/4 (1,439,895,040 fp32 elems) minus the 24*64
# rotary inv_freq buffer elements exactly. So a mismatch here means the weights are not the
# ones audited, and the shard stops instead of publishing numbers from a different model.
#
# The three asserts after it guard the three transformers-5.3.0-vs-open_lm API-drift failures
# actually hit on 2026-09-21 while making this family loadable, all of which were fixed in
# $ROOT/shared_pylibs/open_lm/hf/modeling_openlm.py (original kept as .orig beside it):
#   1. `all_tied_weights_keys` -- only ever set by PreTrainedModel.post_init(), which this
#      barebones wrapper never calls. All 219 tensors loaded, then from_pretrained died in
#      post-load bookkeeping (mark_tied_weights_as_initialized, modeling_utils.py:4598).
#   2. `tie_weights()` -- upstream signature takes no arguments; 5.3.0 calls it as
#      tie_weights(missing_keys=..., recompute_mapping=False) -> TypeError, again AFTER all
#      weights had loaded correctly.
#   3. `generate` -- transformers removed GenerationMixin from PreTrainedModel (announced 4.50,
#      done by 5.x), so `model.generate` raised AttributeError even though can_generate() is
#      True. Every short-context task and the long-context harness's default decode path call
#      generate, so without this the whole suite is dead. Fixed by inheriting GenerationMixin
#      AFTER PreTrainedModel (the MRO order transformers' own warning demands) and by
#      overriding `_supports_default_dynamic_cache` -> False so generate stops pre-allocating a
#      DynamicCache that open_lm's list-based `prepare_inputs_for_generation` would subscript.
# A job running against an unpatched shared_pylibs is the one way these come back, and that is
# precisely what the preflight is for: it re-checks the symptom, not the patch.
python3 -c "
import open_lm.hf, torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('{MODEL}', trust_remote_code=True,
                                         torch_dtype=torch.bfloat16, local_files_only=True)
n = sum(p.numel() for p in m.parameters())
print('[preflight] openlm loaded:', type(m).__name__, 'params', n)
assert n == 1439893504, f'unexpected parameter count {{n}} != 1439893504'
assert hasattr(m, 'generate'), 'open_lm patch missing: no .generate (GenerationMixin not inherited)'
assert m._supports_default_dynamic_cache() is False, 'open_lm patch missing: generate would inject a DynamicCache'
assert m.all_tied_weights_keys == {{}}, f'unexpected tied weights {{m.all_tied_weights_keys}} (config says weight_tying: false)'
import torch as _t
ids = _t.tensor([[510, 5347, 273, 6181, 310]])
out = m.generate(input_ids=ids, attention_mask=_t.ones_like(ids), max_new_tokens=4, do_sample=False)
print('[preflight] generate ok, new ids', out[0, ids.shape[1]:].tolist())
" > {LOGD}/dclm1b-preflight.log 2>&1 || {{ echo "PREFLIGHT FAILED -- see {LOGD}/dclm1b-preflight.log"; exit 1; }}
"""


def step(cmd: str, log: str) -> str:
    return (f'echo "=== {log} === $(date -u)"\n'
            f"CUDA_VISIBLE_DEVICES=0 {cmd} > {LOGD}/{log}.log 2>&1 || fail=1\n")


def genmatch(outdir: str, tasks: str, log: str) -> str:
    # NOTE on --trust_remote_code: it is passed for documentation/parity only and is in fact a
    # NO-OP in both short-context scripts -- their `_load_hf_model(model_dir, device,
    # trust_remote_code=True)` never references the parameter in its body and hardcodes
    # `trust_remote_code=True` in the from_pretrained kwargs. What actually makes DCLM loadable
    # is the `import open_lm.hf` registration block added to each script's main(); verified by
    # reproducing the bare ValueError without it.
    return step(
        f"python3 scripts/eval/eval_generic_gen_match.py "
        f"--model_dir {MODEL} --task_data_dir {TASKDATA} --output_dir {outdir} "
        f"--tasks {tasks} --max_samples 200 --max_new_tokens 32 --temperature 0.0 "
        f"--trust_remote_code", log)


def bench(suite: str, datasets: str, data_dir: str, out: str, log: str,
          max_samples: int, mit: int, mnt: int, extra: str = "") -> str:
    # run_open_context_bench.py already passes trust_remote_code=True unconditionally and
    # resolves `openlm` through ensure_model_type_resolvable, so no extra flag is needed.
    return step(
        f"python3 -u {BENCH} --suite {suite} --datasets {datasets} --data_dir {data_dir} "
        f"{extra}--model_dir {MODEL} --arm raw --output {out} "
        f"--max_samples {max_samples} --max_input_tokens {mit} --max_new_tokens {mnt} "
        f"--seed 42 --device cuda", log)


# --- shards, in priority order ----------------------------------------------------------
# A: in-window -> effectively this family's whole contribution. B: in-window length sweep.
# C/D/E: briefed grid, pre-measured out-of-window; run for the truncation audit only.
SHARDS: dict[str, dict] = {
    "A-short-ppl": dict(
        desc=("dclm-1b short-context capability (14 columns) + PPL 7 corpora; all in-window "
              "(2048). This is the shard that carries the family's reportable numbers."),
        hours=6,
        steps=[
            genmatch(GM_MAIN, "mmlu,obqa,race,squad,story_cloze,hellaswag,lambada,siqa",
                     "dclm1b-genmatch-main"),
            genmatch(GM_FIX2, "arc_easy,arc_challenge,winogrande,sciq,commonsenseqa,piqa",
                     "dclm1b-genmatch-fix2"),
            # DTYPE, and it is NOT uniform across this shard -- measured, not assumed:
            #   * eval_generic_gen_match.py's dtype probe WORKS (that file does import `os` at
            #     module level), so it honours config torch_dtype and loads DCLM in FLOAT32.
            #   * eval_lm_ppl.py's probe is INERT: it calls os.path.join with no module-level
            #     `import os`, the NameError is swallowed by a bare `except Exception: pass`,
            #     and every --impl hf row silently falls back to BFLOAT16.
            #   * run_open_context_bench.py hardcodes bfloat16 for --model_dir.
            # So DCLM's capability columns are fp32 while its PPL row is bf16. Both are left
            # UNMODIFIED on purpose: all 8 existing ppl16 HF rows were produced under the inert
            # probe, so "fixing" it would make DCLM the only fp32 PPL row in a bf16 column --
            # a comparability break dressed up as a fix, and none of the result files record
            # dtype so it would be invisible. Reported in the ledger instead; the size of the
            # bf16-vs-fp32 gap is measured directly in check_dclm_load.py.
            # EXPECT wikitext2 == wikitext103 EXACTLY in the output, and do not "fix" it.
            # CORPORA maps wikitext2 -> preprocessed_data/wikitext2_full, and its first
            # 10631 chunks are byte-identical to wikitext103's (verified here by comparing
            # chunk_000000/1/2 of each). With --max_files 100 both rows therefore read the
            # same 100 chunks. Every existing ppl16 row shows the same tie (e.g. qwen2.5-3b
            # 11.13 / 11.13), so DCLM matching itself is a property of the shared protocol,
            # not a DCLM bug and not a copy-paste error in this script.
            # 256-token chunks are far inside 2048, so PPL is fully in-window.
            step(f"python3 scripts/eval/eval_lm_ppl.py --impl hf --model_dir {MODEL} "
                 f"--mode raw --trust_remote_code "
                 f"--corpora wikitext2,wikitext103,pg19,lm1b,ptb,wmt14,xsum "
                 f"--max_files 100 --output {REPO}/results/benchmarks/ppl16/dclm-1b.json",
                 "ppl16-dclm-1b"),
            # Score in-job so the numbers exist even if I lose the shell. TWO calls, because
            # score_gen_match.py's main() prints only the 8 non-option tasks its hardcoded
            # `tasks` list names -- the 6 option columns in fix2choices/ would silently not
            # appear at all. The second call drives score_file() directly to cover them.
            # SCORE_LETTER_FIRST is left UNSET in both: the default rule set is what all nine
            # published baselines used, and enabling it would make DCLM incomparable with them
            # (the OLMo-2 line made the same call for the same reason). Scoring is read-only
            # over the generations, so it can be re-run later under either rule set.
            step(f"python3 -u scripts/eval/score_gen_match.py {GM_MAIN}",
                 "dclm1b-score-main"),
            step("python3 -u -c \""
                 "import sys; sys.path.insert(0, 'scripts/eval'); "
                 "from score_gen_match import score_file, collect; "
                 f"f = collect('{GM_FIX2}'); "
                 "print('task            acc     n'); "
                 "[print(f'{t:14s} {score_file(f[t])[1]:6.1f} {score_file(f[t])[3]:6d}') "
                 "for t in ['arc_easy','arc_challenge','winogrande','sciq','commonsenseqa','piqa'] "
                 "if t in f]\"",
                 "dclm1b-score-fix2"),
        ]),
    "B-babilong-inwindow": dict(
        desc="dclm-1b bAbILong qa1/qa2 at 0k/1k/2k -- the lengths that fit its 2048 window",
        hours=4,
        steps=[bench("babilong", "qa1/0k,qa1/1k,qa1/2k,qa2/0k,qa2/1k,qa2/2k", BABI,
                     f"{INWIN}/dclm-1b__babilong_inwindow.json",
                     "dclm1b-babilong-inwindow", 200, 16384, 32)]),
    "C-babilong-briefed": dict(
        desc=("dclm-1b bAbILong briefed grid 4k/8k/16k -- pre-measured OUT OF WINDOW "
              "(seen_fraction 0.131-0.558), run for the truncation audit"),
        hours=4,
        steps=[bench("babilong", "qa1/4k,qa1/8k,qa1/16k,qa2/4k", BABI,
                     f"{LONGCTX}/dclm-1b__babilong.json",
                     "dclm1b-babilong", 200, 16384, 32)]),
    "D-ruler": dict(
        desc=("dclm-1b RULER 4096+16384 -- pre-measured OUT OF WINDOW "
              "(seen_fraction 0.113-0.688; min shipped length 4096)"),
        hours=6,
        steps=[bench("ruler",
                     "4096/niah_single_1,4096/niah_multikey_1,4096/vt,4096/cwe,4096/fwe,"
                     "4096/qa_1,16384/niah_single_1,16384/vt,16384/cwe", RULER,
                     f"{LONGCTX}/dclm-1b__ruler.json", "dclm1b-ruler",
                     200, 16384, 128)]),
    "E-lb-inf-mem": dict(
        desc=("dclm-1b LongBench + InfiniteBench + MemEval -- pre-measured OUT OF WINDOW "
              "(seen_fraction 0.009-0.271)"),
        hours=6,
        steps=[
            bench("longbench", "narrativeqa,qasper,multifieldqa_en,hotpotqa", LB,
                  f"{LONGCTX}/dclm-1b__longbench.json", "dclm1b-longbench",
                  50, 32768, 512, extra=f"--longbench_config_dir {LBSPEC} "),
            bench("infinitebench",
                  "longbook_qa_eng,longbook_choice_eng,kv_retrieval,code_debug,math_find", INF,
                  f"{LONGCTX}/dclm-1b__infinitebench.json", "dclm1b-infinitebench",
                  50, 32768, 128),
            # MemEval's 50-row slice has a 0.220 random floor and known scorer defects; run
            # for completeness, reported as uninformative at or below that floor.
            bench("memeval", "longmemeval_oracle", MEM,
                  f"{LONGCTX}/dclm-1b__memeval.json", "dclm1b-memeval",
                  50, 131072, 128),
        ]),
}


def body(key: str) -> dict:
    sh = SHARDS[key]
    return {
        "workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
        "name": f"dclm1b-{key}"[:60],
        "description": sh["desc"],
        "framework": "pytorch",
        "command": PREAMBLE + "".join(sh["steps"]) + f'echo "DONE fail=$fail $(date -u)"\nexit $fail\n',
        "max_running_time_ms": str(sh["hours"] * 3600 * 1000),
        "task_priority": 4,
        "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                              "instance_count": 1, "spec_id": SPEC_1GPU, "shm_gi": SHM_1GPU}],
    }


def append_ledger(rows: list[str]) -> None:
    """Append job rows under the '### DCLM' table only.

    Three other agents write their own sections of this file concurrently, so this re-reads
    the file and does a targeted insert at the end of the DCLM table rather than rewriting
    the whole document. Anchored on the DCLM heading, because the `| family | suite | ...`
    table marker appears once per family section and `str.index` would otherwise hit
    Pythia's.
    """
    text = LEDGER.read_text()
    sec = text.index("### DCLM")
    marker = "| family | suite | job_id | submitted UTC | log path |\n|---|---|---|---|---|\n"
    idx = text.index(marker, sec) + len(marker)
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
        rows.append(f"| DCLM | {k} | `{jid}` | {ts} | `{LOGD}/dclm1b-*.log` |\n")
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

#!/usr/bin/env python3
"""Submit the 8 missing efficiency cells for the paper's tab:resources.

The 09-20 batch landed 8 of 16 rows. This adds the other 8, through the SAME
scripts/eval/eval_efficiency.py, so the numbers stay comparable with the ones
already published. Why each of the 8 was missing (read from logs/qz/eff-*.log):

  gla-1.3b        AutoModelForCausalLM could not resolve model_type `gla` -- the
                  fla-hub snapshot ships no remote-code file, so the class only
                  exists after `fla.models` registers it.
  rwkv6-world     ) both died with "contains custom code which must be executed to
  rwkv7-world3    ) correctly load the model. Please pass trust_remote_code=True" --
                  the driver passed a store_true flag that defaults to False.
  mdlm-owt        )
  sedd-medium     ) never attempted: no efficiency branch existed for models with
  cola-dlm        ) no standard decode loop.
  elf-l / elf-m   )

One job per model, one output FILE per model (never a shared directory), so two
jobs can never overwrite each other's row.

  python3 submit_efficiency_missing8.py --plan      # show the commands, submit nothing
  python3 submit_efficiency_missing8.py --submit
  python3 submit_efficiency_missing8.py --status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
MODELS = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/models")
HF_CACHE = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache")
SHARED_PYLIBS = "/inspire/hdd/global_user/zhangjiaquan-253108540222/shared_pylibs"
OUT = REPO / "results/benchmarks/efficiency"
LOGS = REPO / "logs/qz"
EVIDENCE = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv"
                "/.omo/evidence/qz-training-skill")

WS = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
LCG = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
IMAGE = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
SPEC1 = ("79fe954a-be92-4772-ac0b-94ad8a79b7bb", 100)      # 1 x H100
GLA = (HF_CACHE / "hub/models--fla-hub--gla-1.3B-100B/snapshots"
       "/46b15820a4df269e99aed9d709e017677c15d24b")

# name -> eval_efficiency.py flags. `impl` picks the measurement protocol:
#   hf    = autoregressive, has a real per-token decode step
#   mdlm/sedd/cola/elf = non-autoregressive, decode_ms_tok is UNDEFINED and the
#           script records null + a reason string plus a fixed-length generation time.
CELLS = {
    "gla-1.3b":     f"--impl hf --model_dir {GLA}",
    "rwkv7-world3": f"--impl hf --model_dir {MODELS / 'RWKV7-Goose-World3-2.9B-HF'}",
    "rwkv6-world":  f"--impl hf --model_dir {MODELS / 'rwkv6-world-3b'}",
    "mdlm-owt":     f"--impl mdlm --model_dir {MODELS / 'MDLM-owt'} --diff_steps 128",
    # lowercase sedd-medium: the uppercase copy's pytorch_model.bin was a partial
    # download (353 MB of 1698 MB) and is now only a symlink back to this one.
    "sedd-medium":  f"--impl sedd --model_dir {MODELS / 'sedd-medium'} --diff_steps 128",
    # lowercase cola-dlm: models/Cola-DLM is TRUNCATED (DiT short 460 MB, VAE 24.8 MB)
    # and dies with SafetensorError: MetadataIncompleteBuffer.
    "cola-dlm":     f"--impl cola --model_dir {MODELS / 'cola-dlm'} --diff_steps 16",
    "elf-l":        f"--impl elf --model_dir {MODELS / 'ELF-L-owt-torch'} --diff_steps 64",
    "elf-m":        f"--impl elf --model_dir {MODELS / 'ELF-M-owt-torch'} --diff_steps 64",
    # --- re-measurement of the two relay rows with the WARMED-UP time_decode ---------
    # The 09-20 rows are cold single-shot timings. Proof, measured 2026-09-21 on the bare
    # frozen backbone these two share: warm 73.71 ms/tok vs cold 384.43 ms/tok. The whole
    # published 373-381 cluster is the cold figure, and global-dit-50k's 199.66 is a
    # partially-warmed point on the same triton-JIT curve (rwkv7-world3's own cold bs8 came
    # out at 199.10). New FILENAMES, so the published rows stay on disk as evidence.
    "champ-dynamic-30k-warm": (
        "--impl relay --ckpt_dir " + str(REPO / "outputs_dynamic_basis"
        / "laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03" / "step_00030000")),
    "global-dit-50k-warm": (
        "--impl relay --ckpt_dir " + str(REPO / "outputs_dynamic_basis"
        / "laces-2.9B-dynlowrank-r32-global-joint-b16-50k" / "step_00050000")),
}


def body(name: str) -> dict:
    flags = CELLS[name]
    log = LOGS / f"eff2-{name}.log"
    out = OUT / f"{name}.json"
    spec, shm = SPEC1
    cmd = (
        "set -uo pipefail\n"
        f"mkdir -p {LOGS} {OUT}\n"
        f"cd {REPO}\n"
        f"export HF_HOME={HF_CACHE}\n"
        # ${PYTHONPATH:+:$PYTHONPATH}, never a bare $PYTHONPATH: PYTHONPATH is UNSET in
        # this image and under `set -u` a bare reference aborts the shell before any log
        # redirect exists -- that silently killed dozens of jobs with no log at all.
        # shared_pylibs carries rotary-embedding-torch, a hard import for CoLA's DiT.
        f"export PYTHONPATH={SHARED_PYLIBS}${{PYTHONPATH:+:$PYTHONPATH}}\n"
        "export HF_HUB_OFFLINE=1\n"
        "export TRANSFORMERS_OFFLINE=1\n"
        f"echo \"=== efficiency {name} ===\"\n"
        "CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"python scripts/eval/eval_efficiency.py {flags} "
        f"--len 512 --gen_len 64 --output {out} > {log} 2>&1 || fail=1\n"
        f"tail -40 {log}\n"
        "exit ${fail:-0}\n"
    )
    return {"workspace_id": WS, "project_id": PROJECT, "logic_compute_group_id": LCG,
            "name": f"eff2-{name}"[:60],
            "description": f"efficiency/memory measurement: {name} (paper tab:resources)",
            "framework": "pytorch", "command": cmd,
            "max_running_time_ms": "10800000", "task_priority": 4,
            "framework_config": [{"image": IMAGE, "image_type": "SOURCE_PRIVATE",
                                  "instance_count": 1, "spec_id": spec, "shm_gi": shm}]}


def submit(name: str, dry: bool):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    b = body(name)
    fn = EVIDENCE / f"eff2_{name}.json"
    fn.write_text(json.dumps(b, indent=2, ensure_ascii=False) + "\n")
    if dry:
        r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{fn}", "--dry-run"],
                           capture_output=True, text=True, timeout=120)
        if "DRY RUN" not in r.stdout:
            return None, f"dry-run failed: {r.stdout[-300:]}{r.stderr[-300:]}"
        return "DRYRUN-OK", None
    r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{fn}"],
                       capture_output=True, text=True, timeout=180)
    try:
        return json.loads(r.stdout)["Result"]["job_id"], None
    except Exception:
        return None, f"submit failed: {r.stdout[:300]}{r.stderr[:300]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="comma list of cell names")
    a = ap.parse_args()

    names = [n.strip() for n in a.only.split(",") if n.strip()] or list(CELLS)

    if a.status:
        # Count NON-EMPTY files: a 0-byte file is a failed run, not a finished one.
        print(f"{'model':16s} {'json':>8s} {'bytes':>8s}  decode_ms_tok")
        for n in CELLS:
            f = OUT / f"{n}.json"
            sz = f.stat().st_size if f.exists() else 0
            d = None
            if sz > 0:
                try:
                    d = json.loads(f.read_text()).get("decode_ms_tok_bs1", "MISSING")
                except Exception:
                    d = "UNPARSEABLE"
            print(f"{n:16s} {'yes' if sz else 'NO':>8s} {sz:>8d}  {d}")
        return

    if a.plan:
        for n in names:
            print(f"\n===== {n} =====\n{body(n)['command']}")
        print(f"\n{len(names)} cells, 1 GPU each, 1 job each.")
        return

    if not a.submit:
        print("nothing to do; pass --plan, --submit or --status")
        return

    for n in names:
        jid, err = submit(n, a.dry_run)
        print(f"  {'OK  ' if jid else 'FAIL'} {n:16s} {jid or err}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()

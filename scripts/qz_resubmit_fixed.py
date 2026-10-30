#!/usr/bin/env python3
"""Submit bl-* baseline-eval jobs with a repaired command preamble.

WHY THIS EXISTS
---------------
``run_baseline_evals.py`` is frozen (do-not-edit).  Its generated preamble is::

    set -uo pipefail
    ...
    export PYTHONPATH=<shared_pylibs>:$PYTHONPATH

In the compute container (image relay2:v2) ``PYTHONPATH`` is *unset*.  Under ``set -u``
the expansion ``$PYTHONPATH`` is an unbound-variable error, and bash aborts the whole
non-interactive shell on that line -- **before** ``mkdir``, before ``cd``, and before any
``> logs/qz/*.log`` redirect is ever opened.

Effect: every job whose command carries that line fails in ~60s having written *nothing*
at all -- no log files, no traceback.  That is why `logs/qz/*` mtimes look stale/absent
for those runs and why "the log" appears to be from a previous run.

Proven, not inferred -- probe job ``job-d4f1cce9-e1e8-4cc4-bd63-cd0ace810fd3``
(2026-09-20 17:12 UTC) ran::

    touch  .../logs/qz/PROBE_A_before_pythonpath.txt   -> CREATED
    export PYTHONPATH=<shared_pylibs>:$PYTHONPATH      -> shell dies here
    touch  .../logs/qz/PROBE_B_after_pythonpath.txt    -> NEVER CREATED
    echo REACHED_END > .../PROBE_result.txt            -> NEVER CREATED

job_failed after 63s.  Cross-check on the fleet: of the 6 jobs submitted at/after
16:52:34, all 6 carry the line and all 6 failed; every earlier job lacks the line and
the ones with fixed code succeeded.

WHAT THIS DOES
--------------
Reuses the frozen driver's *own* planning (``cells``/``cell_done``/``build_job``) so cell
lists, per-model output dirs, commands, images and quota stay bit-identical, then rewrites
only that one line to the ``set -u``-safe form::

    export PYTHONPATH=<shared_pylibs>${PYTHONPATH:+:$PYTHONPATH}

and submits.  Nothing in run_baseline_evals.py is modified.

Usage
  python3 scripts/qz_resubmit_fixed.py --models Mamba-2,MDLM-OWT,SEDD-medium
  python3 scripts/qz_resubmit_fixed.py --models Gemma-2-2B --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
sys.path.insert(0, str(REPO))

import run_baseline_evals as R  # noqa: E402  (frozen driver, imported read-only)

EVIDENCE = REPO / "logs/qz/resubmit"
BROKEN = "shared_pylibs:$PYTHONPATH"
FIXED = "shared_pylibs${PYTHONPATH:+:$PYTHONPATH}"


def repair(cmd: str) -> tuple[str, bool]:
    if BROKEN not in cmd:
        return cmd, False
    return cmd.replace(BROKEN, FIXED), True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list of baseline names")
    ap.add_argument("--cells-per-job", type=int, default=14)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--gpu", type=int, default=1, choices=sorted(R.SPEC_BY_GPU))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    want = [m.strip() for m in a.models.split(",") if m.strip()]
    unknown = [m for m in want if m not in R.BASELINES and m not in R.RELAY_MODELS]
    if unknown:
        print(f"unknown model name(s): {unknown}", file=sys.stderr)
        return 2

    all_cells = R.cells()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    rc = 0
    for model in want:
        todo = [c for c in all_cells if c["model"] == model and not R.cell_done(c)]
        print(f"\n=== {model}: {len(todo)} cell(s) still to land "
              f"(of {sum(1 for c in all_cells if c['model'] == model)})")
        if not todo:
            print("    nothing to do (all cells landed) -- a job would only carry PPL")
        batch = todo[: a.cells_per_job] if todo else []
        if not batch:
            continue
        body = R.build_job(batch, a.max_samples, a.gpu)
        body["command"], patched = repair(body["command"])
        if not patched:
            print("    !! preamble did not contain the broken line -- refusing to submit "
                  "blind", file=sys.stderr)
            rc = 1
            continue
        fn = EVIDENCE / f"fixed_{body['name']}.json"
        fn.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
        labels = ",".join(c["label"] for c in batch)
        print(f"    job={body['name']}  cells=[{labels}]")
        print(f"    repaired preamble line -> {FIXED}")
        if a.dry_run:
            print("    (dry-run: not submitting)")
            continue
        r = subprocess.run(["qz", "train", "CreateJob", "--data", f"@{fn}"],
                           capture_output=True, text=True, timeout=180)
        try:
            jid = json.loads(r.stdout)["Result"]["job_id"]
            print(f"    SUBMITTED {jid}")
        except Exception:
            print(f"    SUBMIT FAILED: {r.stdout[:300]}{r.stderr[:300]}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

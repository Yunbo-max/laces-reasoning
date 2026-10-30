#!/usr/bin/env python3
"""D2: pin the 200/200/200 evaluation set for the five-arm latent-use controls.

The harness default ``--task_data_dir baseline/Cola-DLM/generate_task_data`` does
NOT exist inside laces-reasoning, so this script sources prompts from already
produced result jsonls.  That has a second benefit the protocol demands: the
example ids and prompt strings are then bit-identical to the rows that were
already scored, so the five new arms are compared on the same set.

RACE ships in a 100-row and a 200-row flavour on disk; the protocol says 200, so
every count is asserted, not merely reported.  SciQ and CommonsenseQA also ship
in TWO PROMPT VARIANTS at 200 rows: an older one with no option list and a newer
``fix2choices`` one that inlines ``(A) ...``.  Picking the wrong variant silently
changes the task, so the prompt-set sha256 is pinned too.

    python3 scripts/eval/latentuse/fix_examples.py \
        --out scripts/eval/latentuse/examples_fixed.json

Nothing is copied: sources are referenced by absolute path plus a content hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]

EXPECT_ROWS = 200

# Chosen sources.  The rationale for each is recorded in the output record.
#
# sciq / commonsenseqa: results/benchmarks/fix2choices/baseline-global-dit-50k/
#   is the champion checkpoint's OWN run (run_summary.json ckpt_dir ==
#   .../laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000) and it
#   carries the OPTION-INLINED prompt variant, which 40/41 of the 200-row dirs
#   share.  The sibling results/benchmarks/baseline-global-dit-50k/{sciq,cqa}
#   is the SAME checkpoint but the OLD no-options prompt (24 dirs) -- do not use
#   it, or the five arms answer a different question than the reported table.
#
# race: results/benchmarks/baseline-global-dit-50k/race.jsonl, also the champion
#   checkpoint, 200 rows.  RACE has only ONE 200-row prompt variant (32 dirs all
#   agree), so there is no variant ambiguity here, only the 100-vs-200 one.
SOURCES: dict[str, str] = {
    "sciq": "results/benchmarks/fix2choices/baseline-global-dit-50k/sciq.jsonl",
    "commonsenseqa": "results/benchmarks/fix2choices/baseline-global-dit-50k/commonsenseqa.jsonl",
    "race": "results/benchmarks/baseline-global-dit-50k/race.jsonl",
}

# Prompt-set sha256 over "<id>\x00<prompt>" joined in file order, FULL 64 hex
# digits.  Computed 2026-09-23 from the sources above and cross-checked against
# every other 200-row dir under results/benchmarks: the sciq/cqa values are the
# majority variant (40 and 41 dirs respectively), the race value is the only
# 200-row variant (32 dirs).  The runner asserts against these, so a rewritten
# source file surfaces as a mismatch instead of being silently averaged in.
#
# A truncated hash is not a gate: a 16-hex prefix pinned here once matched a
# value whose tail had been invented, and the check still "passed" on the prefix.
# Store all 64 digits or store nothing.
EXPECTED_PROMPT_SET_SHA256 = {
    "sciq": "2532316b0fa26f27e60ac26303ecf018ae8c43e878944ebe11a6b1bd216c1484",
    "commonsenseqa": "e26382954b91739e1fb35c8229807ccbf9b8e380360667ad88c6905270e07841",
    "race": "db5f20ed3d402c66c67a4ab54d25dcaa60f705dd28c679c70df5f7cc174d6549",
}

# WARNING -- id_list_sha256 is NOT a task discriminator.  All three tasks are
# ids 0..199, so their id-list hashes are byte-identical
# (3aff9a6f...f015a25).  A runner that only checks the id-list hash would accept
# race.jsonl in place of sciq.jsonl.  Only prompt_set_sha256 separates the tasks,
# and only it separates the two sciq/cqa prompt VARIANTS at the same 200 ids.
# The runner therefore gates on prompt_set_sha256 and treats the id-list hash as
# an ordering check alone.
ID_LIST_SHA_IS_SHARED_ACROSS_TASKS = True


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_rows(p: Path) -> list[dict[str, Any]]:
    rows = []
    with p.open(encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def prompt_set_sha(rows: list[dict[str, Any]]) -> str:
    return sha256_text("".join(f"{r['id']}\x00{r['prompt']}" for r in rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("examples_fixed.json")))
    ap.add_argument("--rows", type=int, default=EXPECT_ROWS)
    args = ap.parse_args()

    tasks: dict[str, Any] = {}
    failures: list[str] = []

    for task, rel in SOURCES.items():
        src = REPO / rel
        print(f"\n[fix_examples] task={task}\n  source={src}")
        if not src.is_file():
            failures.append(f"{task}: source missing {src}")
            continue
        rows = load_rows(src)
        n = len(rows)
        print(f"  DENOMINATOR rows_in_file={n} (required exactly {args.rows})")
        if n != args.rows:
            failures.append(f"{task}: {n} rows, required {args.rows} ({src})")
            continue

        ids = [r["id"] for r in rows]
        if len(set(ids)) != n:
            failures.append(f"{task}: duplicate ids ({n - len(set(ids))} dupes)")
            continue

        examples = []
        for r in rows:
            prompt = r["prompt"]
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{task} id={r.get('id')}: empty/non-str prompt")
            choices = r.get("choices") or []
            if not isinstance(choices, list):
                raise TypeError(f"{task} id={r.get('id')}: choices is {type(choices)}")
            gold = str(r.get("ground_truth", ""))
            if gold == "":
                raise ValueError(f"{task} id={r.get('id')}: empty ground_truth")
            examples.append({
                "id": r["id"],
                "prompt_sha256": sha256_text(prompt),
                "prompt_len_chars": len(prompt),
                "prompt_len_bytes": len(prompt.encode("utf-8")),
                "gold": gold,
                "choices": choices,
                "n_choices": len(choices),
            })

        set_sha = prompt_set_sha(rows)
        id_list_sha = sha256_text(json.dumps(ids, separators=(",", ":")))
        exp = EXPECTED_PROMPT_SET_SHA256.get(task)
        if exp is None:
            failures.append(f"{task}: no pinned prompt_set_sha256 in this file")
        elif len(exp) != 64:
            failures.append(f"{task}: pinned prompt_set_sha256 has {len(exp)} digits, need 64")
        elif exp != set_sha:
            failures.append(
                f"{task}: prompt set CHANGED\n    pinned   = {exp}\n    computed = {set_sha}"
            )
        else:
            print(f"  PASS prompt_set_sha256 matches the pinned value")
        n_choiceless = sum(1 for e in examples if e["n_choices"] == 0)
        print(f"  prompt_set_sha256 = {set_sha}")
        print(f"  id_list_sha256    = {id_list_sha}")
        print(f"  ids[:5]={ids[:5]} ids[-3:]={ids[-3:]}  choiceless_rows={n_choiceless}")
        if n_choiceless:
            failures.append(f"{task}: {n_choiceless} rows have no choices; "
                            "the answer parser would have nothing to match")

        tasks[task] = {
            "source_path": str(src),
            "source_sha256": sha256_file(src),
            "source_run_summary": str(src.parent / "run_summary.json"),
            "n_examples": n,
            "ids": ids,
            "id_list_sha256": id_list_sha,
            "id_list_sha256_is_task_discriminator": False,
            "prompt_set_sha256": set_sha,
            "examples": examples,
        }

    if failures:
        for f in failures:
            print(f"\nFAIL {f}")
        raise SystemExit(f"{len(failures)} assertion(s) failed; nothing written")

    rec = {
        "generated_by": "scripts/eval/latentuse/fix_examples.py",
        "rows_required_per_task": args.rows,
        "selection_rationale": {
            "sciq": "champion ckpt's own fix2choices run; OPTION-INLINED prompt variant "
                    "(majority: 40 of the 200-row dirs). The sibling non-fix2choices dir "
                    "is the same ckpt but the pre-fix no-options prompt (24 dirs).",
            "commonsenseqa": "same dir and same reason (41 dirs share this variant).",
            "race": "champion ckpt's own run; RACE's single 200-row prompt variant "
                    "(32 dirs agree). The 100-row RACE dirs are excluded by the "
                    "rows_required_per_task assertion.",
        },
        "prompt_regeneration_note":
            "build_prompt() in scripts/eval/run_cola_dlm_tasks_prefix_suffix_cfg.py applied to "
            "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/baseline/"
            "Cola-DLM/generate_task_data/{task}.jsonl[:200] reproduces all 600 prompts "
            "byte-identically (verified 2026-09-23, 0 mismatches). That dir lives in the "
            "DiffRwkv repo, not laces-reasoning, which is why the harness default path fails.",
        "tasks": tasks,
        "macro_denominator": sum(t["n_examples"] for t in tasks.values()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"\n[fix_examples] {len(tasks)} tasks, macro denominator "
          f"{rec['macro_denominator']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

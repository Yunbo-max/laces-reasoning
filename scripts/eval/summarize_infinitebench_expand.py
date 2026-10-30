#!/usr/bin/env python3
"""Aggregate the InfiniteBench 7-task expansion into the deliverable table.

Reads only `results/benchmarks/infbench_expand/*__infinitebench.json` (this line's staging
directory) and prints, per cell: n, input budget, seen_fraction, rows_truncated, and the
score -- every score labelled "approx.".

Three things this does that a score-only reader cannot:

1. **Needle-visibility ceiling.** For passkey/number_string it recomputes, from the source
   jsonl, whether the gold string survived middle-truncation for each evaluated row. A model
   whose input never contained the key cannot answer; its ceiling is arithmetic, not
   capability, and a 0.0 against a 0.27 ceiling means something completely different from a
   0.0 against a 1.0 ceiling.
2. **Degeneracy census.** Empty generations, and generations whose unique-token ratio is tiny
   (the repetition/quote-spam signature that produced several earlier false readings).
3. **Depth curve.** Because the needle files are depth-sorted and we sample stride-10, score
   as a function of needle depth is directly available -- which is the actual scientific
   object here, not the scalar mean.

Read generations, do not trust scores: `--quote N` prints raw predictions so any cell that
looks like a win can be checked by eye.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
STAGE = REPO / "results/benchmarks/infbench_expand"
DATA = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/"
            "datasets--xinrongzhang2022--InfiniteBench/snapshots/"
            "90f0394333616266d9fe85824ceaf505093cbaa5")

NEEDLE_TASKS = ("passkey", "number_string")


def needle_depths(task: str) -> dict[int, float]:
    """Row id -> needle depth as a fraction of context length, from the source jsonl."""
    out: dict[int, float] = {}
    path = DATA / f"{task}.jsonl"
    if not path.exists():
        return out
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            context = str(item.get("context", ""))
            answers = item.get("answer") or []
            gold = str(answers[0] if isinstance(answers, list) and answers else answers)
            position = context.find(gold)
            if position >= 0 and context:
                out[item.get("id")] = position / len(context)
    return out


def needle_survives(depth: float, untruncated: int, seen: int) -> bool:
    """Did middle-truncation (head + tail) keep the needle in the model's input?

    Mirrors truncate_ids: keep ids[:budget//2] and ids[-(budget-budget//2):].
    """
    if seen >= untruncated:
        return True
    head = seen // 2
    tail = seen - head
    return depth < (head / untruncated) or depth > 1.0 - (tail / untruncated)


def degeneracy(prediction: str) -> str:
    """One-word verdict on a generation's shape, for the census.

    The `counting` case was added after the unique-token-ratio rule scored gla-1.3b's
    `'1, 2, 3, 4, 5, ...'` as "ok" on all 59 number_string rows: a monotone counter has ~100%
    distinct tokens, so the ratio rule cannot see it, yet it made the `digits` column read
    59/59 as if the model had attempted a 10-digit answer. Without this the census understates
    degeneracy exactly where it would flatter a baseline.
    """
    if not prediction.strip():
        return "empty"
    tokens = prediction.split()
    integers = [int(t) for t in re.findall(r"\d+", prediction)]
    if len(integers) >= 8 and all(b - a == 1 for a, b in zip(integers, integers[1:])):
        return "counting"
    if len(tokens) >= 8 and len(set(tokens)) / len(tokens) < 0.20:
        return "repetitive"
    return "ok"


def lenient_credit(rows: list[dict]) -> int:
    """How many rows would score if the gold merely APPEARED anywhere in the generation?

    This is the old containment rule -- strictly more generous than the official
    `first_int_match`. Reporting it alongside the strict score answers the only question that
    matters about a 0.0: is the zero an artefact of the strict metric, or is the answer simply
    not in the generation at all? If lenient credit is also 0, no scoring rule would have
    produced a nonzero number and the zero is a property of the model.
    """
    total = 0
    for row in rows:
        prediction = row.get("prediction") or ""
        for answer in (row.get("answers") or []):
            if str(answer) and str(answer) in prediction:
                total += 1
                break
    return total


def emits_digits(rows: list[dict]) -> int:
    """Rows whose generation contains at least one digit -- did the model even attempt a number?"""
    return sum(1 for row in rows if re.search(r"[0-9]", row.get("prediction") or ""))


def emits_plausible_run(rows: list[dict]) -> int:
    """Rows containing a digit run at least as long as the gold key.

    `emits_digits` is too weak to mean "attempted the task": gla-1.3b emits digits on 59/59
    number_string rows, but only as `'1, 2, 3, ...'` whose longest run is 2 characters against
    a 10-digit gold. A run of the gold's own length is the weakest signal that the model
    produced something that COULD have been the key, so this is the column to read when asking
    whether a 0.0 is a retrieval failure or a formatting failure.
    """
    total = 0
    for row in rows:
        prediction = row.get("prediction") or ""
        need = min((len(str(a).strip()) for a in (row.get("answers") or []) if str(a).strip()),
                   default=1)
        runs = [run for run in re.split(r"[^0-9]", prediction) if run]
        if any(len(run) >= need for run in runs):
            total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quote", type=int, default=0,
                        help="print this many raw generations per cell")
    parser.add_argument("--depth_curve", action="store_true",
                        help="print score vs needle depth for the needle tasks")
    args = parser.parse_args()

    files = sorted(STAGE.glob("*__infinitebench.json"))
    if not files:
        print(f"no result files yet in {STAGE}")
        return

    depths = {task: needle_depths(task) for task in NEEDLE_TASKS}
    table: dict[tuple[str, str], dict] = {}

    for path in files:
        blob = json.loads(path.read_text())
        parts = path.name.split("__")
        model = parts[0]
        # The length-ladder control writes `<model>__ladder4k__infinitebench.json`. Those cells
        # are the SAME task at a shorter context, so they must not be averaged into the 125k row;
        # they get their own model label and are printed as a separate ladder block.
        rung = parts[1] if len(parts) > 2 and parts[1].startswith("ladder") else ""
        if rung:
            model = f"{model} @{rung[len('ladder'):]}"
        config = blob.get("config", {})
        by_task: dict[str, list] = defaultdict(list)
        for row in blob.get("rows", []):
            by_task[row["dataset"]].append(row)
        for task, rows in by_task.items():
            aggregate = blob.get("aggregates", {}).get(task, {})
            census = Counter(degeneracy(r.get("prediction", "")) for r in rows)
            cell = {
                "n": len(rows),
                "budget": config.get("effective_input_budget"),
                "requested": config.get("requested_max_input_tokens"),
                "window": config.get("positional_window"),
                "score": aggregate.get("score"),
                "untrunc": aggregate.get("mean_prompt_tokens_untruncated"),
                "seen": aggregate.get("mean_input_tokens_seen"),
                "rows_truncated": aggregate.get("rows_truncated"),
                "seen_fraction": aggregate.get("seen_fraction"),
                "census": census,
                "rows": rows,
                "decode": config.get("decode"),
                "lenient": lenient_credit(rows),
                "digits": emits_digits(rows),
                "goldlen_run": emits_plausible_run(rows),
            }
            if task in NEEDLE_TASKS:
                visible = sum(
                    1 for r in rows
                    if r["id"] in depths[task]
                    and needle_survives(depths[task][r["id"]],
                                        r.get("prompt_tokens_untruncated", 0) or 1,
                                        r.get("input_tokens", 0))
                )
                cell["needle_visible"] = visible
                cell["ceiling"] = visible / len(rows) if rows else 0.0
            table[(task, model)] = cell

    # Ceilings that are arithmetic, not capability: these cells are bounded before any model
    # runs, and quoting them as capability measurements would be wrong in our own favour.
    CAPS = {
        "math_calc": ("max_new_tokens 128 expresses ~40 of a 14999-element ordered gold "
                      "list; official metric is correct-ordered-prefix/len(gold)", 0.0027),
        "longbook_sum_eng": ("gold ~1250 tokens vs 128 generated, so ROUGE recall is "
                             "capped", 0.10),
    }

    order = list(NEEDLE_TASKS) + ["longdialogue_qa_eng", "longbook_sum_eng",
                                  "math_calc", "code_run", "longbook_qa_chn"]
    tasks = [t for t in order if any(k[0] == t for k in table)]
    tasks += sorted({k[0] for k in table} - set(tasks))

    for task in tasks:
        print("=" * 112)
        star = "  [NEEDLE: depth-sorted file, stride-10 sample]" if task in NEEDLE_TASKS else ""
        print(f"{task}{star}")
        if task in CAPS:
            reason, ceiling = CAPS[task]
            print(f"  CAP-BOUNDED (ceiling ~{ceiling:.4f}): {reason}.")
            print("  Report this as a cap, NOT as a capability measurement.")
        header = (f"  {'model':22s} {'n':>3s} {'budget':>8s} {'seen_frac':>9s} "
                  f"{'trunc':>5s} {'score(approx.)':>14s}")
        if task in NEEDLE_TASKS:
            header += f" {'ceiling':>7s} {'vis':>6s}"
        print(header + f" {'lenient':>7s} {'digits':>6s} {'run>=gold':>9s}  {'degeneracy':>12s}")
        for (this_task, model), cell in sorted(table.items()):
            if this_task != task:
                continue
            score = cell["score"]
            line = (f"  {model:22s} {cell['n']:>3d} {str(cell['budget']):>8s} "
                    f"{cell['seen_fraction']:>9.3f} {str(cell['rows_truncated']):>5s} "
                    f"{score:>14.4f}")
            if task in NEEDLE_TASKS:
                line += f" {cell['ceiling']:>7.3f} {cell['needle_visible']:>3d}/{cell['n']:<2d}"
            census = cell["census"]
            bad = ", ".join(f"{k}={v}" for k, v in sorted(census.items()) if k != "ok") or "-"
            print(line + f" {cell['lenient']:>3d}/{cell['n']:<3d} "
                         f"{cell['digits']:>3d}/{cell['n']:<2d} "
                         f"{cell['goldlen_run']:>4d}/{cell['n']:<4d} {bad:>12s}")

        # A score on a middle-truncated excerpt is not a whole-context result. Flag it here so
        # the number can never be quoted without the qualifier, including when the truncated
        # arm is OUR arm.
        truncated = [(m, c) for (t, m), c in sorted(table.items())
                     if t == task and (c["rows_truncated"] or 0) > 0]
        if truncated:
            print("  NOT like-for-like -- these cells scored a middle-truncated excerpt:")
            for model, cell in truncated:
                print(f"    {model:22s} {cell['rows_truncated']}/{cell['n']} rows truncated, "
                      f"mean {cell['untrunc']:.0f} tokens untruncated vs "
                      f"{cell['seen']:.0f} seen (seen_fraction {cell['seen_fraction']:.3f})")

        if args.depth_curve and task in NEEDLE_TASKS:
            print(f"\n  score vs needle depth ({task}):")
            for (this_task, model), cell in sorted(table.items()):
                if this_task != task:
                    continue
                buckets: dict[int, list[float]] = defaultdict(list)
                for row in cell["rows"]:
                    depth = depths[task].get(row["id"])
                    if depth is None:
                        continue
                    buckets[min(4, int(depth * 5))].append(float(row.get("score", 0.0)))
                cells = " ".join(
                    f"q{q}:{sum(v)/len(v):.2f}" if (v := buckets.get(q)) else f"q{q}:  - "
                    for q in range(5))
                print(f"    {model:22s} {cells}")

        if args.quote:
            for (this_task, model), cell in sorted(table.items()):
                if this_task != task:
                    continue
                print(f"\n  --- generations, {model} / {task} ---")
                for row in cell["rows"][:args.quote]:
                    pred = (row.get("prediction") or "").replace("\n", "\\n")
                    gold = ", ".join(str(a) for a in (row.get("answers") or [])[:3])
                    depth = depths.get(task, {}).get(row["id"])
                    tag = f" depth={depth:.3f}" if depth is not None else ""
                    print(f"    id={row['id']}{tag} score={row.get('score')} gold=[{gold}]")
                    print(f"      pred: {pred[:220]!r}")
        print()

    print("=" * 112)
    print("Every score is an APPROXIMATION of the official InfiniteBench metric: the per-task")
    print("METRICS are now verbatim ports of the official get_score_one_<task> (differential")
    print("tested, 15000 cases, 0 mismatches), but the PROMPTS and the truncation are ours, so")
    print("these are not official-leaderboard numbers and must be reported as 'approx.'.")

    ladder_verdict(table)


def ladder_verdict(table: dict[tuple[str, str], dict]) -> None:
    """Pair each model's short-context ladder rungs against its 125k row and state the reading.

    The reading is PRE-REGISTERED in logs/qz/INFBENCH_EXPAND_JOBS.md: the ladder holds the prompt
    template, the rows, the scorer and the budget fixed and varies only the context length, so it
    is the only evidence here that separates "125k length" from "bare `Answer:` suffix" as the
    cause of the 0.0000 at 125k. Printing the verdict mechanically keeps it from being read
    selectively after the fact.
    """
    rungs = sorted({m.split(" @")[1] for (_, m) in table if " @" in m},
                   key=lambda r: int(r.rstrip("k")))
    if not rungs:
        return
    print()
    print("=" * 112)
    print("LENGTH-LADDER CONTROL: same rows, same bare-'Answer:' prompt, same scorer; only the")
    print("context length differs. Needle in-input on 59/59 rows at every rung by construction.")
    for task in NEEDLE_TASKS:
        bases = sorted({m.split(" @")[0] for (t, m) in table if t == task and " @" in m})
        if not bases:
            continue
        print(f"\n  {task}:")
        print(f"    {'model':22s} " + " ".join(f"{r+' (approx.)':>16s}" for r in rungs)
              + f" {'125k (approx.)':>16s}")
        for base in bases:
            parts = []
            for rung in rungs:
                cell = table.get((task, f"{base} @{rung}"))
                parts.append(f"{cell['score']:>16.4f}" if cell else f"{'not run':>16s}")
            full = table.get((task, base))
            parts.append(f"{full['score']:>16.4f}" if full else f"{'not run':>16s}")
            print(f"    {base:22s} " + " ".join(parts))
    # The verdict, from the arms' own numbers.
    arms = ("champ-dynamic-30k", "global-dit-50k")
    short = [c["score"] for (t, m), c in table.items()
             if " @" in m and m.split(" @")[0] in arms and t in NEEDLE_TASKS]
    llama = [c["score"] for (t, m), c in table.items()
             if " @" in m and m.split(" @")[0] == "llama-3.2-3b" and t in NEEDLE_TASKS]
    print()
    if llama and max(llama) < 0.5:
        print("  VOID: llama-3.2-3b fails the short rung too, so the derived dataset or the")
        print("  template is broken. The ladder proves nothing until that is fixed.")
    elif short and max(short) >= 0.5:
        print("  READING: the arms DO answer at short context under this exact prompt, so the")
        print("  0.0000 at 125k is a LENGTH effect, not a prompt-format artefact.")
    elif short:
        print("  READING: the arms fail at short context too, under a prompt a transformer")
        print("  answers. So the 125k 0.0000 is NOT established as a length effect -- the bare")
        print("  'Answer:' template is implicated and the mechanism claim must be weakened.")


if __name__ == "__main__":
    main()

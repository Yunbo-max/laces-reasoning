#!/usr/bin/env python3
"""Build SHORT-context variants of InfiniteBench passkey/number_string.

Why this exists
---------------
The headline finding is that both relay arms score 0.0000 (approx.) on `passkey` and
`number_string` at 125k tokens while two 131072-window transformers score ~1.0, with the needle
provably in-input on 59/59 rows. The natural mechanism claim is "length", and the natural
supporting evidence is the RULER `niah_single_1` ladder (1.000 at 4k, 1.000 at 16k for
global-dit-50k). But RULER's prompt ends in a *completion cue*
(`"The special magic number ... is"`) while our InfiniteBench prompt ends in a bare `"Answer:"`.
So that ladder varies BOTH length and prompt format, and cannot separate them.

The `kv_retrieval`/`math_find` "control" recorded earlier does not separate them either: both of
those tasks are in INFINITEBENCH_CONTINUATION_TASKS and their own `input` ends in ":", so they are
also fed as completion cues.

This script builds the missing discriminator: the SAME task, the SAME rows, the SAME
`build_infinitebench_prompt` template (bare `"Answer:"`), the SAME scorer -- only the context
length changes. If the arms answer at 4k and fail at 125k, length is the cause. If they fail at 4k
too, the bare `"Answer:"` template is implicated and the mechanism story must be rewritten.

How the shortening is done
--------------------------
Truncating the *prompt* at a small budget would delete the needle on most rows (middle truncation
keeps head+tail), which is exactly the unanswerable-by-construction trap the main run avoided with
a 131072 budget. So this shortens the *context* instead, and keeps the needle by construction:

- The context is `PREFIX + needle_sentences + FILLER * k + SUFFIX`, where the filler is the
  repeated `"The grass is green. The sky is blue. The sun is yellow. Here we go. There and back
  again. "` sentence group (measured: 5220 repeats in row 0 of both tasks).
- We keep the needle at its ORIGINAL RELATIVE DEPTH and drop filler either side to hit the target
  character budget. Depth is preserved to within one filler group, so the depth curve remains
  comparable to the 125k run.

The output files are written to a separate data dir and are NOT the official dataset; any cell
produced from them is labelled as a length-ladder control, still "approx.".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SNAP = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/"
            "datasets--xinrongzhang2022--InfiniteBench/snapshots/"
            "90f0394333616266d9fe85824ceaf505093cbaa5")
FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")


def rebuild(context: str, gold: str, target_chars: int) -> tuple[str, float, float]:
    """Shorten `context` to ~`target_chars` keeping `gold` at its original relative depth.

    Returns (new_context, old_depth, new_depth). Raises if the gold is absent.
    """
    first = context.find(gold)
    if first < 0:
        raise ValueError("gold not in context")
    old_depth = first / len(context)

    # The needle block is the span covering both gold occurrences plus their sentence framing.
    last = context.rfind(gold)
    block_start = context.rfind(". ", 0, first)
    block_start = 0 if block_start < 0 else block_start + 2
    block_end = context.find(". ", last + len(gold))
    block_end = len(context) if block_end < 0 else block_end + 2
    needle_block = context[block_start:block_end]

    room = max(0, target_chars - len(needle_block))
    # Split the filler budget so the needle lands at old_depth in the NEW context.
    before_chars = int(round(room * old_depth))
    after_chars = room - before_chars
    before = (FILLER * (before_chars // len(FILLER) + 1))[:before_chars]
    after = (FILLER * (after_chars // len(FILLER) + 1))[:after_chars]
    new_context = before + needle_block + after
    new_depth = new_context.find(gold) / len(new_context)
    return new_context, old_depth, new_depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_tokens", type=int, required=True,
                        help="approx token budget; chars ~= 3.6 x tokens for this filler")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--chars_per_token", type=float, default=3.6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_chars = int(args.target_tokens * args.chars_per_token)

    for task in ("passkey", "number_string"):
        src = SNAP / f"{task}.jsonl"
        kept, depth_err = 0, 0.0
        with (out_dir / f"{task}.jsonl").open("w") as out:
            with src.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if int(item.get("id", -1)) % args.stride:
                        continue
                    answers = item.get("answer") or []
                    gold = str(answers[0] if isinstance(answers, list) else answers)
                    new_context, old_depth, new_depth = rebuild(
                        str(item["context"]), gold, target_chars)
                    depth_err = max(depth_err, abs(old_depth - new_depth))
                    item["context"] = new_context
                    out.write(json.dumps(item) + "\n")
                    kept += 1
        print(f"{task}: {kept} rows -> {out_dir/f'{task}.jsonl'}  "
              f"~{target_chars} chars (~{args.target_tokens} tok), "
              f"max |depth shift| = {depth_err:.4f}")


if __name__ == "__main__":
    main()

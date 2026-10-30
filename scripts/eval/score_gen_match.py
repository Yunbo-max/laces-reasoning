#!/usr/bin/env python3
"""Unified generate-then-match scorer for all variants (ours + external baselines).

Why this exists: earlier ad-hoc scorers mismatched task formats (e.g. RACE ground
truths are full sentences, not letters), understating every model's RACE score.
This scorer applies ONE rule set to every variant:

  free-form tasks (squad, ...)   -> token-F1 / EM, generation truncated at first newline
  choice tasks (mmlu, obqa, race, story_cloze, hellaswag, ...) -- the rule SCORING.md
  states and the default here:
      1. pick the choice with the best match to the generation, in priority:
         exact normalized equality > generation starts with choice (or vice versa)
         > highest token-F1
      2. if no choice matches, fall back to parsing an option letter
      3. correct iff the picked choice is the ground-truth choice

  The note above step 1 used to be a lie: this file's own header claimed the letter was
  tried FIRST while the code (and SCORING.md) tried it last.  The two agree now.

  ONE AMENDMENT, opt-in via SCORE_LETTER_FIRST=1, because the prompt fix makes models
  answer with a bare option label ('(A)', 'B.', 'C') rather than free text:

      a generation that is NOTHING BUT an option label is scored as that option.

  Without it, a bare label reaches step 1 and matches a choice by its own first letter
  -- normalize_text('(A)') == 'a', so 'airplane' wins step 1's "generation starts with
  choice" test and the letter is never consulted.  Measured on the fix2choices runs, a
  bare label agrees with its own letter only 4-7% of the time, i.e. bare-label answers
  are effectively randomised.  The default (off) reproduces every previously reported
  number; turn it on to score bare-label answers by their label.

Usage:  score_gen_match.py <variant_dir> [<variant_dir> ...]
        (prints one row per variant; dirs may contain gen/ or flat jsonl)
        SCORE_LETTER_FIRST=1 score_gen_match.py <variant_dir>
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/baseline/Cola-DLM")
from scripts.acc_calc import normalize_text, process_line  # noqa: E402

LETTERS = "ABCDEFGH"
# Opt-in protocol amendment: score a generation that is nothing but an option label as
# that option, before choice matching gets a chance to match it by first letter. Off by
# default so every previously reported number still reproduces byte for byte.
LETTER_FIRST = os.environ.get("SCORE_LETTER_FIRST", "0") == "1"


def truncate(gen: str) -> str:
    """Cut at the first newline (mirrors acc_calc.process_line)."""
    if not isinstance(gen, str):
        return ""
    idx = gen.find("\n")
    return gen[:idx] if idx >= 0 else gen


def letter_of(text: str) -> str | None:
    """Parse an option letter from '(C) ...', 'C) ...', 'C.' or a bare 'C'.

    The third case is a bare label glued to the next token with no separator:
    under the 32-token cap the decoder often emits 'BQuestion: ...', running the
    answer straight into the next few-shot item. Requiring a separator dropped
    those as unmatched, which cost the write-in rows far more than the
    HuggingFace-path rows (the pattern occurs 77-176 times per row there against
    0-1 for the autoregressive baselines), so the rule is applied to every row.
    Verified as signal rather than a length heuristic: on the entries it newly
    scores, the letter is correct 74-78% of the time against a 24-37% chance
    baseline for those tasks' option counts.
    """
    t = text.strip()
    m = re.match(r"^[\(\[]?([A-Ha-h])[\)\].:,]\s*", t)
    if m:
        return m.group(1).upper()
    if len(t) <= 2 and t.upper() in LETTERS:
        return t.upper()
    m = re.match(r"^[\(\[]?([A-Ha-h])(?=[A-Z])", t)
    if m:
        return m.group(1).upper()
    return None


def f1(pred: str, gt: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gt).split()
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def strip_label(text: str) -> str:
    """Remove a leading answer label such as 'A:', '(A)', 'Answer:' so that the
    following choice TEXT (not the letter) drives the match when both appear."""
    return re.sub(r"^\s*(?:answer\s*[:\-]\s*)?[\(\[]?[A-Ha-h][\)\].:,]\s*", "", text, count=1, flags=re.IGNORECASE).strip()


ANSWER_LEAD = re.compile(
    r"^\s*(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)?\s*", re.IGNORECASE)


def strip_answer_lead(text: str) -> str:
    """Drop a lead-in like 'The answer is' / 'Answer:' so the payload starts the string."""
    return ANSWER_LEAD.sub("", text, count=1).strip()


def contains_choice(gen: str, choices: list[str]) -> int | None:
    """Pick the (longest) choice whose normalised text occurs inside the generation.

    Free-form answering ('The answer is nervous system. Explanation: ...') states the
    option without opening with it, so containment is the rule that scores it; taking
    the longest match keeps nested choices (e.g. 'refrigerator' inside 'the refrigerator')
    unambiguous.
    """
    ng = normalize_text(gen)
    if not ng:
        return None
    best, best_len = None, 0
    for i, c in enumerate(choices):
        nc = normalize_text(c)
        if nc and nc in ng and len(nc) > best_len:
            best, best_len = i, len(nc)
    return best


def match_choice(gen: str, choices: list[str]) -> int | None:
    """Return the index of the choice the generation matches, or None."""
    if not choices:
        return None
    ng = normalize_text(gen)
    if not ng:
        return None
    norms = [normalize_text(c) for c in choices]
    # 1) exact
    for i, c in enumerate(norms):
        if c and c == ng:
            return i
    # 2) prefix (either direction), prefer the longest match
    best, best_len = None, 0
    for i, c in enumerate(norms):
        if not c:
            continue
        if ng.startswith(c) or c.startswith(ng):
            L = min(len(c), len(ng))
            if L > best_len:
                best, best_len = i, L
    if best is not None:
        return best
    # 2.5) containment: the option text appears somewhere in the generation
    hit = contains_choice(gen, choices)
    if hit is not None:
        return hit
    # 3) token-F1 argmax, with a floor so garbage does not match by accident
    scores = [(f1(gen, c), i) for i, c in enumerate(choices)]
    scores.sort(reverse=True)
    if scores and scores[0][0] >= 0.5:
        return scores[0][1]
    return None


def score_file(path: str) -> tuple[str, float, float, int]:
    task = os.path.basename(path)[:-6]
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n = len(recs)
    if n == 0:
        return task, float("nan"), float("nan"), 0
    acc = 0
    f1s, ems = [], []
    for r in recs:
        raw = r.get("generate", "")
        gen = truncate(raw)
        gt = str(r.get("ground_truth", ""))
        choices = r.get("choices") or []
        if choices:
            # prefer the choice TEXT (after stripping any label), then the letter
            pred = None
            if LETTER_FIRST:
                # A generation that carries no content beyond an option label is that
                # option. Without this, "(A)" normalises to "a" and wins the prefix
                # test against any choice starting with "a", so the label is ignored.
                # Two spellings count as a bare label: "(A)" / "A." / "A:" (strip_label
                # leaves nothing behind) and a bare "A" / "B)" (<= 2 chars in total).
                bare = letter_of(gen)
                if bare is not None and (len(strip_label(gen)) == 0 or len(gen.strip()) <= 2) \
                        and LETTERS.index(bare) < len(choices):
                    pred = LETTERS.index(bare)
            if pred is None:
                pred = match_choice(gen, choices)
            if pred is None:
                pred = match_choice(strip_label(gen), choices)
            if pred is None:
                pred = match_choice(strip_answer_lead(strip_label(gen)), choices)
            if pred is None:
                lt = letter_of(gen)
                if lt is not None and LETTERS.index(lt) < len(choices):
                    pred = LETTERS.index(lt)
            # ground-truth index
            gold = None
            ng = normalize_text(gt)
            norms = [normalize_text(c) for c in choices]
            for i, c in enumerate(norms):
                if c == ng:
                    gold = i
                    break
            if gold is None:
                gold = match_choice(gt, choices)
            acc += int(pred is not None and gold is not None and pred == gold)
        else:
            # free-form (squad-like) or word-level (lambada)
            if task.startswith("lambada"):
                # LAMBADA: the gold answer is the final word; the generation must
                # open with it (allow leading whitespace/punctuation).
                first = normalize_text(gen).split()
                head = " ".join(first[:3])
                f1s.append(1.0 if normalize_text(gt) and normalize_text(gt) in head else 0.0)
                ems.append(float(bool(first) and first[0] == normalize_text(gt)))
            else:
                f1s.append(f1(gen, gt))
                ems.append(float(normalize_text(gen) == normalize_text(gt)))
    if f1s:
        return task, sum(f1s) / len(f1s) * 100, sum(ems) / len(ems) * 100, n
    return task, acc / n * 100, float("nan"), n


def collect(d: str) -> dict:
    out = {}
    for sub in ("gen", "cola8", ""):
        base = os.path.join(d, sub) if sub else d
        if not os.path.isdir(base):
            continue
        for f in os.listdir(base):
            if f.endswith(".jsonl") and not f.endswith(("_correct.jsonl", "_wrong.jsonl")):
                out.setdefault(f[:-6], os.path.join(base, f))
    return out


def main():
    dirs = sys.argv[1:]
    tasks = ["mmlu", "obqa", "race", "story_cloze", "hellaswag", "siqa", "lambada", "squad"]
    print(f"{'variant':24s}" + "".join(f"{t[:9]:>10s}" for t in tasks))
    for d in dirs:
        files = collect(d)
        name = os.path.basename(d.rstrip("/"))
        row = f"{name:24s}"
        for t in tasks:
            p = files.get(t)
            if not p:
                row += f"{'--':>10s}"
                continue
            _, a, _, _ = score_file(p)
            row += f"{a:10.1f}" if a == a else f"{'--':>10s}"
        print(row, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aligned open-ended context evaluation for the 30k dynamic LACES checkpoint.

Suites (``--suite``):
  longbench      LongBench QA/summarisation via the pinned ``dataset2prompt`` templates
  babilong       bAbILong, the HF cache ``data/qa<N>/<len>.json`` files
  ruler          RULER via the cached per-length parquet
  infinitebench  InfiniteBench long-context tasks (jsonl)
  memeval        LongMemEval ``longmemeval_{s,m,oracle}`` transcripts

Every loader returns rows shaped ``{"id", "prompt", "answers", ...}``; the runner is
model-agnostic (``--model_dir`` HF baseline or ``--ckpt_dir`` relay checkpoint).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

# Running this file as a script puts ``scripts/eval/`` on sys.path, NOT the repo root, so the
# relay branch's ``from models.laces_latent_refiner import LACESRuntime`` dies with
# ModuleNotFoundError (reproduced 2026-09-20 with the job's own env: the job's PYTHONPATH
# holds only shared_pylibs). Every sibling evaluator in this directory
# (sample_prefix_suffix_cfg.py, run_cola_dlm_tasks_prefix_suffix_cfg.py) inserts the repo root
# the same way -- keep this consistent with them. The --model_dir (HF) branch never needs it,
# which is why the baseline runs were unaffected.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


QA_DATASETS = {
    "hotpotqa", "2wikimqa", "musique", "narrativeqa", "multifieldqa_en"
}
SUMMARY_DATASETS = {"gov_report", "qmsum", "multi_news"}
BABILONG_TASK_LABELS = {
    "qa1": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa2": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa3": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa6": ["no", "yes"],
    "qa9": ["no", "yes"],
}

# --- InfiniteBench task families (see infinitebench_score for the metric rules) --------
# Tasks whose gold label is one of the ``options`` list and whose official answer format
# is a multiple-choice letter.
INFINITEBENCH_CHOICE_TASKS = {"longbook_choice_eng", "code_debug"}
# Tasks whose gold label is *every* value in a list (official: all of them must appear).
INFINITEBENCH_ALL_REF_TASKS = {"kv_retrieval"}
# Tasks whose gold label is one or more numbers (official compares parsed numbers).
# ``math_calc`` and ``code_run`` USED to live here and were moved out on 2026-09-21 when the
# official ``src/compute_scores.py`` was finally retrieved (see the block comment below): both
# have exact official rules that the set-recall branch approximates in the *inflating*
# direction. ``math_find`` is deliberately left on the old branch so its already-published
# cells stay bit-identical -- it is another line's result, not this one's, to change.
INFINITEBENCH_NUMERIC_TASKS = {"math_find"}
# ``passkey`` / ``number_string``: official rule is ``label == first_int_match(pred)`` -- the
# FIRST run of digits in the generation must equal the gold string exactly. NOT containment.
# This distinction is the whole experiment: a passkey context states the key twice in its
# first sentence, so a model that merely echoes the document satisfies a containment test and
# scores 1.0 while the official scorer gives it 0.0.
INFINITEBENCH_FIRST_INT_TASKS = {"passkey", "number_string"}
# ``code_run``: official parses the LAST whitespace-separated word as an int.
INFINITEBENCH_LAST_INT_TASKS = {"code_run"}
# ``math_calc``: official is an ordered PREFIX match over the intermediate-result list that
# stops at the first mismatch, not set recall.
INFINITEBENCH_PREFIX_NUM_TASKS = {"math_calc"}
# ``longdialogue_qa_eng``: official upper-cases the prediction and accepts ANY alias.
INFINITEBENCH_ANY_LABEL_TASKS = {"longdialogue_qa_eng"}
# ``longbook_qa_chn``: official is character-level F1 after Chinese punctuation stripping.
INFINITEBENCH_ZH_F1_TASKS = {"longbook_qa_chn"}
# Tasks whose gold label is long prose (official reports ROUGE-L).
INFINITEBENCH_ROUGE_TASKS = {"longbook_sum_eng"}
# Tasks whose ``input`` field is already a completion cue ("... is: ") that the official
# prompts feed to the model verbatim; everything else gets an explicit "Answer:" suffix.
INFINITEBENCH_CONTINUATION_TASKS = {"longbook_sum_eng", "kv_retrieval", "math_find", "code_run"}


def as_list(value: Any) -> list[Any]:
    """Normalise an answers/options field to a list.

    The cached copies are inconsistent: LongBench's jsonl stores ``"['Miller v. California']"``
    (a *stringified* list -- ``list("...")`` would hand back single characters, which is how
    every LongBench prediction silently picked up ROUGE-F1 against letters), RULER's parquet
    hands pandas numpy arrays, and MemEval stores a bare string. Accept all four shapes.
    """
    if value is None:
        return []
    if hasattr(value, "tolist"):                 # numpy array out of pandas
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return [value]
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        return [value]
    return [value]


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalized_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def qa_f1(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    common = Counter(pred) & Counter(gold)
    overlap = sum(common.values())
    if overlap == 0 or not pred or not gold:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, ground_truth: str) -> float:
    """LongBench's official ``rouge==1.0.1`` ROUGE-L F1."""
    from rouge import Rouge

    try:
        return float(Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"])
    except (ValueError, RecursionError):
        return 0.0


def babilong_score(task: str, prediction: str, target: str, question: str) -> float:
    """BABILong's official label-aware ``compare_answers`` for selected tasks."""
    if task not in BABILONG_TASK_LABELS:
        raise KeyError(
            f"babilong_score has no official label set for task {task!r}; it currently "
            f"covers {sorted(BABILONG_TASK_LABELS)}. Add the task's label set before "
            f"running it -- scoring an unsupported task silently would be worse."
        )
    output = prediction.lower().split(".")[0]
    output = output.split("<context>")[0].split("<example>")[0].split("question")[0]
    labels = {label.lower() for label in BABILONG_TASK_LABELS[task]}
    labels_in_question = {label for label in labels if label in question.lower()}
    labels_in_output = {label for label in labels if label in output} - labels_in_question
    target = target.lower()
    if "," in target and len(target) > 3:
        subtargets = target.split(",")
        return float(all(item in labels_in_output for item in subtargets) and
                     len(labels_in_output) == len(subtargets))
    return float(target in labels_in_output and len(labels_in_output) == 1)


# RULER assigns each task one of two official metrics (see RULER's
# scripts/data/synthetic/constants.py).  ``string_match_part`` takes the max over
# golds and is used only by the two QA tasks, where the references are alternative
# spellings of one answer.  Everything else uses ``string_match_all``, i.e. the
# per-sample recall over all golds -- taking a max there inflates cwe (10 golds)
# and fwe (3 golds) several-fold.
RULER_MATCH_PART_TASKS = {"qa_1", "qa_2"}


def ruler_score(task: str, prediction: str, references: list[str]) -> float:
    """Per-sample equivalent of RULER's official substring metrics."""
    prediction = prediction.lower()
    hits = [float(reference.lower() in prediction) for reference in references]
    if not hits:
        return 0.0
    if task in RULER_MATCH_PART_TASKS:
        return max(hits)
    return sum(hits) / len(hits)


def normalize_freeform(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace -- for substring comparisons.

    U+FFFD is mapped to whitespace first. Our relay arms *emit* token 193 where every HF
    baseline emits token 11: per ``rwkv_vocab_v20230424.txt`` id 193 is the lone byte 0xC0
    (invalid UTF-8 on its own, so it decodes to U+FFFD) while the real newline is id 11.
    This is NOT a tokenizer-directory mismatch -- ``rwkv_vocab_v20230424.txt``,
    ``tokenizer_config.json``, ``added_tokens.json`` and ``hf_rwkv_tokenizer.py`` are
    byte-identical (``cmp``) between the relay's ``RWKV7-Goose-World3-2.9B-HF-fla-v2`` and the
    baseline's ``RWKV7-Goose-World3-2.9B-HF``, and both decode 193 -> U+FFFD, 11 -> "\\n". The
    difference is in which id the model emits: over the 50 memeval rows champ-dynamic-30k
    emits 193 173 times and 11 zero times, while rwkv7-world3 emits 11 899 times and 193
    never. So it is a defect in our arm's own output distribution, affecting 133/250
    InfiniteBench rows and 41/50 memeval rows. U+FFFD is not in ``string.punctuation``, so
    without this it survived normalisation and *glued* the words on either side of it
    ("answer<FFFD>United" -> "answer<FFFD>united"), making a containment test fail on a gold
    the model had actually produced. Measured effect on InfiniteBench: our arm gains at most
    +0.001 and every baseline is bit-identical, so it changes no conclusion there -- but a
    scorer must not be sensitive to which tokens produced the text. Credit: found by the
    MemEval line, where the same defect was worth -0.02 because that harness cuts the answer
    span at a newline (and there it was *inflating* our arm, not deflating it).
    """
    text = text.lower().replace("�", " ")
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


# --------------------------------------------------------------------------------------
# InfiniteBench
# --------------------------------------------------------------------------------------
# Source: InfiniteBench (xinrongzhang2022/InfiniteBench, also published as
# zhang-wei-cs/InfiniteBench), ``src/eval_utils.py``, which defines one
# ``get_score_one_<task>`` per task and an ``extract_answer`` post-processor.
#
# APPROXIMATION, stated plainly: this node has no route to github (checked 2026-09-20:
# the curl to raw.githubusercontent.com times out), so the official file could not be
# fetched and the rules below are re-derived from the task definitions rather than
# copied. Every branch that is not a direct substring comparison says "approx." in its
# comment. Nothing here should be reported as "the official InfiniteBench metric".
#
# UPDATE 2026-09-21: the official files WERE retrieved, from the *login* node, which does
# reach raw.githubusercontent.com (the compute nodes still do not; the 2026-09-20 note above
# was measured from a compute node and is still true there). Retrieved:
#   https://raw.githubusercontent.com/OpenBMB/InfiniteBench/main/src/compute_scores.py
#   https://raw.githubusercontent.com/OpenBMB/InfiniteBench/main/src/eval_utils.py
# The per-task scorer dispatch lives in compute_scores.py (get_score_one_<task>), NOT in
# eval_utils.py where the earlier search looked -- which is why it was believed unreachable.
# Branches ported verbatim from it are now marked "DIRECT port of official ..." and may be
# called exact; branches still marked "approx." are the ones the official code reaches
# through a dependency this image lacks (HF `evaluate` rougeLsum, jieba) or through a chat
# template a completion harness cannot reproduce. The suite as a whole is still reported as
# "approx." because the PROMPTS remain ours and truncation is ours -- a matching metric on a
# different prompt is not the official benchmark number.


def build_infinitebench_prompt(item: dict[str, Any], task: str = "") -> str:
    """``context`` + the task's own ``input``, plus options for the multiple-choice tasks.

    Our template, not the official string: the official harness renders these through
    per-task prompt files. Tasks listed in INFINITEBENCH_CONTINUATION_TASKS already end in
    a completion cue ("The summary of the book is: ") and are fed verbatim; the rest get an
    explicit "Answer:" so a fixed-greedy decoder does not ramble.

    ``task`` has to be passed in: the jsonl files carry no task column, so the filename stem
    is the only source. Without it this silently fell through to the ``endswith(":")``
    heuristic and appended "Answer:" to code_run, whose own instruction already ends the
    response.
    """
    context = str(item.get("context", "")).rstrip()
    question = str(item.get("input", "")).strip()
    options = [str(option) for option in as_list(item.get("options"))]
    if options:
        block = "\n".join(f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(options))
        return f"{context}\n\n{question}\n\nOptions:\n{block}\n\nAnswer:"
    if task in INFINITEBENCH_CONTINUATION_TASKS or question.endswith(":"):
        return f"{context}\n\n{question}"
    return f"{context}\n\n{question}\n\nAnswer:"


def _infinitebench_choice_index(prediction: str, options: list[str]) -> int | None:
    """Best-effort option index from a free-form generation.

    approx.: the official scorer extracts the option *letter* from the generation. We
    accept the option text as well, and try the text first -- a letter regex fires on
    ordinary prose ("A function ...") far too easily. Returns None when nothing matches.

    Two defects fixed 2026-09-21, both found while auditing the all-zero InfiniteBench run:

    1. FIRST occurrence, not LONGEST. The old rule picked whichever option text was longest
       among those present in the generation. But a greedy decoder fed an "Options:\\nA...D"
       block very often echoes the whole block back, so several option texts appear and the
       longest one is arbitrary -- it is a property of the option list, not of the answer.
       Measured disagreement between longest and first-mentioned on the stored predictions:
       gla-1.3b 20/50 on longbook_choice_eng, qwen2.5-1.5b 14/50, champ-dynamic-30k 11/50.
       The model's answer is what it says FIRST; an echo of the menu comes after it.

    2. Glued bare letter. The letter regex required a separator after the letter, so a
       generation like "BQuestion: ..." -- the letter running straight into the next
       question with no space, which is common under a tight token cap -- was judged
       unmatched and scored 0 even when B was right. This is the same defect that cost the
       short-context raw control 5.2 points in score_gen_match.letter_of. A bare leading
       letter is now accepted when what follows is not itself a lowercase word character,
       so "B" / "B." / "BQuestion:" all resolve to 1 while "Bella walked" does not.
    """
    prediction_normalized = normalize_freeform(prediction)
    best, best_position = None, None
    for index, option in enumerate(options):
        normalized = normalize_freeform(option)
        if not normalized:
            continue
        position = prediction_normalized.find(normalized)
        if position >= 0 and (best_position is None or position < best_position):
            best, best_position = index, position
    if best is not None:
        return best
    probe = re.sub(r"^(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*(?:is)?\s*[:\-]?\s*",
                   "", prediction.strip(), flags=re.IGNORECASE)
    match = (re.match(r"[\(\[]?([A-Da-d])[\)\]\.\:\s,]", probe)
             or re.match(r"^[\(\[]?([A-Da-d])$", probe)
             # Glued: "BQuestion:" -- a letter followed by anything that cannot continue a
             # lowercase word. Requiring [A-Z(] or similar keeps "Bella" from matching as B.
             or re.match(r"^([A-D])(?=[^a-z])", probe))
    if not match:
        return None
    index = ord(match.group(1).upper()) - ord("A")
    return index if index < len(options) else None


def infinitebench_score(task: str, prediction: str, references: Any,
                        options: Any = None) -> float:
    """Per-sample InfiniteBench score. See the block comment above: APPROXIMATION."""
    answers = [str(reference) for reference in as_list(references)]
    if not answers:
        return 0.0
    option_list = [str(option) for option in as_list(options)]

    if task in INFINITEBENCH_CHOICE_TASKS:
        # approx.: option-text-or-letter match against the gold option.
        gold_normalized = {normalize_freeform(answer) for answer in answers}
        if not option_list:
            prediction_normalized = normalize_freeform(prediction)
            return float(any(gold in prediction_normalized for gold in gold_normalized))
        gold_index = next((i for i, option in enumerate(option_list)
                           if normalize_freeform(option) in gold_normalized), None)
        predicted_index = _infinitebench_choice_index(prediction, option_list)
        if gold_index is None:
            # Gold is not one of the offered options: fall back to a text match.
            return float(predicted_index is not None
                         and normalize_freeform(option_list[predicted_index]) in gold_normalized)
        return float(predicted_index is not None and predicted_index == gold_index)

    if task in INFINITEBENCH_ROUGE_TASKS:
        # Direct: the official summarisation score is ROUGE-L against the reference summary.
        return max((rouge_l_f1(prediction, answer) for answer in answers), default=0.0)

    if task in INFINITEBENCH_ALL_REF_TASKS:
        # Direct-in-spirit: official kv_retrieval requires *every* recalled value to appear.
        prediction_normalized = normalize_freeform(prediction)
        return float(all(normalize_freeform(answer) in prediction_normalized
                         for answer in answers))

    if task in INFINITEBENCH_NUMERIC_TASKS:
        # approx.: official parses the numbers out of both sides and compares them in
        # order (math_calc scores the whole intermediate-result list). This is set-recall
        # over the gold numbers -- order-insensitive and indifferent to duplicates.
        gold = [n for answer in answers for n in re.findall(r"-?\d+", answer)]
        if not gold:
            return 0.0
        predicted = set(re.findall(r"-?\d+", prediction))
        return sum(1 for number in gold if number in predicted) / len(gold)

    if task in INFINITEBENCH_FIRST_INT_TASKS:
        # DIRECT port of official get_score_one_passkey / _number_string:
        #     return label == first_int_match(pred)
        # where first_int_match splits on [^0-9] and takes the first non-empty run.
        # Exact string equality, so "00071432" != "71432" and a leading distractor digit
        # scores 0 even when the gold appears later. Load-bearing for this suite: the
        # passkey context opens with "The pass key is 71432. Remember it. 71432 is the pass
        # key." -- under the previous default (containment) ANY model that echoed its own
        # prompt scored 1.0. That is the single most inflating defect possible here, and it
        # would have inflated our arm exactly where we claim a win.
        digits = [run for run in re.split(r"[^0-9]", prediction) if run]
        first = digits[0] if digits else ""
        return float(any(first == answer.strip() for answer in answers))

    if task in INFINITEBENCH_LAST_INT_TASKS:
        # DIRECT port of official get_score_one_code_run: strip \n . ` ' " : to spaces, take
        # the LAST whitespace-separated word, int() it, compare numerically.
        cleaned = prediction.strip()
        for char in ("\n", ".", "`", "'", '"', ":"):
            cleaned = cleaned.replace(char, " ")
        words = cleaned.split()
        if not words:
            return 0.0
        try:
            value = int(words[-1])
        except ValueError:
            return 0.0
        for answer in answers:
            try:
                if value == int(str(answer).strip()):
                    return 1.0
            except ValueError:
                continue
        return 0.0

    if task in INFINITEBENCH_PREFIX_NUM_TASKS:
        # DIRECT port of official get_score_one_math_calc: walk the gold list and the
        # predicted number list in lockstep and BREAK at the first mismatch, scoring
        # matched_prefix_len / len(gold). The old set-recall branch was strictly more
        # generous -- it credited every gold number appearing anywhere, in any order, so a
        # model emitting "0 1 2 ... 99" would score highly on a gold list drawn from 0-99.
        gold_numbers = [n for answer in answers for n in re.findall(r"-?\d+", str(answer))]
        if not gold_numbers:
            return 0.0
        predicted = [run for run in re.split(r"[^0-9]", prediction) if run]
        matched = 0
        for index, gold in enumerate(gold_numbers):
            if index >= len(predicted) or int(predicted[index]) != int(gold):
                break
            matched += 1
        return matched / len(gold_numbers)

    if task in INFINITEBENCH_ANY_LABEL_TASKS:
        # DIRECT port of official get_score_one_longdialogue_qa_eng: upper-case both sides
        # and accept if ANY alias is a substring. Punctuation is NOT stripped officially, so
        # this uses a plain case-folded containment rather than normalize_freeform.
        upper = prediction.strip().upper()
        return float(any(str(answer).upper() in upper for answer in answers if str(answer)))

    if task in INFINITEBENCH_ZH_F1_TASKS:
        # approx. of official qa_f1_score_zh: character-level F1 after stripping ASCII and
        # CJK punctuation and all whitespace. Official additionally runs no segmentation for
        # this metric (one character = one token), which is reproduced exactly; what is NOT
        # reproduced is jieba-based ROUGE (that is the summarisation metric, not this one).
        cjk_punctuation = ("！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、"
                           "〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.")
        drop = set(string.punctuation) | set(cjk_punctuation)

        def normalize_zh(text: str) -> str:
            return "".join(ch for ch in text.lower() if ch not in drop and not ch.isspace())

        predicted_chars = list(normalize_zh(prediction))
        best = 0.0
        for answer in answers:
            gold_chars = list(normalize_zh(str(answer)))
            common = Counter(predicted_chars) & Counter(gold_chars)
            overlap = sum(common.values())
            if overlap == 0 or not predicted_chars or not gold_chars:
                continue
            precision = overlap / len(predicted_chars)
            recall = overlap / len(gold_chars)
            best = max(best, 2 * precision * recall / (precision + recall))
        return best

    # Default (longbook_qa_eng / anything unseen): approx. -- the official QA scorers reduce
    # to "is the gold answer present in the generation", after a task-specific
    # post-processor (first_capital_postprocess etc.) that we do not reproduce.
    prediction_normalized = normalize_freeform(prediction)
    if any(normalize_freeform(answer) in prediction_normalized for answer in answers):
        return 1.0
    return max((qa_f1(prediction, answer) for answer in answers), default=0.0)


def load_infinitebench_rows(path: Path) -> list[dict[str, Any]]:
    """InfiniteBench jsonl rows.

    Layout verified against the cached snapshot 2026-09-20: every task ships
    ``id / context / input / answer / options``. ``context`` is the long document, ``input``
    the question or completion cue, ``answer`` a list of acceptable strings, ``options`` a
    list (empty for the free-form tasks).
    """
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append({
                "id": item.get("id", len(rows)),
                "prompt": build_infinitebench_prompt(item, task=path.stem),
                "answers": [str(answer) for answer in as_list(item.get("answer"))],
                "options": [str(option) for option in as_list(item.get("options"))],
            })
    return rows


# --------------------------------------------------------------------------------------
# LongMemEval
# --------------------------------------------------------------------------------------
# Source: LongMemEval (xiaowu0162/LongMemEval). The official scorer is
# ``src/evaluation/evaluate_qa.py``: per question type it builds a judge prompt with
# ``get_anscheck_prompt(...)`` and asks an LLM (gpt-4o) for a yes/no verdict, so the
# published LongMemEval number is *LLM-judged accuracy* (plus a separate abstention
# accuracy for the question ids ending in ``_abs``).
#
# That judge cannot run here -- no network on the compute nodes. What follows is an
# APPROXIMATION and is named as one in the function name on purpose.
# It also cannot score abstention: a model that correctly answers "I don't know" gets 0.0
# from a containment check. Expect the numbers to sit *below* the official ones and to be
# systematically 0 on the 30 ``_abs`` questions of each split.


def build_memeval_prompt(item: dict[str, Any]) -> str:
    """Flatten the haystack sessions into a plain-text transcript.

    Our template, not the official one: the official harness feeds ``haystack_sessions``
    through the model's chat template. A completion harness has to serialise them, so we
    prefix each session with its date -- which matters, because two of the six question
    types (``temporal-reasoning`` and ``knowledge-update``) cannot be answered without it.
    """
    lines: list[str] = []
    sessions = item.get("haystack_sessions") or []
    dates = item.get("haystack_dates") or []
    for index, session in enumerate(sessions):
        date = dates[index] if index < len(dates) else "unknown date"
        lines.append(f"=== Session {index + 1} ({date}) ===")
        for message in session:
            lines.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        lines.append("")
    transcript = "\n".join(lines)
    return (
        f"Below are the sessions of my past conversation history with you.\n\n"
        f"{transcript}\n"
        f"Question (asked on {item.get('question_date', '')}): {item.get('question', '')}\n"
        f"Answer:"
    )


def memeval_answer_span(prediction: str) -> str:
    """The model's FIRST answer turn, before it drifts back into transcript mode.

    Load-bearing. ``build_memeval_prompt`` ends with "Question (asked on ...): ...\\nAnswer:",
    so a *completion* model very often answers and then keeps going: it invents the next
    "Question (asked on ...)" turn, or re-emits "user:"/"assistant:" dialogue, and those
    continuations mention the gold string for unrelated reasons. Scoring the whole 128-token
    window therefore credits text that is not the answer at all. Same convention as
    ``babilong_score``, which cuts at "question" for exactly this reason.
    """
    span = prediction
    # The relay arms EMIT token 193 == b'\xc0' (a lone invalid UTF-8 byte, per
    # rwkv_vocab_v20230424.txt, so it decodes to U+FFFD) where every HF baseline emits
    # token 11 == "\n". Not a tokenizer-directory mismatch: the relay's -fla-v2 dir and the
    # baseline's RWKV7-Goose-World3-2.9B-HF have byte-identical (cmp) vocab/tokenizer files
    # and both decode 193 -> U+FFFD and 11 -> "\n". It is our arm's output distribution:
    # over these 50 rows champ-dynamic-30k emits 193 173x and 11 0x, while rwkv7-world3
    # emits 11 899x and 193 0x; 41/50 of our rows contain U+FFFD and 0/50 contain a newline,
    # versus 0/50 and 8-50/50 for the baselines. Every cut pattern below is newline-anchored,
    # so without this normalisation the span is NEVER cut for our arm alone and its
    # continuations get scored -- which silently inflated champ-dynamic-30k (e.g. gold
    # "United Airlines", answer "I flew with American Airlines the most", credited 1.0
    # because a later invented turn said "via United Airlines"). U+FFFD is never legitimate
    # answer content, so mapping it to a line break restores one scoring rule for all models.
    span = span.replace("�", "\n")
    # "\nAnswer" is a self-continuation marker, not answer content: models that keep going
    # after their reply invent a further "Question:/Answer:" turn. It never appears before
    # character 81 in any of the 10 result files (min 81 champ, 91 gla, 119 phi-2, 335 gemma),
    # so cutting there can never remove the model's own first answer.
    for cut in ("\nQuestion", "Question (asked", "\nuser:", "\nUser:",
                "\nassistant:", "\nAssistant:", "\nAnswer", "\n\n", "=== Session"):
        index = span.find(cut)
        if index != -1:
            span = span[:index]
    return span.strip()


def memeval_option_head(option: str) -> str:
    """The single most distinctive content word of an option, for a PARAPHRASED mention.

    ``memeval_alternatives`` returns the question's own wording, e.g. "one to Europe with
    family", but a model paraphrases it ("a trip to Europe with your family"). The literal
    option string then does not appear, only the gold option does, and the first-commitment
    rule credits a wrong answer 1.0 -- measured on global-dit-50k, gold "The solo trip to
    Thailand", span "You took a trip to Europe with your family first, and then you took a
    solo trip to Thailand."

    Returns ONE word, not a phrase: the paraphrase reorders and interpolates words ("a trip to
    Europe with YOUR family"), so any multi-word join is not a contiguous substring of the
    span and can never match. The longest content word ("europe", "thailand") is the reliable
    anchor. Stopwords are dropped so the anchor is never "with" or "trip".
    """
    stop = {"one", "to", "the", "a", "an", "my", "of", "with", "and", "or", "for", "in",
            "on", "at", "trip", "event", "task", "thing", "first", "time"}
    words = [word for word in normalize_freeform(option).split() if word not in stop]
    return max(words, key=len) if words else ""


def memeval_alternatives(question: str) -> list[str] | None:
    """The two options of a "Which X, A or B?" question, or None.

    LongMemEval's ``temporal-reasoning`` block -- which is *all* of the first 50 oracle rows,
    because the file is sorted by question_type -- is dominated by this shape, and for those
    the gold answer is a verbatim substring of the question itself (21/50 measured). A plain
    containment metric consequently scores the *question* 0.45 and a model that merely echoes
    it above every real model, so containment cannot be used unmodified on this suite.
    """
    match = re.search(r",\s*(?:the\s+)?(.+?)\s+or\s+(?:the\s+)?(.+?)\s*\?", question, re.I)
    return [match.group(1).strip(), match.group(2).strip()] if match else None


def memeval_find_phrase(haystack: str, needle: str) -> int:
    """Index of ``needle`` in ``haystack`` on WORD boundaries, or -1.

    Plain ``str.find`` is wrong here and silently inverts verdicts. Measured on phi-2's
    fixed run: for "Which vehicle did I take care of first, the bike or the car?" the answer
    "I took care of the bike first" contains "car" inside "ca(re)" at position 7, ahead of
    "bike" at 19, so a substring-based commitment rule credits the distractor and scores a
    correct answer 0.0. Both strings are already ``normalize_freeform``-ed (lowercased,
    punctuation stripped), so ``\\b`` is safe.
    """
    if not needle:
        return -1
    match = re.search(rf"\b{re.escape(needle)}\b", haystack)
    return match.start() if match else -1


def memeval_gold_variants(answer: str) -> list[str]:
    """Split a LongMemEval gold that is a RUBRIC, not an answer, into acceptable strings.

    9 of the first 50 oracle rows store the judge's rubric in the ``answer`` field, e.g.
    ``'7 days. 8 days (including the last day) is also acceptable.'`` and
    ``'One week. Answers ranging from 7 days to 10 days are also acceptable.'``. The official
    gpt-4o judge reads that prose and honours the alternatives; any containment rule instead
    asks whether the model emitted the whole sentence, which nothing ever does, so all 9 rows
    score 0.0 for every model no matter how right the answer is. That is a scorer bug worth
    ~0.18 of absolute accuracy, and it hides exactly the temporal-reasoning items this 50-row
    block consists of.

    Returns the primary answer first, then each additional acceptable form.
    """
    text = (answer or "").strip()
    if "acceptable" not in text.lower():
        return [text]
    match = re.match(r"^(.*?[.!?])\s+(.*)$", text, re.S)
    if not match:
        return [text]
    variants = [match.group(1).rstrip(".").strip()]
    rest = match.group(2)
    spread = re.search(r"ranging from\s+(\d+)\s*(\w+)?\s+to\s+(\d+)\s*(\w+)?", rest, re.I)
    if spread:
        low, high = int(spread.group(1)), int(spread.group(3))
        unit = (spread.group(4) or spread.group(2) or "").strip()
        if high >= low and high - low <= 60:          # sanity-bound the expansion
            variants += [f"{n} {unit}".strip() for n in range(low, high + 1)]
    else:
        alternative = re.match(r"\s*([\w\-]+(?:\s+\w+)?)", rest)
        if alternative:
            variants.append(alternative.group(1).strip())
    unique: list[str] = []
    for variant in variants:
        if variant and variant.lower() not in {seen.lower() for seen in unique}:
            unique.append(variant)
    return unique


def longmemeval_score_approx(prediction: str, answer: str, question: str = "") -> float:
    """APPROXIMATION of LongMemEval's LLM-judged accuracy. Not the official metric.

    Official (xiaowu0162/LongMemEval, ``src/evaluation/evaluate_qa.py``): a gpt-4o judge
    reads question/gold/response and returns yes/no with a question-type-specific prompt;
    accuracy is the fraction of "yes". No judge is reachable from a compute node, so:

      * Score only ``memeval_answer_span`` -- the first answer turn, not the whole window.
      * Accept any form the rubric-style gold licenses (``memeval_gold_variants``).
      * For a two-alternative question, credit the option the span commits to FIRST, and
        require it to be the gold one. "A happened first, B came later" commits to A even
        though both names appear; "A or B" and an echoed question commit to neither in a
        way that can be credited, which is the point.
      * Otherwise, containment of the normalised gold in the normalised span.

    All matching is on word boundaries (``memeval_find_phrase``); plain substring matching
    made "car" fire inside "care" and scored correct answers 0.0.

    Exact-match token-F1 partial credit was REMOVED (2026-09-21). It was pure noise on this
    suite: measured over the six existing runs it contributed 0.011-0.049 of each score
    while a null "echo the question" control earned 0.45 overall, i.e. the old metric ranked
    a model that reads nothing above every model that answers. Verified after this change:
    echo-question, question-repeated, empty string and phi-2's all-newline degenerate output
    all score 0.00; the residual floor is the ~0.21 that blind guessing earns on the
    binary-choice items, which is why 50 rows of this block cannot separate models.

    ``question_type`` is still ignored (the official judge prompt differs per type), and
    abstention (``_abs`` ids) is still unscorable -- though none of those land in the first
    50 rows.
    """
    if not answer:
        return 0.0
    span = normalize_freeform(memeval_answer_span(prediction))
    options = memeval_alternatives(question or "")
    for candidate in memeval_gold_variants(answer):
        gold = normalize_freeform(candidate)
        if not gold:
            continue
        if options:
            normalised = [normalize_freeform(option) for option in options]
            # Which listed option IS the gold one (gold may be a sub/superstring of it)?
            gold_option = next((option for option in normalised
                                if option and (option in gold or gold in option)), None)
            if gold_option is not None:
                # Locate each option, falling back to its distinctive content words so a
                # PARAPHRASED distractor still counts as mentioned. Without the fallback,
                # only the gold matches literally and a wrong answer is credited 1.0.
                positions = {}
                for option in normalised:
                    if not option:
                        continue
                    at = memeval_find_phrase(span, option)
                    if at == -1:
                        head = memeval_option_head(option)
                        at = memeval_find_phrase(span, head) if head else -1
                    if at != -1:
                        positions[option] = at
                if not positions:
                    return 0.0      # committed to neither option
                return float(min(positions, key=positions.get) == gold_option)
        if memeval_find_phrase(span, gold) != -1:
            return 1.0
    return 0.0


def load_memeval_rows(path: Path) -> list[dict[str, Any]]:
    """LongMemEval rows from the HF cache blob.

    Layout verified against the cached snapshot 2026-09-20: ``longmemeval_{s,m,oracle}`` are
    extensionless *files* (symlinks into ``blobs/``) holding a JSON list of 500 items with
    ``question_id / question_type / question / answer / question_date / haystack_dates /
    haystack_session_ids / haystack_sessions / answer_session_ids``. ``haystack_sessions`` is
    a list of sessions, each a list of ``{"role", "content"}`` messages (messages carry a
    ``has_answer`` flag we do not use).
    """
    if path.is_dir():
        candidates = [p for p in sorted(path.iterdir())
                      if p.is_file() and not p.name.startswith(".")]
        if not candidates:
            raise FileNotFoundError(f"no LongMemEval split inside {path}")
        path = candidates[0]
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:              # stream: longmemeval_m is ~2.7 GB
        items = json.load(handle)
    return [{
        "id": item.get("question_id", index),
        "prompt": build_memeval_prompt(item),
        "answers": [str(item.get("answer", ""))],
        # The scorer needs the bare question: on this suite the gold answer is often a
        # verbatim substring of it, so containment has to be made question-aware.
        "question": item.get("question", ""),
        "question_type": item.get("question_type"),
        "is_abstention": str(item.get("question_id", "")).endswith("_abs"),
    } for index, item in enumerate(items)]


def retrieval_score(prediction: str, ground_truth: str) -> float:
    match = re.search(r"Paragraph (\d+)", ground_truth)
    if not match:
        return 0.0
    numbers = re.findall(r"\d+", prediction)
    return 0.0 if not numbers else sum(x == match.group(1) for x in numbers) / len(numbers)


def load_longbench_rows(path: Path, dataset: str, prompts: dict[str, str]) -> list[dict[str, Any]]:
    """LongBench rows from ``<data_dir>/<dataset>.jsonl`` + the pinned prompt templates.

    The pinned jsonl stores ``answers`` as a *stringified* list (``"['Ada']"``), which the
    previous ``list(...)`` call turned into ``['[', "'", 'A', 'd', 'a', "'", ']']`` -- every
    generation then matched some single character and scored a small non-zero F1. ``as_list``
    parses it back into a list, and still accepts a real JSON list (what the unit test and a
    fresh LongBench download give).
    """
    template = prompts[dataset]
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        answers = as_list(item.get("answers"))
        if not answers:
            answers = as_list(item.get("answer"))
        rows.append({
            "id": item.get("_id", index),
            "prompt": template.format(**item),
            "answers": [str(answer) for answer in answers],
        })
    return rows


def load_babilong_rows(path: Path) -> list[dict[str, Any]]:
    items = json.loads(path.read_text())
    return [{
        "id": index,
        "prompt": f"{item['input'].rstrip()}\nQuestion: {item['question'].strip()}\nAnswer:",
        "question": item["question"].strip(),
        "answers": [str(item["target"])],
    } for index, item in enumerate(items)]


def load_ruler_rows(path: Path, task: str | None = None) -> list[dict[str, Any]]:
    """RULER rows from either JSONL or the parquet the HF cache actually ships.

    The cache layout is ``<length>/test-00000-of-00001.parquet`` holding every RULER task,
    with a ``task`` column -- not the per-task ``validation.jsonl`` this loader first assumed.
    When a parquet is given, filter it to ``task``; otherwise fall back to JSONL.
    """
    rows = []
    if path.suffix == ".parquet" or path.is_dir():
        import pandas as pd
        files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        for f in files:
            df = pd.read_parquet(f)
            if task and "task" in df.columns:
                df = df[df["task"] == task]
            for i, item in enumerate(df.to_dict("records")):
                # Real RULER columns (verified against the cached parquet 2026-09-20):
                # context / question / answer_prefix / answer / task / max_new_tokens.
                # The earlier version of this assumed input/outputs and would KeyError.
                # Official RULER prompts the model with context + question + answer_prefix
                # and continues; that is exactly what we concatenate here.
                prompt = f"{item.get('context', '')}\n{item.get('question', '')}\n" \
                         f"{item.get('answer_prefix', '')}"
                # `answer` arrives as a *stringified* list, e.g. "['8090293']" (or a numpy
                # array when pandas hands one back); `as_list` covers both.
                answers = [str(a) for a in as_list(item.get("answer"))] or [""]
                rows.append({"id": i, "prompt": prompt, "answers": answers,
                             "max_new_tokens": item.get("max_new_tokens")})
        return rows
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append({"id": item.get("index", index), "prompt": item["input"],
                     "answers": [str(a) for a in as_list(item.get("outputs"))]})
    return rows


def metric_for(suite: str, dataset: str):
    if dataset in SUMMARY_DATASETS:
        return rouge_l_f1
    if dataset == "passage_retrieval_en":
        return retrieval_score
    return qa_f1


def positional_window(model) -> int | None:
    """The model's hard positional ceiling in tokens, or None if it has no such ceiling.

    Why this exists: ``--max_input_tokens`` alone is not a safe truncation budget. The
    InfiniteBench run of 2026-09-20 passed ``--max_input_tokens 32768`` and fed every row
    whole at exactly 32768 tokens; phi-2's learned absolute position embeddings stop at
    2048, so it indexed 16x past the end of its own position table and emitted nothing but
    token 198 (newline) on 131 of 250 rows -- an exact 0.0 on all five datasets. The log
    said so out loud and was ignored: "Token indices sequence length is longer than the
    specified maximum sequence length for this model (90849 > 2048)".

    But ``min(cap, config.max_position_embeddings)`` is NOT a safe blanket rule either, and
    applying it as one would destroy evidence. Verified on this machine 2026-09-21 by
    grepping the modeling code that actually runs:

      * ``fla/models/gla/modeling_gla.py``: the only ``max_position_embeddings`` reference is
        inside ``GLABlock``'s optional hybrid-``Attention`` branch, guarded by
        ``config.attn is not None``. gla-1.3B-100B ships ``"attn": null``, so that branch is
        dead code and no rotary module is ever built. ``GLAForCausalLM``'s own body: 0 hits.
      * ``models/RWKV7-Goose-World3-2.9B-HF/modeling_rwkv7.py``                 -- 0 hits
      * ``transformers/models/mamba/modeling_mamba.py``                         -- 0 hits

    for every one of ``position_ids`` / ``max_position_embeddings`` / ``position_embed`` /
    ``rotary``. These are recurrent architectures: they carry a fixed-size state forward and
    never index a position table, so the ``max_position_embeddings: 2048`` that fla-hub's
    gla-1.3B config.json happens to declare is an inert leftover field. Truncating GLA to
    2048 would throw away 94% of the context for no reason and manufacture a false ceiling.

    So: trust the declared window by DEFAULT, and drop it only for architectures on the
    explicit recurrent list below. The first version of this inferred "recurrent" from a
    source grep over ``type(model)``, which is fail-dangerous and was wrong twice over: the
    grep answers "no positional machinery" for anything whose own class body happens not to
    mention positions -- a thin wrapper, a stub, a subclass that inherits its forward -- and
    so would silently hand back "no limit" for a genuinely positional model, reproducing the
    exact phi-2 newline collapse this function exists to prevent. Guessing "limited" on a
    recurrent model only costs context and the audit reports it; guessing "unlimited" on a
    positional model destroys the run. An unrecognised recurrent architecture is therefore
    fixed by adding its ``model_type`` here, not by making the detection cleverer.
    """
    module = getattr(model, "rwkv_model", model)     # relay checkpoints wrap the decoder
    config = getattr(module, "config", None)
    if config is None:
        return None
    # Known-recurrent model_types. Each carries a fixed-size state forward and never indexes
    # a position table, so no config field about positions can apply to it. Verified from the
    # modeling source for the four this paper runs (rwkv7, gla, mamba, and the relay arms'
    # frozen RWKV-7 decoder); the rest of the fla family is listed on the same grounds.
    RECURRENT_MODEL_TYPES = {"rwkv6", "rwkv7", "gla", "gated_deltanet", "mamba", "mamba2",
                             "rwkv", "retnet", "hgrn", "hgrn2", "delta_net", "abc",
                             "linear_attn", "based", "rebased", "lightnet", "samba"}
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type in RECURRENT_MODEL_TYPES:
        return None
    window = (getattr(config, "max_position_embeddings", None)
              or getattr(config, "n_positions", None)
              # openlm (DCLM) spells its context length `seq_len` and declares NEITHER of the
              # two names above. Without this clause the function returns None for DCLM-1B --
              # i.e. the "no positional ceiling" verdict meant for recurrent models -- and the
              # harness would feed 16k/32k-token prompts to a 2048-trained rotary+qk-norm
              # attention stack. That is exactly the phi-2 failure this function exists to
              # prevent (0.0 on all five InfiniteBench datasets), and it would fail OPEN and
              # SILENTLY here, because open_lm's RotaryEmbedding._update_cos_sin_tables
              # REBUILDS its cos/sin cache for any requested length
              # (positional_embedding/rotary.py: `self._update_cos_sin_tables(k.shape[1] +
              # offset, ...)`). Unlike phi-2's learned position table there is no index error
              # and no warning -- the model just extrapolates RoPE far past its trained
              # length and returns confident garbage. Verified 2026-09-21 against
              # TRI-ML/DCLM-1B rev f43c272: config.json has seq_len 2048, and
              # max_position_embeddings / n_positions are both absent.
              or getattr(config, "seq_len", None))
    return int(window) if window else None


def effective_input_budget(model, max_input_tokens: int, max_new_tokens: int) -> tuple[int, dict]:
    """``max_input_tokens`` clamped to the model's real positional ceiling, plus an audit.

    The generated tokens occupy positions too, so the prompt has to leave room for them:
    phi-2 at a 2048-token prompt with 128 new tokens still walks off the end at position
    2048. Reserve ``max_new_tokens``.

    Returns (budget, audit) and the audit goes into the output JSON ``config`` so a reader
    can tell a genuine capability result from a truncation artefact. A baseline that scores
    badly only because we truncated its input is our artefact, not its limitation, and the
    number is not interpretable without the budget printed next to it.
    """
    window = positional_window(model)
    audit = {"requested_max_input_tokens": int(max_input_tokens),
             "positional_window": window,
             "reserved_for_generation": int(max_new_tokens)}
    if window is None:
        audit["effective_input_budget"] = int(max_input_tokens)
        audit["truncation_reason"] = "cli_cap_only (recurrent model: no positional ceiling)"
        return int(max_input_tokens), audit
    budget = min(int(max_input_tokens), max(2, window - int(max_new_tokens)))
    audit["effective_input_budget"] = budget
    audit["truncation_reason"] = ("positional_window" if budget < int(max_input_tokens)
                                 else "cli_cap_only")
    return budget, audit


def truncate_ids(tokenizer, prompt: str, max_input_tokens: int, device: str,
                 stats: dict | None = None, add_bos: bool = False) -> torch.Tensor:
    """Middle-truncate to ``max_input_tokens`` -- head + tail, drop the middle.

    The shape is LongBench's official protocol (`pred.py` keeps `max_length//2` from each
    end) and is deliberately unchanged here; only the *budget* handed in is now per-model
    (see ``effective_input_budget``).

    ``stats``, if given, receives the *pre*-truncation length. Without it a row records only
    what the model saw, which cannot distinguish "answered a 2k prompt badly" from "was shown
    2k of a 136k document" -- the difference between a capability result and our own artefact.
    InfiniteBench prompts run ~30k-170k tokens, so for most models this ratio is the headline.

    ``add_bos`` prepends the tokenizer's BOS id AFTER truncation. ``add_special_tokens=False``
    below is required -- the template already contains the task framing and letting the
    tokenizer inject specials mid-document would corrupt the middle-truncation offsets -- but
    it also suppresses BOS for the two models whose tokenizer_config sets
    ``add_bos_token: true`` (gemma-2-2b ``<bos>``, gla-1.3B ``<s>``). Gemma in particular is
    trained to require it: measured on bAbILong qa1/4k, 13% without BOS vs **53%** with it
    (``results/benchmarks/falsify_zero/gemma-bos-only.json``). Prepending after truncation,
    not before, keeps the head/tail split byte-identical to the no-BOS case so the only
    difference is the one token.
    """
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full = int(ids.numel())
    # The BOS counts against the budget: it occupies a position like any other token. Taking
    # it out of `room` rather than adding it on top keeps `len(prefix_ids) <= budget` exact.
    # Off by one matters here -- gemma-2-2b's budget is window(8192) - max_new(512) = 7680, so
    # 7680 document tokens + BOS + 512 generated = 8193 would overrun the position table on
    # the final step and re-trigger the very overflow this budget exists to prevent.
    room = max_input_tokens - (1 if add_bos else 0)
    if ids.numel() > room:
        left = room // 2
        ids = torch.cat([ids[:left], ids[-(room - left):]])
    if ids.numel() < 2:
        raise ValueError("Prompt must contain at least two tokens")
    if add_bos:
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is not None and int(ids[0]) != int(bos):
            ids = torch.cat([torch.tensor([int(bos)], dtype=ids.dtype), ids])
    if stats is not None:
        stats["prompt_tokens_untruncated"] = full
        stats["dropped_tokens"] = full - int(ids.numel())
    return ids.unsqueeze(0).to(device)


def tokenizer_wants_bos(tokenizer) -> bool:
    """True if this tokenizer normally starts a sequence with BOS.

    Decided by *probing the tokenizer*, not by reading ``add_bos_token``. Reading the flag
    looks equivalent and is not: measured 2026-09-21 on this machine's own snapshots, the
    loaded object reports ``add_bos_token=False`` and carries no ``init_kwargs`` even when
    its ``tokenizer_config.json`` literally contains ``"add_bos_token": true`` --

        gemma-2-2b   (GemmaTokenizer)    config true  -> attr False, init_kwargs absent
        gla-1.3B     (LlamaTokenizer)    config true  -> attr False, init_kwargs absent
        llama-3.2-3B (TokenizersBackend) no such key  -> attr False, but DOES prepend 128000

    so the flag-reading version returned False for all three models it existed to catch and
    the BOS prepend was silently inert -- the failure mode where a fix lands, compiles, runs,
    and changes nothing. llama-3.2 also shows the flag is not even the right question: it
    prepends BOS from its template with no ``add_bos_token`` key at all.

    The probe compares the two tokenizations of a short string, which is by construction the
    thing the caller needs to undo: exactly the token(s) ``add_special_tokens=False`` drops at
    the front. Guarded because a tokenizer may raise on odd input; on failure, fall back to
    the old flag read rather than silently answering False.
    """
    if getattr(tokenizer, "bos_token_id", None) is None:
        return False
    try:
        with_special = tokenizer("x", add_special_tokens=True).input_ids
        without = tokenizer("x", add_special_tokens=False).input_ids
        if with_special and without and with_special[0] != without[0]:
            return with_special[0] == tokenizer.bos_token_id
        return False
    except Exception as error:
        print(f"[load] BOS probe failed ({error!r}); falling back to add_bos_token flag",
              flush=True)
        flag = getattr(tokenizer, "add_bos_token", None)
        if flag is None:
            flag = (getattr(tokenizer, "init_kwargs", None) or {}).get("add_bos_token")
        return bool(flag)


def patch_fla_recurrent_generate(model) -> bool:
    """Make HF ``generate`` correct on fla's recurrent caches. Returns True if patched.

    fla 0.5.0 + transformers 5.3.0 are mutually incompatible in a way that silently corrupts
    every ``model.generate()`` result for rwkv7 / gla / gated_deltanet:

      1. ``fla/models/utils.py`` probes ``hasattr(self, '_cache_dependant_input_preparation')``,
         a method transformers 5.3.0 DELETED, so it falls back to its own hand-written slicer.
      2. That fallback guards on ``input_ids.shape[1] != cache_position.shape[0]``, but
         transformers now *accumulates* ``cache_position`` (``torch.cat``), so the two lengths
         stay equal forever, the guard never fires, and the slice never happens.

    Net effect: every decode step re-feeds the ENTIRE sequence into a recurrent state that has
    already advanced past it. ``cache_seq_len`` grows quadratically (0->33->67->102->138) and
    the generation degenerates into verbatim prompt recitation. Measured token agreement
    against a correct hand-written greedy loop: 6/1200 = 0.5%. Fingerprint: the first token is
    100% identical (prefill is shared and correct) and it collapses from the second onward.

    NOT affected, and deliberately not patched: rwkv6 (``modeling_rwkv6.py`` ships its own
    correct ``prepare_inputs_for_generation``), mamba (not fla), and ordinary transformers.
    The guard below is the reason -- it only patches when fla's own mixin is in the MRO.

    Patch verified 2026-09-21 (logs/qz/verify-fla-generate-fix-v2.log): generate-vs-loop
    agreement 0/200 -> 132/240 (55.0%); the residual is an EOS asymmetry, not this bug.
    """
    cls = type(model)
    # Only fla's generation mixin is broken. Anything else either has no recurrent cache or
    # already prepares its inputs correctly, and patching it would be an unforced change.
    if not any("fla" in base.__module__ for base in cls.__mro__
               if getattr(base, "__module__", None)):
        return False
    original = cls.prepare_inputs_for_generation
    if getattr(original, "_fla_recurrent_patched", False):
        return True
    import functools

    # The **kwargs parameter MUST be named exactly `kwargs`: transformers'
    # _validate_model_kwargs does `if "kwargs" in model_args: model_args |=
    # signature(self.forward)`, so naming it `kw` makes generate reject attention_mask with
    # "The following model_kwargs are not used by the model". functools.wraps keeps the
    # signature that transformers introspects.
    @functools.wraps(original)
    def fixed(self, input_ids=None, past_key_values=None, **kwargs):
        model_inputs = original(self, input_ids=input_ids,
                                past_key_values=past_key_values, **kwargs)
        if past_key_values is not None:
            try:
                seen = int(past_key_values.get_seq_length())
            except Exception:
                seen = 0
            if (seen > 0 and model_inputs.get("input_ids") is not None
                    and model_inputs["input_ids"].shape[1] > 1):
                model_inputs["input_ids"] = model_inputs["input_ids"][:, -1:].contiguous()
                model_inputs["cache_position"] = None
        return model_inputs

    fixed._fla_recurrent_patched = True
    cls.prepare_inputs_for_generation = fixed
    return True


def resolve_stop_ids(model, tokenizer) -> list[int]:
    """Every id that should end generation, unioned over the three places models declare it.

    2026-09-21: ``greedy_loop`` was called with ``eos_id=tokenizer.eos_token_id`` alone. For
    RWKV7-Goose-World3 that resolves to **65530**, an id that never appears in any generation,
    because the tokenizer's ``eos_token`` is the *string* ``'\\n\\n'``; the real end-of-text is
    id **0** (``<|rwkv_tokenizer_end_of_text|>``), declared in ``generation_config.json``, while
    ``config.eos_token_id`` is a third value (2 = ``'\\x01'``). So the loop's stop check could never
    fire and every row ran to its full budget: id 0 appears *inside* 141/450 published longbench
    generations, 671/1800 ruler, 12/400 bAbILong, 4/250 InfiniteBench.

    Scoring impact is metric-shaped, which is why it hid: the containment suites are unchanged at
    RULER 0/1800, bAbILong 0/400, InfiniteBench 0/250 rows, because trailing junk after the answer
    cannot remove a match. LongBench is F1/ROUGE-F1, where the same junk costs precision --
    honouring the stop id moves macro 0.0955 -> 0.1197 (+0.0242), 82 rows changing, 67 up / 15 down.

    Do not over-generalise that to "containment metrics are immune". Junk cannot LOWER such a
    score, but it CAN RAISE one whenever the metric is fractional over a gold set (a partial-coverage
    sum), because the extra text may match aliases the true answer alone does not -- on such a
    metric the defect flatters whoever has it. RULER happens to escape on both counts: its
    fractional tasks (cwe/fwe/vt) have single-word golds that are simply present or absent, and its
    one multi-alias task (qa_1) is max-scored via ``RULER_MATCH_PART_TASKS``. So verify by rescoring
    and counting rows, never by reasoning from the metric family.

    ⚠️ If you rescore RULER offline, derive the task with ``dataset.split("_", 1)[1]``: the stored
    ``dataset`` is ``4096_qa_1`` (``{length}_{task}``, no ``/``), so ``split("/")[-1]`` returns the
    whole string, misses ``RULER_MATCH_PART_TASKS`` and silently takes the ``string_match_all``
    branch. That mistake made qa_1 read 0.3333 instead of 1.0000 and produced a phantom "RULER
    changed 1 row" result. Start every rescore with a control pass over the UNMODIFIED text
    asserting it reproduces the stored score (it fails 26/1800 under the bad parse, 0/1800 under
    the right one).

    Union rather than first-hit: a wrong extra id is harmless (it never appears), whereas a
    missing one silently disables stopping, and these three sources disagree in practice.

    ⚠️ DO NOT "clean up" this union down to ``generation_config``'s id 0, however inert the
    others look. 2026-09-21: id 65530 appears inside 0 stored generations of any cell, which
    reads like proof it is never emitted -- but ``generate`` and ``greedy_loop`` both ``break``
    BEFORE appending, so a stop id that *fires* leaves no trace in the stored ids. "0 rows
    contain it" is therefore equally consistent with "never emitted" and "always fired". For
    the champ-dynamic arm it is the latter: 65530 (= ``'\\n\\n'``) is that checkpoint's ONLY
    working stop token, and it is what ends its 32/450 short rows (all of which end on sentence
    punctuation, at 30 distinct lengths, so no budget cap explains them). Replacing the union
    with id 0 would delete that arm's stop condition and move its published score.

    ``generation_config`` is probed FIRST because it is the only one of the three that was
    correct for every model checked, so ``ids[0]`` is a usable single-id fallback for any
    caller that cannot take a set. The relay runtimes take the full set, so that path is no
    longer load-bearing -- but the ordering is still the difference between a working and a
    dead stop check, so do not reorder these.

    ``model`` here may be a LACES/Global relay wrapper rather than the decoder itself. The
    wrapper carries no ``generation_config``, so probe ``model.rwkv_model`` too -- the same
    ``getattr(model, "rwkv_model", model)`` idiom used at the audit site above. Without this the
    relay arms resolve only the tokenizer's value, which is exactly the broken case: measured on
    the published cells, our own ``global-dit-50k`` ran 450/450 LongBench rows to full budget
    with id 0 sitting inside 79 of them, worth macro 0.0955 -> 0.1198 (+0.0243).
    """
    ids: list[int] = []
    inner = getattr(model, "rwkv_model", None)
    for source in (getattr(model, "generation_config", None),
                   getattr(inner, "generation_config", None),
                   getattr(model, "config", None),
                   getattr(inner, "config", None),
                   tokenizer):
        value = getattr(source, "eos_token_id", None)
        for one in (value if isinstance(value, (list, tuple)) else [value]):
            if isinstance(one, int) and one not in ids:
                ids.append(one)
    return ids


def greedy_loop(model, prefix_ids: torch.Tensor, max_new_tokens: int,
                eos_id: int | None = None, eos_ids: list[int] | None = None) -> list[int]:
    """Reference greedy decode: prefill once, then feed exactly one token per step.

    Why this exists alongside ``patch_fla_recurrent_generate``: the patch is necessary but
    not sufficient. Measured on rwkv7-world3 (20 bAbILong qa1/4k prompts,
    ``results/benchmarks/falsify_zero/rwkv7-fingerprint.json``), token agreement with this
    loop is 2/20 at every prefix length 1..32 whether generate is patched or not, and the
    task score is unpatched 0% / patched 5% / loop 40%. gla-1.3B is the same story, less
    severe: patched-vs-loop 14/20 at prefix 32, score 5% / 20% / 20%. So for fla recurrent
    models any ``model.generate`` number -- patched included -- is a harness artefact.

    This is also the exact shape ``LACESRuntime._stream`` / ``GlobalRuntime.generate`` use
    for our own arms, so a baseline decoded here is protocol-matched to ours rather than
    being handicapped by a decode path ours never takes.

    Two cache conventions have to be handled. Most models thread state through
    ``past_key_values``; native-transformers Mamba uses ``cache_params`` instead and its output
    has no ``past_key_values`` attribute at all (``AttributeError: 'MambaCausalLMOutput' object
    has no attribute 'past_key_values'``, hit 2026-09-21). Passing the state under the wrong
    keyword would be worse than crashing: the model would silently re-prefill from scratch each
    step, i.e. exactly the bug this function exists to avoid, and the run would look fine.
    """
    out = model(input_ids=prefix_ids, use_cache=True, return_dict=True)
    # Ask the prefill output which convention this model uses, rather than inspecting the
    # forward signature -- the output is ground truth about where the state actually came back.
    kwarg = "cache_params" if getattr(out, "cache_params", None) is not None else "past_key_values"
    cache, logits = getattr(out, kwarg), out.logits[0, -1]
    if cache is None:
        raise RuntimeError(
            f"greedy_loop: {type(model).__name__} returned no {kwarg}; without a cache every "
            f"step would re-read the whole prefix, which is the defect this decode path exists "
            f"to avoid. Check the model's cache keyword before using --decode loop."
        )
    # Mamba additionally REFUSES to infer `cache_position` once a cache is passed in
    # ("You have to specify the `cache_position` manually when `use_cache=True` and
    # `cache_params` is passed", hit 2026-09-21 after the cache_params fix landed). It is not an
    # absolute sequence index: `MambaCache.update_conv_state` clamps it to
    # [0, conv_kernel-1] and uses it to address a rolling conv buffer, and
    # `cache_position.shape[0] == conv_kernel` is the sentinel that selects the PREFILL branch.
    # So the decode step must pass a length-1 tensor, or a 2048-token prefill would be re-run
    # as a prefill on every step. Values are mirrored from Mamba's own
    # `_update_model_kwargs_for_generation` (prefill arange(0, conv_kernel), then last+1 each
    # step) so this loop differs from `generate` ONLY in feeding one token at a time -- which is
    # the entire point of the comparison it exists to support.
    step_kwargs: dict = {}
    if kwarg == "cache_params":
        conv_kernel = int(getattr(model.config, "conv_kernel", 4))
        step_kwargs["cache_position"] = torch.tensor([conv_kernel], device=prefix_ids.device,
                                                    dtype=torch.long)
    ids: list[int] = []
    stop = set(eos_ids or ([] if eos_id is None else [eos_id]))
    for index in range(max_new_tokens):
        token = int(logits.argmax())
        if token in stop:
            break
        ids.append(token)
        if index + 1 < max_new_tokens:
            out = model(input_ids=torch.tensor([[token]], device=prefix_ids.device,
                                               dtype=torch.long),
                        use_cache=True, return_dict=True, **{kwarg: cache}, **step_kwargs)
            cache, logits = getattr(out, kwarg), out.logits[0, -1]
            if "cache_position" in step_kwargs:
                step_kwargs["cache_position"] = step_kwargs["cache_position"] + 1
    return ids


def ensure_model_type_resolvable(model_dir: str) -> None:
    """Import ``fla.models`` only when transformers cannot resolve the checkpoint's model_type.

    fla-hub snapshots (e.g. ``gla-1.3B-100B``) declare ``model_type: gla`` and ship **no**
    remote-code file, so ``AutoModelForCausalLM.from_pretrained`` fails outright until the
    fla package registers the architecture. Importing fla unconditionally is wrong, though:
    ``fla.models`` ALSO re-registers the standard ``mamba`` model_type and so hijacks
    mamba-2.8b into fla's own Mamba, whose slow cached-decoding path rejects the multi-token
    prefill this harness uses ("Mamba cached decoding only supports a single new token per
    step."). transformers ships a native ``mamba``, so only reach for fla when the
    checkpoint's model_type is unknown to transformers.

    Reads ``config.json`` via ``Path`` on purpose: this module does not import ``os``, and
    an ``os.path.join`` here would raise NameError inside a swallowed except (the exact
    dormant bug that silently pinned another evaluator to bfloat16).
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    model_type, has_auto_map = None, False
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
        model_type = config.get("model_type")
        has_auto_map = bool((config.get("auto_map") or {}).get("AutoModelForCausalLM"))
    except Exception as error:
        # Loud, not silent: a failed probe must not look like a successful one.
        print(f"[load] model_type probe failed for {model_dir} ({error!r}); "
              f"assuming transformers can resolve the checkpoint", flush=True)
        return
    if model_type and model_type in CONFIG_MAPPING:
        print(f"[load] model_type={model_type!r} is native to transformers; fla not imported",
              flush=True)
        return
    if has_auto_map:
        # e.g. RWKV7-Goose-World3-2.9B-HF: unknown to transformers, but the snapshot ships
        # modeling_rwkv7.py and declares auto_map, so trust_remote_code loads it. Importing
        # fla here would re-register the same model_type and could shadow that remote class,
        # so leave an already-working path alone.
        print(f"[load] model_type={model_type!r} is not native, but the snapshot declares "
              f"auto_map; using its remote code and not importing fla", flush=True)
        return
    if model_type == "openlm":
        # DCLM (TRI-ML/DCLM-1B, apple/DCLM-7B) ships model_type "openlm" with
        # architectures ["OpenLMModel"], no auto_map and no modeling file, so neither
        # transformers nor trust_remote_code can resolve it. mlfoundations' `open_lm`
        # package provides the class and self-registers it with AutoConfig /
        # AutoModelForCausalLM on import of ``open_lm.hf`` (see its
        # configuration_openlm.py / modeling_openlm.py module tails). `fla.models` knows
        # nothing about openlm, so the fla import below would not help and its blanket
        # re-registration of `mamba` is a documented hazard -- return before it.
        import open_lm.hf  # noqa: F401  -- registers OpenLMConfig / OpenLMForCausalLM
        print(f"[load] model_type={model_type!r}: imported open_lm.hf to register "
              f"OpenLMForCausalLM with the Auto classes", flush=True)
        return
    import fla.models  # noqa: F401  -- registers gla / rwkv6 / rwkv7 with the Auto classes
    print(f"[load] model_type={model_type!r} is not in transformers.CONFIG_MAPPING and the "
          f"snapshot has no auto_map; imported fla.models to register it", flush=True)


class GlobalRuntime:
    """The ``prefix_cfg`` (single-z) arm, for ``trajectory_mode=False`` relay checkpoints.

    Why this class exists: ``LACESRuntime`` is the *trajectory* arm, and
    ``validate_pretrained_laces`` hard-refuses any checkpoint without a ``trajectory_dit`` --
    which is exactly the global-writer checkpoint (global-dit-50k, ``trajectory_mode=False``).
    So ``--arm dynamic`` cannot produce that row at all; it dies with
    "ValueError: checkpoint step=50000 != required step=30000" or, past the step check,
    "Pretrained trajectory S2 is missing" (both reproduced in logs/qz/lc-smoke-globaldit.log).

    ``run_baseline_evals`` already pairs each relay model with its own arm and records why:
    champ-dynamic-30k -> ``traj_cfg``, global-dit-50k -> ``prefix_cfg``, "running either
    through the other arm yields garbage -- measured: champ via prefix_cfg scored 1.00% with
    24% degenerate output while its own arm scored 24.67%". This class is that ``prefix_cfg``
    arm, so the long-context cell uses the same protocol as the model's short-context cell.

    The two stages are imported rather than re-derived: ``encode_prefix`` (S0: pooled hidden
    -> cond) and ``sample_ddim_cfg`` (S2: ``plan_steps``-step DDIM with zero-condition CFG)
    from ``sample_prefix_suffix_cfg``. The zero null-condition is the checkpoint's own
    convention: ``forward_prefix_suffix_diffusion`` drops the condition with ``cond * keep``,
    i.e. to 0, not to a learned null embedding.
    """

    def __init__(self, model, checkpoint, *, expected_step=None, plan_steps=100,
                 cfg_scale=1.5, protocol="aligned"):
        if expected_step is not None and checkpoint.get("step") != expected_step:
            raise ValueError(
                f"checkpoint step={checkpoint.get('step')} != required step={expected_step}")
        if getattr(model, "trajectory_enabled", False):
            raise ValueError(
                "GlobalRuntime is the single-z arm but this checkpoint is trajectory_mode=True; "
                "use --arm dynamic, which is the chunked protocol that checkpoint was trained with")
        if getattr(model, "latent_dit", None) is None:
            raise ValueError("Pretrained global S2 (latent_dit) is missing")
        if protocol not in ("aligned", "legacy"):
            raise ValueError("protocol must be aligned or legacy")
        self.model = model.eval().requires_grad_(False)
        self.protocol = protocol
        self.plan_steps = int(plan_steps)
        self.cfg_scale = float(cfg_scale)
        # Same dtype source as LACESRuntime: the denoiser the sampler actually calls.
        self.dtype = next(model.latent_dit.parameters()).dtype
        self.step = checkpoint.get("step")
        self.audit = {
            "step": self.step, "writer_type": model.s1_writer_type, "arm": "global",
            "trajectory_mode": bool(getattr(model, "trajectory_enabled", False)),
            "latent_dim": int(model.latent_dim), "plan_steps": self.plan_steps,
            "cfg_scale": self.cfg_scale,
        }

    def close(self):
        return None

    @torch.no_grad()
    def prepare(self, prefix_ids, *, seed):
        """S0 encode + S2 DDIM sample of the single global z. No labels enter this function."""
        from scripts.eval.sample_prefix_suffix_cfg import encode_prefix, sample_ddim_cfg
        if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1 or prefix_ids.shape[1] < 2:
            raise ValueError("Expected a single prefix with at least two tokens")
        devices = [prefix_ids.device.index or 0] if prefix_ids.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            cond = encode_prefix(self.model, prefix_ids, torch.ones_like(prefix_ids))
            z = sample_ddim_cfg(self.model, cond, self.plan_steps, self.cfg_scale,
                                prefix_ids.device, self.dtype)
        if not torch.isfinite(z).all():
            raise ValueError("S2 produced nonfinite latents")
        return z.detach().float(), None, cond.detach().float()

    @torch.no_grad()
    def generate(self, prefix_ids, z, *, max_new_tokens, raw=False, temperature=0.,
                 top_k=0, top_p=1., repetition_penalty=1., eos_id=None, eos_ids=None):
        """Greedy (or sampled) generation under one injected global state, aligned boundary.

        Mirrors ``sample_prefix_suffix_cfg.generate``: inject, then re-run the whole prefix
        against the injected cache. The re-run is load-bearing -- injection changes every
        position after the write, so the logits predicting the FIRST answer token have to be
        recomputed, which is the "aligned" protocol LACESRuntime documents on the other arm.
        """
        from scripts.eval.sample_prefix_suffix_cfg import apply_repetition_penalty, apply_top_p
        if raw:
            raise ValueError("the global arm has no raw mode; --arm raw belongs to the "
                             "trajectory runtime's no-plan control")
        if z.ndim != 2 or z.shape[0] != 1 or z.shape[-1] != self.model.latent_dim:
            raise ValueError("Expected a single global latent [1, latent_dim]")
        states = self.model.predict_states(z.to(self.dtype))
        stop = set(eos_ids or ([] if eos_id is None else [eos_id]))
        out = self.model.rwkv_model(input_ids=prefix_ids,
                                    attention_mask=torch.ones_like(prefix_ids).bool(),
                                    use_cache=True, return_dict=True)
        cache = self.model.inject_into_cache(out.past_key_values, states)
        out = self.model.rwkv_model(input_ids=prefix_ids, past_key_values=cache,
                                    use_cache=True, return_dict=True)
        cache, logits = out.past_key_values, out.logits[0, -1]
        ids, logps = [], []
        history = prefix_ids[0].tolist()
        for index in range(max_new_tokens):
            sample_logits = apply_repetition_penalty(logits.float(), history, repetition_penalty)
            if temperature <= 0:
                token = int(sample_logits.argmax())
            else:
                probabilities = torch.softmax(sample_logits / temperature, -1)
                if top_k > 0:
                    vals, indices = torch.topk(probabilities, min(top_k, probabilities.numel()))
                    probabilities = torch.zeros_like(probabilities).scatter(-1, indices, vals)
                    probabilities /= probabilities.sum()
                probabilities = apply_top_p(probabilities, top_p)
                token = int(torch.multinomial(probabilities, 1))
            if token in stop:
                break
            ids.append(token)
            history.append(token)
            logps.append(float(torch.nn.functional.log_softmax(logits.float(), -1)[token]))
            if index + 1 < max_new_tokens:
                out = self.model.rwkv_model(
                    input_ids=torch.tensor([[token]], device=prefix_ids.device, dtype=torch.long),
                    past_key_values=cache, use_cache=True, return_dict=True)
                cache, logits = out.past_key_values, out.logits[0, -1]
        return ids, logps


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",
                        choices=("longbench", "babilong", "ruler", "infinitebench", "memeval"),
                        required=True)
    parser.add_argument("--datasets", required=True,
                        help="Comma-separated dataset names or relative paths. Per suite: "
                             "longbench '<name>' (e.g. hotpotqa); babilong '<qa>/<len>' "
                             "(e.g. qa1/16k); ruler '<len>/<task>' (e.g. 4096/niah_single_1); "
                             "infinitebench '<task file>' (e.g. longbook_qa_eng); memeval "
                             "'<split>' (e.g. longmemeval_s)")
    parser.add_argument("--data_dir", required=True,
                        help="Suite root. longbench .../extract/data; babilong .../data; "
                             "ruler .../snapshots/<rev>; infinitebench .../snapshots/<rev>; "
                             "memeval .../snapshots/<rev>")
    # The shipped default (/root/benchmarks/...) does not exist on this machine; the pinned
    # copies are checked in locally.
    parser.add_argument("--longbench_config_dir",
                        default="/inspire/hdd/global_user/zhangjiaquan-253108540222/longbench_local/pinned_spec")
    parser.add_argument("--ckpt_dir", default=None, help="relay checkpoint (our models)")
    parser.add_argument("--model_dir", default=None,
                        help="plain HF checkpoint dir (baselines). Exactly one of --ckpt_dir/--model_dir.")
    parser.add_argument("--arm", choices=("dynamic", "raw", "global"), default="raw",
                        help="relay arm: 'dynamic' = chunked trajectory (LACESRuntime, needs "
                             "trajectory_mode=True); 'global' = single-z prefix_cfg "
                             "(GlobalRuntime, needs trajectory_mode=False); 'raw' = the "
                             "trajectory runtime with an all-zero plan (control)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--sample_stride", type=int, default=1,
                        help="Take every Nth row instead of the first N. REQUIRED for "
                             "InfiniteBench passkey/number_string, whose 590 rows are SORTED "
                             "by needle depth (59 depth bins x 10 rows, verified: each bin has "
                             "one identical depth and the bins strictly increase). rows[:50] is "
                             "therefore not a sample of the task -- it is the 5 shallowest bins, "
                             "needle depth 0.000-0.069, i.e. the easiest 8.5%% of the benchmark. "
                             "Stride 10 walks all 59 depths and makes the cell a depth curve.")
    parser.add_argument("--max_input_tokens", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--plan_steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--blend", type=float, default=0.7)
    parser.add_argument("--diffusion_sampler", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--expected_step", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode", choices=("hf", "loop"), default="hf",
                        help="baseline decode path (--model_dir only). 'hf' = "
                             "model.generate(); 'loop' = prefill once then feed exactly one "
                             "token per step. For fla recurrent models (rwkv7/gla) 'hf' is "
                             "measurably wrong even WITH patch_fla_recurrent_generate -- "
                             "measured token agreement against the loop is 2/20 at prefix 32 "
                             "(results/benchmarks/falsify_zero/rwkv7-fingerprint.json) -- so "
                             "their 'hf' numbers are a harness artefact, not a capability. "
                             "'loop' is also the shape our own relay arms decode in, so it is "
                             "the protocol-matched comparison.")
    return parser.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    if bool(args.ckpt_dir) == bool(args.model_dir):
        raise SystemExit("give exactly one of --ckpt_dir (relay) or --model_dir (HF baseline)")
    if args.model_dir and args.arm in ("dynamic", "global"):
        # A plain HF model has no latent plan to write, so the row would be generated raw
        # and *labelled* dynamic -- a silent misreport of what was run.
        raise SystemExit(f"--arm {args.arm} needs --ckpt_dir; a plain HF model only supports --arm raw")

    data_dir = Path(args.data_dir)
    prompt_map = {}
    max_gen_map = {}
    if args.suite == "longbench":
        config_dir = Path(args.longbench_config_dir)
        prompt_map = json.loads((config_dir / "dataset2prompt.json").read_text())
        max_gen_map = json.loads((config_dir / "dataset2maxlen.json").read_text())

    runtime = None
    checkpoint = {}
    # Bound here, not only inside the --model_dir branch: the result dict reads it for
    # `decode_is_trustworthy`, and the relay branch never calls the fla patch, so leaving it
    # unbound would raise NameError at write time -- after the whole run had been spent.
    patched = False
    if args.model_dir:
        # Plain HF baseline: no plan, no state injection -- just the decoder answering.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ensure_model_type_resolvable(args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True,
                                                  local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
            local_files_only=True).to(args.device).eval()
        # rwkv7 / gla / gated_deltanet reach ``generate`` through fla's own mixin, which is
        # broken against transformers 5.3.0 (see patch_fla_recurrent_generate). Unpatched,
        # every generated row for those models is invalid.
        patched = patch_fla_recurrent_generate(model)
        print(f"[load] fla recurrent-generate patch applied: {patched}", flush=True)
    else:
        from scripts.eval.relay_utils import load_relay_model
        model, _rwkv, tokenizer, checkpoint, _cfg = load_relay_model(args.ckpt_dir, args.device)
        if args.arm == "global":
            # Single-z arm: LACESRuntime would refuse this checkpoint outright (no
            # trajectory_dit), and forcing it through the trajectory sampler is the
            # 1.00%-vs-24.67% mismatch run_baseline_evals measured.
            runtime = GlobalRuntime(
                model, checkpoint, expected_step=args.expected_step,
                plan_steps=args.plan_steps, cfg_scale=args.cfg_scale, protocol="aligned",
            )
        else:
            from models.laces_latent_refiner import LACESRuntime
            runtime = LACESRuntime(
                model, checkpoint, expected_step=args.expected_step, plan_steps=args.plan_steps,
                cfg_scale=args.cfg_scale, blend=args.blend,
                diffusion_sampler=args.diffusion_sampler, protocol="aligned",
            )
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)

    # Resolved once, and PRINTED: `tokenizer.eos_token_id` alone silently disabled stopping for
    # rwkv7 (it gives 65530; the real end-of-text is 0). Logging the resolved set is how a future
    # run can tell whether its cell honoured a stop id, since the ids are not otherwise in the
    # result file -- the same log-only-provenance trap as the fla patch flag and the budget.
    stop_ids = resolve_stop_ids(model, tokenizer)
    print(f"[load] resolved stop ids: {stop_ids} "
          f"(generation_config/config/tokenizer union)", flush=True)

    # Clamp the truncation budget to the model's real positional ceiling, if it has one.
    # Recurrent models (rwkv/gla/mamba and our relay arms' frozen RWKV-7 decoder) have none
    # and keep the full CLI cap -- see positional_window for the per-architecture evidence.
    # `--max_input_tokens` alone is not safe: it was 32768 for the LongBench sweep, i.e. 16x
    # phi-2's 2048-position table, which is what produced the all-newline degenerate rows.
    input_budget, budget_audit = effective_input_budget(
        model, args.max_input_tokens, args.max_new_tokens)
    # Restore the BOS token that `add_special_tokens=False` in truncate_ids suppresses. Only
    # for tokenizers that ask for it (gemma-2-2b, gla-1.3B); adding one where the model never
    # saw it in training would be a new bug, not a fix. Recorded in the result `config` so a
    # reader can tell which rows had it. Measured effect, budget held constant: bAbILong
    # qa1/4k gemma 13% -> 53% (results/benchmarks/falsify_zero/gemma-bos-only.json).
    budget_audit["prepend_bos"] = tokenizer_wants_bos(tokenizer)
    budget_audit["bos_token_id"] = getattr(tokenizer, "bos_token_id", None)
    print(f"[budget] {json.dumps(budget_audit)}", flush=True)

    started = time.time()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    aggregates = {}
    for dataset_spec in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        if args.suite == "longbench":
            dataset = dataset_spec
            rows = load_longbench_rows(data_dir / f"{dataset}.jsonl", dataset, prompt_map)
        elif args.suite == "babilong":
            dataset = dataset_spec.replace("/", "_")
            rows = load_babilong_rows(data_dir / f"{dataset_spec}.json")
        elif args.suite == "infinitebench":
            dataset = dataset_spec.replace("/", "_")
            rows = load_infinitebench_rows(data_dir / f"{dataset_spec}.jsonl")
        elif args.suite == "memeval":
            dataset = dataset_spec.replace("/", "_")
            rows = load_memeval_rows(data_dir / dataset_spec)
        else:
            dataset = dataset_spec.replace("/", "_")
            # dataset_spec is "<length>/<task>"; the cache stores one parquet per length.
            if "/" in dataset_spec:
                length, task = dataset_spec.split("/", 1)
                cand = data_dir / length / "validation.jsonl"
                rows = (load_ruler_rows(cand, task) if cand.exists()
                        else load_ruler_rows(data_dir / length, task))
            else:
                rows = load_ruler_rows(data_dir / dataset_spec / "validation.jsonl")
        # Stride BEFORE the head-slice: [::stride][:n] samples the whole file, [:n][::stride]
        # would only thin out the head we already restricted to. See --sample_stride for why
        # the head is not a valid sample of the depth-sorted needle tasks.
        rows = rows[::max(1, args.sample_stride)][:args.max_samples]
        metric = metric_for(args.suite, dataset_spec)
        scores = []
        for index, row in enumerate(rows):
            truncation_stats: dict = {}
            prefix_ids = truncate_ids(tokenizer, row["prompt"], input_budget, args.device,
                                      stats=truncation_stats,
                                      add_bos=budget_audit["prepend_bos"])
            # Only the relay runtime consumes `z`; building it unconditionally read
            # model.trajectory_horizon off a plain HF model and killed every --model_dir run
            # with AttributeError before the first generation.
            z = None
            if runtime is not None:
                if args.arm in ("dynamic", "global"):
                    # Both plan arms sample a private plan from a row-unique seed: dynamic
                    # gets a chunked [1, H, D] trajectory, global a single [1, D] latent.
                    z, _hidden, _prefix = runtime.prepare(prefix_ids, seed=args.seed + index)
                else:
                    z = torch.zeros(1, int(model.trajectory_horizon),
                                    int(model.latent_dim), device=args.device)
            limits = [args.max_new_tokens,
                      int(max_gen_map.get(dataset_spec, args.max_new_tokens))]
            if row.get("max_new_tokens"):
                limits.append(int(row["max_new_tokens"]))
            if runtime is not None and args.arm != "global":
                # Only the chunked trajectory has a horizon that bounds the answer length; a
                # single global z is written once and imposes no such cap.
                limits.append(int(model.trajectory_horizon) * int(model.trajectory_chunk_size))
            generation_limit = min(limits)
            if runtime is None and args.decode == "loop":
                token_ids = greedy_loop(model, prefix_ids, generation_limit,
                                        eos_ids=stop_ids)
                logps = []
            elif runtime is None:
                out = model.generate(
                    input_ids=prefix_ids, max_new_tokens=generation_limit, do_sample=False,
                    pad_token_id=(getattr(tokenizer, "pad_token_id", None)
                                  or getattr(tokenizer, "eos_token_id", None)),
                )
                token_ids = out[0, prefix_ids.shape[1]:].tolist()
                logps = []
            else:
                token_ids, logps = runtime.generate(
                    prefix_ids, z, max_new_tokens=generation_limit, raw=args.arm == "raw",
                    temperature=0.0, repetition_penalty=1.0,
                    eos_ids=stop_ids,
                )
            prediction = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            if args.suite == "babilong":
                task = dataset_spec.split("/")[0]
                score = babilong_score(task, prediction, row["answers"][0], row["question"])
            elif args.suite == "ruler":
                task = dataset_spec.split("/")[-1]
                score = ruler_score(task, prediction, row["answers"])
            elif args.suite == "infinitebench":
                task = dataset_spec.split("/")[-1]
                score = infinitebench_score(task, prediction, row["answers"],
                                            row.get("options"))
            elif args.suite == "memeval":
                # longmemeval_score_approx is an approximation of the official gpt-4o
                # judge -- see its docstring before quoting the number anywhere. The
                # question is passed because the gold answer is frequently a substring of
                # it, which a bare containment check cannot survive.
                score = longmemeval_score_approx(prediction, row["answers"][0],
                                                 row.get("question", ""))
            else:
                score = max((metric(prediction, answer) for answer in row["answers"]), default=0.0)
            scores.append(score)
            rows_out.append({
                "suite": args.suite, "dataset": dataset, "id": row["id"], "arm": args.arm,
                "input_tokens": int(prefix_ids.shape[1]), **truncation_stats,
                "prediction": prediction,
                "prediction_token_ids": token_ids, "mean_logp": sum(logps) / len(logps) if logps else None,
                "answers": row["answers"], "score": score,
                # Persist the options: without them a stored row cannot be re-scored, because
                # every choice metric needs the option list. Re-scoring the 2026-09-21 runs
                # without this silently passed options=None and appeared to show a -0.40 swing
                # that did not exist.
                **({"options": row["options"]} if row.get("options") else {}),
            })
            print(f"[{args.suite}/{dataset}/{args.arm}] {index + 1}/{len(rows)} score={score:.3f}", flush=True)
        # The score alone is not a reportable number for this suite: every aggregate has to
        # carry how much of the document the model was actually shown. `seen_fraction` is the
        # one figure that separates "this model cannot do the task" from "we deleted the
        # evidence" -- at 0.02 the task is unanswerable in principle and the score says
        # nothing about the model.
        dataset_rows = [r for r in rows_out if r["dataset"] == dataset]
        full_lengths = [r["prompt_tokens_untruncated"] for r in dataset_rows
                        if r.get("prompt_tokens_untruncated")]
        seen_lengths = [r["input_tokens"] for r in dataset_rows]
        aggregates[dataset] = {
            "n": len(scores),
            "score": sum(scores) / len(scores) if scores else 0.0,
            "mean_prompt_tokens_untruncated": (sum(full_lengths) / len(full_lengths)
                                               if full_lengths else None),
            "mean_input_tokens_seen": (sum(seen_lengths) / len(seen_lengths)
                                       if seen_lengths else None),
            "rows_truncated": sum(1 for r in dataset_rows if r.get("dropped_tokens", 0) > 0),
            "seen_fraction": ((sum(seen_lengths) / sum(full_lengths))
                              if full_lengths and sum(full_lengths) else None),
        }

    peak_mb = None
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        peak_mb = torch.cuda.max_memory_allocated(args.device) / 1024**2
    result = {
        # `config` carries the *effective* budget, not just the requested one, so a reader can
        # tell a capability result from a truncation artefact without rerunning anything.
        "config": {**vars(args), **budget_audit},
        # Provenance, recorded positively so a stale file cannot masquerade as a good one.
        # Files written before --decode existed have NO `decode` key, and for rwkv7/gla that
        # absence means the disproven `hf` path (patched generate still disagrees with a
        # no-cache reference decode 0/8 on first token, scoring 0.0 where the loop scores
        # 37.5). `decode_is_trustworthy` is False exactly when an fla recurrent model was
        # decoded with `hf`, so a table builder can filter on one boolean instead of having to
        # know which architectures are affected.
        "decode_path": args.decode if args.model_dir else f"relay:{args.arm}",
        "stop_ids": stop_ids,
        "decode_is_trustworthy": bool(args.ckpt_dir or args.decode == "loop" or not patched),
        "checkpoint_step": checkpoint.get("step") if checkpoint else None,
        "audit": runtime.audit if runtime is not None else None,
        "aggregates": aggregates, "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_mb": peak_mb, "rows": rows_out,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if runtime is not None:
        runtime.close()
    print(json.dumps({"aggregates": aggregates, "peak_gpu_memory_mb": peak_mb}, indent=2), flush=True)


if __name__ == "__main__":
    main()

import json

import pytest

from scripts.eval.run_open_context_bench import (
    as_list,
    babilong_score,
    effective_input_budget,
    infinitebench_score,
    load_babilong_rows,
    load_infinitebench_rows,
    load_longbench_rows,
    load_memeval_rows,
    longmemeval_score_approx,
    normalize_freeform,
    normalized_exact_match,
    positional_window,
    qa_f1,
    rouge_l_f1,
    ruler_score,
    truncate_ids,
)


def test_longbench_adapter_uses_official_prompt_and_answers(tmp_path):
    data = tmp_path / "hotpotqa.jsonl"
    data.write_text(json.dumps({"input": "Who won?", "context": "Ada won.", "answers": ["Ada"], "_id": "x"}) + "\n")
    prompts = {"hotpotqa": "Context: {context}\nQuestion: {input}\nAnswer:"}

    rows = load_longbench_rows(data, "hotpotqa", prompts)

    assert rows == [{"id": "x", "prompt": "Context: Ada won.\nQuestion: Who won?\nAnswer:", "answers": ["Ada"]}]


def test_babilong_adapter_builds_open_answer_prompt(tmp_path):
    data = tmp_path / "qa2.json"
    data.write_text(json.dumps([{"input": "Mary went home.", "question": "Where is Mary? ", "target": "home"}]))

    rows = load_babilong_rows(data)

    assert rows == [{
        "id": 0,
        "prompt": "Mary went home.\nQuestion: Where is Mary?\nAnswer:",
        "question": "Where is Mary?",
        "answers": ["home"],
    }]


def test_open_text_metrics_are_answer_normalized():
    assert normalized_exact_match("The Bathroom.", "bathroom") == 1.0
    assert qa_f1("Miller v California", "Miller v. California") == 1.0
    assert 0.0 < rouge_l_f1("alpha beta gamma", "alpha gamma") < 1.0


def test_babilong_score_matches_official_label_filtering():
    assert babilong_score("qa1", "Mary is in the bathroom. Question: Where is John?", "bathroom", "Where is Mary?") == 1.0
    assert babilong_score("qa1", "bathroom and bedroom", "bathroom", "Where is Mary?") == 0.0


def test_ruler_score_uses_official_any_and_all_substring_rules():
    assert ruler_score("qa_1", "The answer is Ada.", ["Ada", "Grace"]) == 1.0
    assert ruler_score("vt", "x and y", ["x", "y"]) == 1.0
    assert ruler_score("vt", "x alone", ["x", "y"]) == 0.5


def test_as_list_parses_the_stringified_answers_the_local_copies_ship():
    # The pinned LongBench jsonl stores the list as text; list("['Ada']") would be 7 chars.
    assert as_list("['Miller v. California']") == ["Miller v. California"]
    assert as_list('["a", "b"]') == ["a", "b"]
    assert as_list(["Ada"]) == ["Ada"]              # already a list
    assert as_list("Ada") == ["Ada"]                # bare string
    assert as_list(None) == []
    assert as_list("['unclosed") == ["['unclosed"]  # unparseable -> keep it whole


def test_longbench_stringified_answers_survive_the_loader(tmp_path):
    data = tmp_path / "hotpotqa.jsonl"
    data.write_text(json.dumps({"input": "Who won?", "context": "Ada won.",
                                "answers": "['Ada']", "_id": "x"}) + "\n")

    rows = load_longbench_rows(data, "hotpotqa", {"hotpotqa": "{context} {input}"})

    assert rows[0]["answers"] == ["Ada"]
    assert qa_f1("Ada", rows[0]["answers"][0]) == 1.0


def test_babilong_score_refuses_tasks_without_a_label_set():
    with pytest.raises(KeyError):
        babilong_score("qa4", "the bedroom", "bedroom", "Where is Mary?")


def test_infinitebench_jsonl_rows_carry_options_and_a_prompt_cue(tmp_path):
    mc = tmp_path / "longbook_choice_eng.jsonl"
    mc.write_text(json.dumps({"id": 3, "context": "Bella walked the dog.",
                              "input": "What did Bella do?",
                              "answer": ["Walking the dog"],
                              "options": ["Sleeping", "Walking the dog"]}) + "\n")
    free = tmp_path / "passkey.jsonl"
    free.write_text(json.dumps({"id": 0, "context": "The pass key is 71432.",
                                "input": "What is the pass key?", "answer": ["71432"],
                                "options": []}) + "\n")
    # code_run's own instruction already ends the response, so it must be fed verbatim.
    # The jsonl has no task column -- the filename stem is the only way the loader can know.
    code = tmp_path / "code_run.jsonl"
    code.write_text(json.dumps({"id": 1, "context": "def func_1(x): return x",
                                "input": "Return value of func_1(2). Your response should "
                                         "end with the sentence 'The return value is:'.",
                                "answer": [2], "options": []}) + "\n")

    mc_row = load_infinitebench_rows(mc)[0]
    free_row = load_infinitebench_rows(free)[0]
    code_row = load_infinitebench_rows(code)[0]

    assert mc_row["options"] == ["Sleeping", "Walking the dog"]
    assert mc_row["prompt"].endswith("Options:\nA. Sleeping\nB. Walking the dog\n\nAnswer:")
    # An empty options list means a free-form task, not a choice block.
    assert "Options:" not in free_row["prompt"]
    assert free_row["prompt"].endswith("Answer:")
    assert not code_row["prompt"].endswith("Answer:")


def test_infinitebench_choice_accepts_the_letter_or_the_option_text():
    options = ["Sleeping", "Walking the dog"]
    gold = ["Walking the dog"]

    assert infinitebench_score("longbook_choice_eng", "B", gold, options) == 1.0
    assert infinitebench_score("longbook_choice_eng", "Walking the dog", gold, options) == 1.0
    assert infinitebench_score("longbook_choice_eng", "A", gold, options) == 0.0
    assert infinitebench_score("longbook_choice_eng", "", gold, options) == 0.0


def test_infinitebench_freeform_scores_gold_and_rejects_empty():
    assert infinitebench_score("passkey", "71432", ["71432"], []) == 1.0
    assert infinitebench_score("passkey", "", ["71432"], []) == 0.0
    # kv_retrieval is the all-references task: one of two values is not enough.
    # Its references are UUIDs, and the check is the official plain substring one -- so a
    # one-character reference would match inside any word containing that letter. That is
    # the official rule's behaviour, not a bug here, which is why the values are realistic.
    first, second = "5e6b7b90-710d-4953-9b18-3e96b2cadbf2", "cb59052b-9128-4979-9c0e-e1de4adcf73b"
    assert infinitebench_score("kv_retrieval", f"{first} and {second}", [first, second], []) == 1.0
    assert infinitebench_score("kv_retrieval", f"only {first}", [first, second], []) == 0.0


def test_memeval_rows_expose_question_type_and_abstention(tmp_path):
    item = {
        "question_id": "abc_abs", "question_type": "single-session-user",
        "question": "What degree did I graduate with?", "answer": "Business Administration",
        "question_date": "2023/05/30 (Tue) 23:40",
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 09:00"],
        "haystack_session_ids": ["s0", "s1"],
        "haystack_sessions": [
            [{"role": "user", "content": "I graduated.", "has_answer": True},
             {"role": "assistant", "content": "Congratulations."}],
            [{"role": "user", "content": "Unrelated."}],
        ],
        "answer_session_ids": ["s0"],
    }
    data = tmp_path / "longmemeval_oracle"          # the cache blob has no extension
    data.write_text(json.dumps([item]))

    row = load_memeval_rows(data)[0]

    assert row["id"] == "abc_abs"
    assert row["answers"] == ["Business Administration"]
    assert row["is_abstention"] is True
    assert "=== Session 1 (2023/05/20 (Sat) 02:21) ===" in row["prompt"]
    assert "user: I graduated." in row["prompt"]
    assert row["prompt"].endswith("Question (asked on 2023/05/30 (Tue) 23:40): "
                                  "What degree did I graduate with?\nAnswer:")


def test_memeval_score_is_an_approximation_with_the_expected_edges():
    gold = "Business Administration"
    assert longmemeval_score_approx("Business Administration", gold) == 1.0
    assert longmemeval_score_approx("I think it is Business Administration.", gold) == 1.0
    assert longmemeval_score_approx("", gold) == 0.0
    # The substitute cannot score abstention: a correct refusal is still a miss.
    assert longmemeval_score_approx("I don't know.", gold) < 1.0
    assert longmemeval_score_approx("totally unrelated words", gold) == 0.0


# --------------------------------------------------------------------------------------
# Regressions for the 2026-09-21 InfiniteBench all-zero investigation.
# --------------------------------------------------------------------------------------


class _Cfg:
    """Minimal stand-in for a HF config; only the fields the budget logic reads."""

    def __init__(self, model_type, max_position_embeddings=None):
        self.model_type = model_type
        self.max_position_embeddings = max_position_embeddings


class _Model:
    def __init__(self, config):
        self.config = config


def test_recurrent_models_keep_the_full_budget_despite_a_declared_window():
    """fla-hub's gla-1.3B config.json declares max_position_embeddings 2048, but GLA is a
    linear-attention recurrent model with no position table -- the field is inert. Clamping
    to it would discard 94% of the context and manufacture a ceiling that does not exist."""
    for model_type in ("gla", "rwkv7", "rwkv6", "mamba", "gated_deltanet"):
        model = _Model(_Cfg(model_type, max_position_embeddings=2048))
        assert positional_window(model) is None, model_type
        budget, audit = effective_input_budget(model, 32768, 128)
        assert budget == 32768, model_type
        assert "recurrent" in audit["truncation_reason"]


def test_positional_models_are_clamped_and_reserve_room_for_generation():
    """phi-2 fed 32768 tokens into a 2048-position table emitted only newlines and scored an
    exact 0.0 on all five InfiniteBench datasets. The generated tokens occupy positions too,
    so the prompt budget must also leave room for max_new_tokens."""
    phi = _Model(_Cfg("phi", max_position_embeddings=2048))
    budget, audit = effective_input_budget(phi, 32768, 128)
    assert budget == 2048 - 128
    assert audit["truncation_reason"] == "positional_window"
    assert audit["positional_window"] == 2048
    # A window far above the cap leaves the cap alone.
    big = _Model(_Cfg("llama", max_position_embeddings=131072))
    budget, audit = effective_input_budget(big, 32768, 128)
    assert budget == 32768
    assert audit["truncation_reason"] == "cli_cap_only"


def test_the_effective_budget_is_recorded_for_reporting():
    """A score is uninterpretable without the input length it was measured at: "truncated to
    2k of ~136k" is a different claim from the same number reported bare."""
    _, audit = effective_input_budget(_Model(_Cfg("phi", 2048)), 32768, 128)
    assert audit["requested_max_input_tokens"] == 32768
    assert audit["effective_input_budget"] == 1920
    assert audit["reserved_for_generation"] == 128


def test_choice_scoring_accepts_a_glued_bare_letter():
    """"BQuestion:" -- a letter running straight into the next question with no separator,
    common under a tight token cap -- was judged unmatched and scored 0 even when correct."""
    options = ["Sleeping", "Walking the dog", "Reading", "Cooking"]
    gold = ["Walking the dog"]
    for prediction in ("B", "B.", "(B)", "B) something", "BQuestion:", "B\nQuestion"):
        assert infinitebench_score("longbook_choice_eng", prediction, gold, options) == 1.0, prediction
    # Must not fire on ordinary prose that merely starts with an option letter.
    assert infinitebench_score("longbook_choice_eng", "Bella walked away", gold, options) == 0.0


def test_choice_scoring_takes_the_first_option_mentioned_not_the_longest():
    """A greedy decoder fed an "Options: A..D" block often echoes the whole menu back, so
    several option texts appear in the generation. The old rule picked the LONGEST, which is
    a property of the option list rather than of the answer; measured disagreement with
    first-mention was 20/50 rows for gla-1.3b on longbook_choice_eng. The answer is what the
    model says first; the echoed menu comes after it."""
    options = ["Sleeping", "Walking the dog", "Reading", "Cooking"]
    echo = "Reading. Options: A. Sleeping B. Walking the dog C. Reading D. Cooking"
    # "Walking the dog" is the longest match present, but "Reading" is what was answered.
    assert infinitebench_score("longbook_choice_eng", echo, ["Reading"], options) == 1.0
    assert infinitebench_score("longbook_choice_eng", echo, ["Walking the dog"], options) == 0.0


def test_quoted_gold_answers_still_match():
    """InfiniteBench stores some gold answers with embedded quotes ('"Peyton"'). The shared
    normaliser strips punctuation, so this was never the cause of the all-zero run -- pinned
    here so a future normaliser change cannot reintroduce it."""
    assert infinitebench_score("longbook_qa_eng", "Peyton", ['"Peyton"'], []) == 1.0


def test_freeform_scorer_credits_question_echo_as_a_perfect_answer():
    """CHARACTERISATION of a known defect, not desired behaviour.

    The free-form branch is ``gold_normalized in prediction_normalized``, a bare substring test
    that cannot separate "answered the question" from "restated the prompt". 5 of the 50
    evaluated longbook_qa_eng rows are "which among A, B, C, D ...?" questions whose gold is
    listed among the candidates, so a model that echoes the question earns 1.0 for free -- which
    preferentially rewards the degenerate, looping models this harness exists to detect.

    Real measured example (global-dit-50k, longbook_qa_eng id=0, gold '"Peyton"'): the model
    answers "Gannonmarie Lucy", which is WRONG, then loops the question back. It scored 1.0.

    This test pins the defect so the inflation is visible in the suite rather than only in
    logs/qz/INFINITEBENCH_FIX_JOBS.md. When the scorer is fixed to require the answer to appear
    outside an echoed question, invert these assertions.
    """
    wrong_answer_then_question_echo = (
        "Gannonmarie Lucy, Mrs. Bronwyn's child.\n"
        "Which among Annalisa, Seb, Gannonmarie Lucy, and Peyton Merida is not "
        "Mrs. Bronwyn's child?"
    )
    assert infinitebench_score(
        "longbook_qa_eng", wrong_answer_then_question_echo, ['"Peyton"'], []) == 1.0

    # A pure loop that merely opens with the gold phrase also collects full credit.
    loop_that_opens_correctly = "A red hunting hat.\n" + "What did Viola do? She left.\n" * 5
    assert infinitebench_score(
        "longbook_qa_eng", loop_that_opens_correctly, ["A red hunting hat"], []) == 1.0


def test_truncation_records_how_much_of_the_document_was_dropped():
    """InfiniteBench prompts are ~30k-170k tokens against windows of 2k-32k, so almost every
    row is truncated. A row that stores only the post-truncation length cannot distinguish
    "answered a 2k prompt badly" from "was shown 1.4% of the document", which is the whole
    difference between a capability result and an artefact of our own harness."""

    class _Tok:
        def __call__(self, text, return_tensors=None, add_special_tokens=None):
            import torch

            class _Out:
                pass

            out = _Out()
            out.input_ids = torch.arange(len(text.split())).unsqueeze(0)
            return out

    stats = {}
    ids = truncate_ids(_Tok(), " ".join(["w"] * 136000), 1920, "cpu", stats=stats)
    assert ids.shape[1] == 1920
    assert stats["prompt_tokens_untruncated"] == 136000
    assert stats["dropped_tokens"] == 136000 - 1920

    # A prompt that fits is recorded as untruncated, not silently omitted.
    stats = {}
    truncate_ids(_Tok(), " ".join(["w"] * 500), 1920, "cpu", stats=stats)
    assert stats == {"prompt_tokens_untruncated": 500, "dropped_tokens": 0}


def test_replacement_char_does_not_glue_words_in_scoring():
    """U+FFFD must not fuse the tokens around it.

    The relay arms' RWKV tokenizer decodes token 193 (the lone byte 0xC0) to U+FFFD wherever
    an HF baseline emits a newline, so our arms' predictions are full of it. U+FFFD is absent
    from ``string.punctuation``, so a normaliser that only strips punctuation leaves it in
    place and a containment check fails on a gold the model did in fact emit -- a scoring
    penalty that applies to our arm and to no baseline.
    """
    assert normalize_freeform("Answer�United Airlines") == "answer united airlines"
    # The whole point: the gold is found with the artefact present.
    assert infinitebench_score(
        "longbook_qa_eng", "Answer�United Airlines", ["United Airlines"]) == 1.0
    # And a real newline behaves identically, so baselines are unaffected.
    assert (normalize_freeform("Answer�United") == normalize_freeform("Answer\nUnited"))


def test_passkey_uses_official_first_int_match_not_containment():
    """passkey / number_string: ``label == first_int_match(pred)``, ported verbatim.

    This is the single most inflating defect this suite could have, and it would have
    inflated OUR arm precisely where the experiment claims a win. Every passkey context
    OPENS with "The pass key is 71432. Remember it. 71432 is the pass key.", so a model that
    merely parrots the start of its own prompt contains the gold string. Under the previous
    default branch (containment, then qa_f1) that scored 1.0 -- a perfect score for echoing
    the input. The official rule takes the FIRST run of digits and demands exact string
    equality, so the echo still scores 1.0 only because the key genuinely is the first
    number; but a leading distractor digit, or zero-padding, now correctly scores 0.
    """
    assert infinitebench_score("passkey", "71432", ["71432"]) == 1.0
    assert infinitebench_score("passkey", "The pass key is 71432.", ["71432"]) == 1.0
    # Exact string equality: zero-padding is NOT the gold (official compares strings).
    assert infinitebench_score("passkey", "00071432", ["71432"]) == 0.0
    # FIRST integer only -- a right answer after a wrong one scores 0.
    assert infinitebench_score("passkey", "maybe 5, actually 71432", ["71432"]) == 0.0
    # No digits at all, and the empty generation.
    assert infinitebench_score("passkey", "I don't know", ["71432"]) == 0.0
    assert infinitebench_score("passkey", "", ["71432"]) == 0.0
    assert infinitebench_score("number_string", "3000077442", ["3000077442"]) == 1.0
    # The old containment branch would have scored this 1.0; the official rule does not.
    assert infinitebench_score("number_string", "1 then 3000077442", ["3000077442"]) == 0.0


def test_code_run_takes_the_last_integer_and_math_calc_is_an_ordered_prefix():
    """code_run: last whitespace word as int. math_calc: prefix match that stops on error."""
    assert infinitebench_score("code_run", "The return value is: 4", ["4"]) == 1.0
    assert infinitebench_score("code_run", "4 then 9", ["4"]) == 0.0
    assert infinitebench_score("code_run", "no number here", ["4"]) == 0.0
    # math_calc stops at the FIRST mismatch: 2 of 4 correct, not set-recall over all golds.
    assert infinitebench_score("math_calc", "79 59 0 0", ["79", "59", "73", "78"]) == 0.5
    # Set-recall would have scored this 1.0 (every gold appears, in any order); the ordered
    # prefix mismatches at index 0 (gold 79 vs predicted 78) and stops there, scoring 0.
    assert infinitebench_score("math_calc", "78 73 59 79", ["79", "59", "73", "78"]) == 0.0
    # A correct leading run still earns its prefix credit even if the tail is wrong.
    assert infinitebench_score("math_calc", "79 59 73 78", ["79", "59", "73", "78"]) == 1.0


def test_longdialogue_accepts_any_alias_case_insensitively():
    """Official upper-cases the prediction and accepts ANY alias as a substring."""
    assert infinitebench_score(
        "longdialogue_qa_eng", "the answer is ace rothstein", ["ACE", "ACE ROTHSTEIN"]) == 1.0
    assert infinitebench_score("longdialogue_qa_eng", "Hushpuppy", ["ACE"]) == 0.0


def test_depth_sorted_needle_files_require_a_stride_to_be_sampled():
    """rows[::stride][:n] samples the file; rows[:n][::stride] only thins the head.

    passkey/number_string ship 590 rows as 59 depth bins x 10 rows, sorted by needle depth
    (verified on the cached snapshot). ``rows[:50]`` is therefore the 5 SHALLOWEST bins --
    needle depth 0.000-0.069 -- not a sample of the benchmark, so a cell built that way
    measures the easiest 8.5% of the task and would overstate a deep-recall claim.
    """
    rows = list(range(590))
    # The order of operations is the whole point.
    assert rows[::10][:50] == list(range(0, 500, 10))       # walks 50 distinct depth bins
    assert rows[:50][::10] == [0, 10, 20, 30, 40]           # still only the shallowest bins

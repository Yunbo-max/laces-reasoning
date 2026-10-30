#!/usr/bin/env bash
# DCLM-1B (TRI-ML/DCLM-1B): full benchmark suite, matching the protocol the 9 existing
# baselines ran. Companion to run_pythia28_full_suite.sh; same structure, same flags.
#
# ARCHITECTURE: this checkpoint is `model_type: openlm` / `architectures: ["OpenLMModel"]`,
# which transformers cannot resolve and which ships no auto_map and no modeling file. It is
# loaded by mlfoundations' own `open_lm` package, whose inference subtree is installed in
# $ROOT/shared_pylibs/open_lm (model.py, attention.py, norms.py, positional_embedding/, hf/).
# `run_open_context_bench.py::ensure_model_type_resolvable` imports `open_lm.hf` for
# model_type "openlm", which self-registers OpenLMConfig/OpenLMForCausalLM with the Auto
# classes. THE PYTHONPATH EXPORT BELOW IS THEREFORE LOAD-BEARING, not boilerplate: without
# it nothing about this model loads at all.
#
# WINDOW -- the fact that governs this whole row. config.json says `seq_len: 2048` and does
# NOT carry max_position_embeddings or n_positions, so `positional_window()` needed a
# `seq_len` fallback (added, regression-checked: every other baseline's window is
# unchanged). Without that fallback the harness classified DCLM as having no positional
# ceiling -- the branch meant for recurrent models -- and would have fed it 16k/32k prompts.
# That fails OPEN and SILENTLY for this architecture, which is worse than the phi-2
# precedent: open_lm's RotaryEmbedding._update_cos_sin_tables REBUILDS the cos/sin cache for
# whatever length it is handed (`self._update_cos_sin_tables(k.shape[1] + offset, ...)`), so
# there is no index error and no warning -- just RoPE extrapolated far past trained length.
#
# Effective budgets after the clamp (window 2048 - max_new_tokens):
#     babilong  (max_new 32)  -> 2016
#     ruler     (max_new 128) -> 1920
#     longbench (max_new 512) -> 1536
#     infbench  (max_new 128) -> 1920
#     memeval   (max_new 128) -> 1920
# The tokenizer-only pre-flight audit (scripts/eval/audit_dclm_window.py ->
# results/benchmarks/longctx/dclm-1b__window_audit.json) measured every briefed cell with
# DCLM's own tokenizer: 0 of 23 are measurable. seen_fraction runs 0.0090 (InfiniteBench
# code_debug) to 0.6881 (RULER 4096 qa_1), and EVERY row of EVERY cell truncates. So, as
# with Pythia, two things run here:
#   (a) the briefed cells, unchanged, so the result files carry the per-row
#       input_tokens / prompt_tokens_untruncated / seen_fraction that PROVE unmeasurability
#       instead of asserting it; and
#   (b) extra IN-WINDOW cells (bAbILong 0k/1k/2k) that DCLM can actually be scored on.
# Reporting the clamped 16k cell as a 16k number is the artefact this file exists to avoid.
#
# BOS: DCLM's GPTNeoXTokenizer returns identical ids with and without special tokens
# (probed: tokenizer("x") -> [89] both ways; bos_id=0 is <|endoftext|>, and BOS==EOS==UNK),
# so tokenizer_wants_bos() correctly returns False and there is no suppressed BOS to
# restore. Measured negative, not an assumption -- the `add_bos_token` flag is
# zero-information in this project and was not relied on.
#
# DTYPE: config.json declares `torch_dtype: float32`. The long-context harness hard-codes
# bfloat16 for --model_dir, and eval_lm_ppl.py's dtype probe is inert (its os.path.join has
# no module-level `import os`, so the NameError is swallowed and every --impl hf row silently
# loads bf16). Both are LEFT ALONE on purpose: all 9 published baselines and all 8 existing
# ppl16 HF rows were produced under exactly that behaviour, so "fixing" it would make DCLM
# the only fp32 row in a bf16 table -- a comparability break dressed up as a fix. The bf16
# cast is verified numerically against fp32 before submission (see the load check), and the
# bug is reported in the ledger.
set -uo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MODEL=$ROOT/hf_cache/hub/models--TRI-ML--DCLM-1B/snapshots/f43c272dab1b06bc5fdd6bbb5819f8a30ef2833b
NAME=dclm-1b
BENCH=$REPO/scripts/eval/run_open_context_bench.py
OUT=$REPO/results/benchmarks/longctx
OUTW=$REPO/results/benchmarks/longctx_inwindow
LOGD=$REPO/logs/qz
TASKDATA=$ROOT/research/DiffRwkv/baseline/Cola-DLM/generate_task_data

mkdir -p "$OUT" "$OUTW" "$LOGD" "$REPO/results/benchmarks/ppl16" \
         "$REPO/results/benchmarks/baseline-dclm-1b" \
         "$REPO/results/benchmarks/fix2choices/baseline-dclm-1b"
cd "$REPO"

BABI=$ROOT/hf_cache/hub/datasets--RMT-team--babilong/snapshots/ee0d588794c7ac098062ee0d247c733d62e94fe2/data
RULER=$ROOT/hf_cache/hub/datasets--simonjegou--ruler/snapshots/24adceac8a0e6532936e8d721cd9e9084d2e4686
LB=$ROOT/longbench_local/extract/data
LBSPEC=$ROOT/longbench_local/pinned_spec
INF=$ROOT/hf_cache/hub/datasets--xinrongzhang2022--InfiniteBench/snapshots/90f0394333616266d9fe85824ceaf505093cbaa5
MEM=$ROOT/hf_cache/hub/datasets--xiaowu0162--longmemeval/snapshots/2ec2a557f339b6c0369619b1ed5793734cc87533

fail=0

# ---------------------------------------------------------------- short context (PRIORITY)
# Every one of these prompts is far inside 2048, so these are the numbers that matter most
# for this family -- and, given 0/23 long-context cells are measurable, very nearly the only
# ones. Flags copied verbatim from the submitted qwen2.5-3b job bodies: 200 samples, 32 new
# tokens, temperature 0.0 (greedy). repetition_penalty is left at the harness default of 1.1,
# exactly as every published baseline run did (no published job passes the flag explicitly).
echo "=== [1/7] short-context: 8 non-option columns -> baseline-dclm-1b ==="
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_generic_gen_match.py \
  --model_dir "$MODEL" \
  --task_data_dir "$TASKDATA" \
  --output_dir "$REPO/results/benchmarks/baseline-dclm-1b" \
  --tasks mmlu,obqa,race,squad,story_cloze,hellaswag,lambada,siqa \
  --max_samples 200 --max_new_tokens 32 --temperature 0.0 --trust_remote_code \
  > "$LOGD/dclm1b-genmatch-main.log" 2>&1 || fail=1

# The six option-bearing columns go to fix2choices/ (the post-prompt-fix set), because
# inline_choices() changes their prompt and mixing the two waves is not a controlled compare.
echo "=== [2/7] short-context: 6 option columns -> fix2choices/baseline-dclm-1b ==="
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_generic_gen_match.py \
  --model_dir "$MODEL" \
  --task_data_dir "$TASKDATA" \
  --output_dir "$REPO/results/benchmarks/fix2choices/baseline-dclm-1b" \
  --tasks arc_easy,arc_challenge,winogrande,sciq,commonsenseqa,piqa \
  --max_samples 200 --max_new_tokens 32 --temperature 0.0 --trust_remote_code \
  > "$LOGD/dclm1b-genmatch-fix2.log" 2>&1 || fail=1

# ---------------------------------------------------------------- PPL (PRIORITY)
# 256-token chunks, comfortably inside 2048, so this is a fully in-window measurement.
echo "=== [3/7] PPL, 7 corpora -> ppl16/dclm-1b.json ==="
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_lm_ppl.py --impl hf \
  --model_dir "$MODEL" --mode raw --trust_remote_code \
  --corpora wikitext2,wikitext103,pg19,lm1b,ptb,wmt14,xsum \
  --max_files 100 --output "$REPO/results/benchmarks/ppl16/dclm-1b.json" \
  > "$LOGD/ppl16-dclm-1b.log" 2>&1 || fail=1

# ---------------------------------------------------------------- bAbILong IN WINDOW
# The long-context cells DCLM can actually be scored on. Same 200 samples and same flags as
# the briefed cells; only the context length differs, so 0k/1k/2k vs 4k/8k/16k is a
# controlled length sweep on one model rather than two protocols.
echo "=== [4/7] bAbILong IN-WINDOW 0k/1k/2k ==="
CUDA_VISIBLE_DEVICES=0 python3 -u "$BENCH" \
  --suite babilong --datasets qa1/0k,qa1/1k,qa1/2k,qa2/0k,qa2/1k,qa2/2k \
  --data_dir "$BABI" --model_dir "$MODEL" --arm raw \
  --output "$OUTW/${NAME}__babilong_inwindow.json" \
  --max_samples 200 --max_input_tokens 16384 --max_new_tokens 32 --seed 42 --device cuda \
  > "$LOGD/dclm1b-babilong-inwindow.log" 2>&1 || fail=1

# ---------------------------------------------------------------- briefed long-context grid
# Run to produce the truncation audit, NOT to harvest scores. Every cell here is expected to
# come back with seen_fraction < 1.0 (pre-measured: 0.131-0.558 for babilong) and is reported
# as "not measurable within this model's window". The result file is the evidence.
echo "=== [5/7] bAbILong briefed grid (expected OUT OF WINDOW) ==="
CUDA_VISIBLE_DEVICES=0 python3 -u "$BENCH" \
  --suite babilong --datasets qa1/4k,qa1/8k,qa1/16k,qa2/4k \
  --data_dir "$BABI" --model_dir "$MODEL" --arm raw \
  --output "$OUT/${NAME}__babilong.json" \
  --max_samples 200 --max_input_tokens 16384 --max_new_tokens 32 --seed 42 --device cuda \
  > "$LOGD/dclm1b-babilong.log" 2>&1 || fail=1

echo "=== [6/7] RULER (expected OUT OF WINDOW: smallest shipped length is 4096) ==="
CUDA_VISIBLE_DEVICES=0 python3 -u "$BENCH" \
  --suite ruler \
  --datasets 4096/niah_single_1,4096/niah_multikey_1,4096/vt,4096/cwe,4096/fwe,4096/qa_1,16384/niah_single_1,16384/vt,16384/cwe \
  --data_dir "$RULER" --model_dir "$MODEL" --arm raw \
  --output "$OUT/${NAME}__ruler.json" \
  --max_samples 200 --max_input_tokens 16384 --max_new_tokens 128 --seed 42 --device cuda \
  > "$LOGD/dclm1b-ruler.log" 2>&1 || fail=1

echo "=== [7/7] LongBench / InfiniteBench / MemEval (expected OUT OF WINDOW) ==="
CUDA_VISIBLE_DEVICES=0 python3 -u "$BENCH" \
  --suite longbench --datasets narrativeqa,qasper,multifieldqa_en,hotpotqa \
  --data_dir "$LB" --longbench_config_dir "$LBSPEC" --model_dir "$MODEL" --arm raw \
  --output "$OUT/${NAME}__longbench.json" \
  --max_samples 50 --max_input_tokens 32768 --max_new_tokens 512 --seed 42 --device cuda \
  > "$LOGD/dclm1b-longbench.log" 2>&1 || fail=1

CUDA_VISIBLE_DEVICES=0 python3 -u "$BENCH" \
  --suite infinitebench \
  --datasets longbook_qa_eng,longbook_choice_eng,kv_retrieval,code_debug,math_find \
  --data_dir "$INF" --model_dir "$MODEL" --arm raw \
  --output "$OUT/${NAME}__infinitebench.json" \
  --max_samples 50 --max_input_tokens 32768 --max_new_tokens 128 --seed 42 --device cuda \
  > "$LOGD/dclm1b-infinitebench.log" 2>&1 || fail=1

# MemEval's 50-row slice has a random floor of 0.220 and known scorer defects, so this is run
# for completeness and will be reported as uninformative at or below that floor regardless.
CUDA_VISIBLE_DEVICES=0 python3 -u "$BENCH" \
  --suite memeval --datasets longmemeval_oracle \
  --data_dir "$MEM" --model_dir "$MODEL" --arm raw \
  --output "$OUT/${NAME}__memeval.json" \
  --max_samples 50 --max_input_tokens 131072 --max_new_tokens 128 --seed 42 --device cuda \
  > "$LOGD/dclm1b-memeval.log" 2>&1 || fail=1

echo "=== ALL STEPS DONE fail=$fail $(date -u) ==="
exit $fail

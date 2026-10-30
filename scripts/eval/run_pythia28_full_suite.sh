#!/usr/bin/env bash
# pythia-2.8b: full benchmark suite, matching the protocol the 9 existing baselines ran.
#
# WINDOW: pythia-2.8b is gpt_neox with max_position_embeddings=2048 and is NOT on the
# runner's RECURRENT_MODEL_TYPES list, so effective_input_budget() clamps every suite
# automatically. Verified on CPU against pythia-410m (identical arch, identical 2048
# window) before submission:
#     babilong  (max_new 32)  -> 2016
#     ruler     (max_new 128) -> 1920
#     longbench (max_new 512) -> 1536
#     infbench  (max_new 128) -> 1920
#     memeval   (max_new 128) -> 1920
# Measured with Pythia's own tokenizer, 20 rows/cell: bAbILong 4k/8k/16k prompts run
# 3140-15805 tokens, i.e. 0/20 fit in 2016, while 0k/1k/2k fit 20/20. RULER's smallest
# shipped length is 4096. So the briefed long-context grid is almost entirely OUT OF
# WINDOW and its cells are NOT reportable as scores.
#
# That is why two things run here:
#   (a) the briefed cells, unchanged, so the result files carry the per-row
#       input_tokens / prompt_tokens_untruncated / seen_fraction that PROVE
#       unmeasurability instead of asserting it; and
#   (b) extra IN-WINDOW cells (bAbILong 0k/1k/2k) that Pythia can actually be scored on.
# Reporting the clamped 16k cell as a 16k number is the artefact this file exists to avoid.
#
# BOS: Pythia's GPTNeoXTokenizer returns identical ids with and without special tokens
# (probed: tokenizer("x") -> [89] both ways, bos_id=0 is <|endoftext|>), so
# tokenizer_wants_bos() correctly returns False and there is no suppressed BOS to restore.
# This is a measured negative, not an assumption.
set -uo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MODEL=$ROOT/hf_cache/hub/models--EleutherAI--pythia-2.8b/snapshots/2a259cdd96a4beb1cdf467512e3904197345f6a9
NAME=pythia-2.8b
BENCH=$REPO/scripts/eval/run_open_context_bench.py
OUT=$REPO/results/benchmarks/longctx
OUTW=$REPO/results/benchmarks/longctx_inwindow
LOGD=$REPO/logs/qz
TASKDATA=$ROOT/research/DiffRwkv/baseline/Cola-DLM/generate_task_data

mkdir -p "$OUT" "$OUTW" "$LOGD" "$REPO/results/benchmarks/ppl16" \
         "$REPO/results/benchmarks/baseline-pythia-28b" \
         "$REPO/results/benchmarks/fix2choices/baseline-pythia-28b"
cd "$REPO"

BABI=$ROOT/hf_cache/hub/datasets--RMT-team--babilong/snapshots/ee0d588794c7ac098062ee0d247c733d62e94fe2/data
RULER=$ROOT/hf_cache/hub/datasets--simonjegou--ruler/snapshots/24adceac8a0e6532936e8d721cd9e9084d2e4686
LB=$ROOT/longbench_local/extract/data
LBSPEC=$ROOT/longbench_local/pinned_spec
INF=$ROOT/hf_cache/hub/datasets--xinrongzhang2022--InfiniteBench/snapshots/90f0394333616266d9fe85824ceaf505093cbaa5
MEM=$ROOT/hf_cache/hub/datasets--xiaowu0162--longmemeval/snapshots/2ec2a557f339b6c0369619b1ed5793734cc87533

fail=0

# ---------------------------------------------------------------- short context (PRIORITY)
# 2.8B vs our 2.9B renderer is the only genuinely size-matched comparison in the table, and
# every one of these prompts is far inside 2048, so these are the numbers that matter most.
# Flags copied verbatim from the submitted qwen2.5-3b job bodies: 200 samples, 32 new tokens,
# temperature 0.0 (greedy). repetition_penalty is left at the harness default of 1.1, exactly
# as every published baseline run did (no published job passes the flag explicitly).
echo "=== [1/7] short-context: 8 non-option columns -> baseline-pythia-28b ==="
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_generic_gen_match.py \
  --model_dir "$MODEL" \
  --task_data_dir "$TASKDATA" \
  --output_dir "$REPO/results/benchmarks/baseline-pythia-28b" \
  --tasks mmlu,obqa,race,squad,story_cloze,hellaswag,lambada,siqa \
  --max_samples 200 --max_new_tokens 32 --temperature 0.0 \
  > "$LOGD/pythia28-genmatch-main.log" 2>&1 || fail=1

# The six option-bearing columns go to fix2choices/ (the post-prompt-fix set), because
# inline_choices() changes their prompt and mixing the two waves is not a controlled compare.
echo "=== [2/7] short-context: 6 option columns -> fix2choices/baseline-pythia-28b ==="
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_generic_gen_match.py \
  --model_dir "$MODEL" \
  --task_data_dir "$TASKDATA" \
  --output_dir "$REPO/results/benchmarks/fix2choices/baseline-pythia-28b" \
  --tasks arc_easy,arc_challenge,winogrande,sciq,commonsenseqa,piqa \
  --max_samples 200 --max_new_tokens 32 --temperature 0.0 \
  > "$LOGD/pythia28-genmatch-fix2.log" 2>&1 || fail=1

# ---------------------------------------------------------------- PPL (PRIORITY)
# eval_lm_ppl.py is run UNMODIFIED and on purpose. Its dtype probe calls os.path.join with no
# module-level `import os`, so the NameError is swallowed by `except Exception: pass` and every
# --impl hf run silently loads bfloat16 regardless of the checkpoint's declared torch_dtype.
# Verified by reproducing the probe. All 8 existing ppl16 HF rows were produced under that
# same bug, so "fixing" it here would make pythia the only fp16 row in a bf16 table -- a
# comparability break dressed up as a fix. The bug is reported in the ledger instead.
echo "=== [3/7] PPL, 7 corpora -> ppl16/pythia-2.8b.json ==="
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_lm_ppl.py --impl hf \
  --model_dir "$MODEL" --mode raw \
  --corpora wikitext2,wikitext103,pg19,lm1b,ptb,wmt14,xsum \
  --max_files 100 --output "$REPO/results/benchmarks/ppl16/pythia-2.8b.json" \
  > "$LOGD/ppl16-pythia-2.8b.log" 2>&1 || fail=1

# ---------------------------------------------------------------- bAbILong IN WINDOW
# The cells Pythia can actually be scored on. Same 100 items/cell and same flags as the
# briefed cells; only the context length differs, so 0k/1k/2k vs 4k/8k/16k is a controlled
# length sweep on one model rather than two protocols.
echo "=== [4/7] bAbILong IN-WINDOW 0k/1k/2k ==="
CUDA_VISIBLE_DEVICES=0 python3 -u "$BENCH" \
  --suite babilong --datasets qa1/0k,qa1/1k,qa1/2k,qa2/0k,qa2/1k,qa2/2k \
  --data_dir "$BABI" --model_dir "$MODEL" --arm raw \
  --output "$OUTW/${NAME}__babilong_inwindow.json" \
  --max_samples 200 --max_input_tokens 16384 --max_new_tokens 32 --seed 42 --device cuda \
  > "$LOGD/pythia28-babilong-inwindow.log" 2>&1 || fail=1

# ---------------------------------------------------------------- briefed long-context grid
# Run to produce the truncation audit, NOT to harvest scores. Every cell here is expected to
# come back with seen_fraction < 1.0 and is reported as "not measurable within this model's
# window". The result file is the evidence for that claim.
echo "=== [5/7] bAbILong briefed grid (expected OUT OF WINDOW) ==="
CUDA_VISIBLE_DEVICES=0 python3 -u "$BENCH" \
  --suite babilong --datasets qa1/4k,qa1/8k,qa1/16k,qa2/4k \
  --data_dir "$BABI" --model_dir "$MODEL" --arm raw \
  --output "$OUT/${NAME}__babilong.json" \
  --max_samples 200 --max_input_tokens 16384 --max_new_tokens 32 --seed 42 --device cuda \
  > "$LOGD/pythia28-babilong.log" 2>&1 || fail=1

echo "=== [6/7] RULER (expected OUT OF WINDOW: smallest shipped length is 4096) ==="
CUDA_VISIBLE_DEVICES=0 python3 -u "$BENCH" \
  --suite ruler \
  --datasets 4096/niah_single_1,4096/niah_multikey_1,4096/vt,4096/cwe,4096/fwe,4096/qa_1,16384/niah_single_1,16384/vt,16384/cwe \
  --data_dir "$RULER" --model_dir "$MODEL" --arm raw \
  --output "$OUT/${NAME}__ruler.json" \
  --max_samples 200 --max_input_tokens 16384 --max_new_tokens 128 --seed 42 --device cuda \
  > "$LOGD/pythia28-ruler.log" 2>&1 || fail=1

echo "=== [7/7] LongBench / InfiniteBench / MemEval (expected OUT OF WINDOW) ==="
CUDA_VISIBLE_DEVICES=0 python3 -u "$BENCH" \
  --suite longbench --datasets narrativeqa,qasper,multifieldqa_en,hotpotqa \
  --data_dir "$LB" --longbench_config_dir "$LBSPEC" --model_dir "$MODEL" --arm raw \
  --output "$OUT/${NAME}__longbench.json" \
  --max_samples 50 --max_input_tokens 32768 --max_new_tokens 512 --seed 42 --device cuda \
  > "$LOGD/pythia28-longbench.log" 2>&1 || fail=1

CUDA_VISIBLE_DEVICES=0 python3 -u "$BENCH" \
  --suite infinitebench \
  --datasets longbook_qa_eng,longbook_choice_eng,kv_retrieval,code_debug,math_find \
  --data_dir "$INF" --model_dir "$MODEL" --arm raw \
  --output "$OUT/${NAME}__infinitebench.json" \
  --max_samples 50 --max_input_tokens 32768 --max_new_tokens 128 --seed 42 --device cuda \
  > "$LOGD/pythia28-infinitebench.log" 2>&1 || fail=1

# MemEval's 50-row slice has a random floor of 0.220 and known scorer defects, so this is run
# for completeness and will be reported as uninformative at or below that floor regardless.
CUDA_VISIBLE_DEVICES=0 python3 -u "$BENCH" \
  --suite memeval --datasets longmemeval_oracle \
  --data_dir "$MEM" --model_dir "$MODEL" --arm raw \
  --output "$OUT/${NAME}__memeval.json" \
  --max_samples 50 --max_input_tokens 131072 --max_new_tokens 128 --seed 42 --device cuda \
  > "$LOGD/pythia28-memeval.log" 2>&1 || fail=1

echo "=== ALL STEPS DONE fail=$fail $(date -u) ==="
exit $fail

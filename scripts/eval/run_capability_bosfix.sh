#!/usr/bin/env bash
# One model, the 14 capability tasks of tab:capability, WITH the leading BOS restored.
#
# Generalisation of run_gemma_capability_bos.sh to the other models whose tokenizer adds a
# BOS that eval_generic_gen_match.py's add_special_tokens=False then strips. Measured
# empirically (two-way tokenisation, not config reading -- config reading gives a false
# negative on Llama, whose BOS comes from tokenizer.json's post-processor):
#
#   adds BOS: gemma-2-2b (already fixed), Llama-3.2-3B, gla-1.3B, rwkv6-world-3b, Nemotron-H-4B
#   no BOS to lose: rwkv7-world3, mamba-2.8b, cola-dlm, DCLM-1B, pythia-2.8b, olmo-2-1b,
#                   qwen2.5, phi-2, and the GPT-2-tokenizer diffusion rows (MDLM/SEDD/ELF/STAR-LDM)
#
# DIRECTION. Every model this repairs is a BASELINE we compare ourselves against, so the
# repair makes our own margin smaller. That is why it runs before the rows support any claim.
#
# CAVEAT for rwkv6-world-3b: its bos_token_id is 0, which is ALSO its eos_token_id and
# unk_token_id. Prepending it is therefore not unambiguously "restoring a BOS" -- it may be
# feeding the model an EOD marker. That model is run BOTH ways by the caller and the decision
# is left to the reader of the two numbers; see --tag.
#
# HELD FIXED, identical to the published rows: same task files, --max_samples 200,
# --max_new_tokens 32, --temperature 0.0 (greedy), --repetition_penalty 1.1 (default),
# --seed 42, same weights. Only the BOS differs.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <model_dir> <out_subdir> <log_name> [--no_bos]" >&2
  exit 2
fi

MODEL="$1"
OUT_SUBDIR="$2"
LOG_NAME="$3"
BOS_FLAG="--restore_bos"
if [[ "${4:-}" == "--no_bos" ]]; then
  BOS_FLAG=""
fi

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
export ROOT
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

TASKS=mmlu,obqa,siqa,story_cloze,hellaswag,piqa,arc_easy,arc_challenge,winogrande,sciq,commonsenseqa,lambada,race,squad
DATA=$ROOT/research/DiffRwkv/baseline/Cola-DLM/generate_task_data
OUT=$REPO/results/benchmarks/$OUT_SUBDIR
LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

if [[ ! -d "$MODEL" ]]; then
  echo "FATAL: model dir does not exist: $MODEL" >&2
  exit 3
fi

python3 scripts/eval/eval_generic_gen_match.py \
  --model_dir "$MODEL" \
  --task_data_dir "$DATA" \
  --output_dir "$OUT" \
  --tasks "$TASKS" \
  --max_samples 200 \
  --max_new_tokens 32 \
  --temperature 0.0 \
  --seed 42 \
  --trust_remote_code \
  ${BOS_FLAG} \
  > "$LOGS/${LOG_NAME}.log" 2>&1

# Do NOT echo "exit=$?": under `set -e` this line is only reached on success, so the stamp
# would always read 0. Job status is the authority.
echo "capability run completed -> $OUT"

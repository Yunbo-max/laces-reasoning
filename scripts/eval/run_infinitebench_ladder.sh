#!/usr/bin/env bash
# LENGTH-LADDER CONTROL for the InfiniteBench passkey/number_string result.
#
# The question this answers, and nothing else
# ------------------------------------------
# Both relay arms score 0.0000 (approx.) on passkey/number_string at 125k tokens while
# llama-3.2-3b and qwen2.5-1.5b score ~1.0 at the same budget with 0 tokens dropped. The
# mechanism claim is "length". But the supporting ladder used so far (RULER niah_single_1, 1.000
# at 4k and 16k) changes the PROMPT as well as the length: RULER ends in a completion cue
# ("The special magic number ... is"), our InfiniteBench prompt ends in a bare "Answer:".
# kv_retrieval/math_find do not fix this -- both are in INFINITEBENCH_CONTINUATION_TASKS and
# their own input ends in ":", so they are completion cues too.
#
# This runs the SAME rows, the SAME build_infinitebench_prompt template, the SAME scorer, at
# ~3.9k and ~15.7k tokens instead of 125k. Only length changes. Reading:
#   answers at 4k/16k, fails at 125k  -> length is the cause; the mechanism story stands
#   fails at 4k too                   -> the bare "Answer:" template is implicated and the
#                                        125k zero is NOT a length result
#
# The contexts are rebuilt by scripts/eval/make_passkey_length_ladder.py, which keeps the needle
# at its original relative depth (max shift 0.0042) and keeps it IN the context on 59/59 rows, so
# no cell here is unanswerable by construction. These are derived files, not the official dataset:
# every number remains "approx." and is labelled a length-ladder control.
set -euo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

OUT=$REPO/results/benchmarks/infbench_expand
LOGS=$REPO/logs/qz
LADDER=$REPO/data/ib_length_ladder
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

CHAMP=$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000
GLOBAL=$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000

# Budget stays 131072 so NOTHING is truncated: the contexts are already short, seen_fraction
# must come back 1.000 with rows_truncated 0. Keeping the budget identical to the 125k run also
# means the only changed variable is the context itself.
COMMON="--suite infinitebench --datasets passkey,number_string \
  --sample_stride 1 --max_samples 59 --max_input_tokens 131072 --max_new_tokens 128 \
  --seed 42 --device cuda"

# stride 1 here, NOT 10: make_passkey_length_ladder.py already applied stride 10 when it wrote
# these files, so they hold exactly the 59 depth bins. Striding again would take 6 rows.

run_relay () {  # name, ckpt, arm, step, gpu, rung
  local name=$1 ckpt=$2 arm=$3 step=$4 gpu=$5 rung=$6
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_open_context_bench.py $COMMON \
    --data_dir "$LADDER/$rung" \
    --ckpt_dir "$ckpt" --arm "$arm" --expected_step "$step" \
    --plan_steps 100 --cfg_scale 2.0 --blend 0.7 --diffusion_sampler ddpm \
    --output "$OUT/${name}__ladder${rung}__infinitebench.json" \
    > "$LOGS/ibexp-${name}-ladder${rung}.log" 2>&1 &
  echo "  [gpu$gpu] $name ($arm) rung=$rung -> $LOGS/ibexp-${name}-ladder${rung}.log"
}

run_hf () {     # name, model_dir, gpu, rung, extra...
  local name=$1 dir=$2 gpu=$3 rung=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/run_open_context_bench.py $COMMON \
    --data_dir "$LADDER/$rung" --model_dir "$dir" --arm raw "$@" \
    --output "$OUT/${name}__ladder${rung}__infinitebench.json" \
    > "$LOGS/ibexp-${name}-ladder${rung}.log" 2>&1 &
  echo "  [gpu$gpu] $name rung=$rung -> $LOGS/ibexp-${name}-ladder${rung}.log"
}

echo "=== length-ladder control: 4k and 16k, same prompt, same rows, same scorer ==="
run_relay champ-dynamic-30k "$CHAMP"  dynamic 30000 0 4k
run_relay global-dit-50k    "$GLOBAL" global  50000 1 4k
run_relay champ-dynamic-30k "$CHAMP"  dynamic 30000 2 16k
run_relay global-dit-50k    "$GLOBAL" global  50000 3 16k
# The frozen backbone is the dominant term in the mechanism story, so it gets the same ladder:
# if rwkv7-world3 answers at 4k under this template, the 125k zero is a length effect in the
# backbone and the relay merely fails to repair it.
run_hf rwkv7-world3 "$ROOT/models/RWKV7-Goose-World3-2.9B-HF" 4 4k  --decode loop
run_hf rwkv7-world3 "$ROOT/models/RWKV7-Goose-World3-2.9B-HF" 5 16k --decode loop
# One transformer at 4k as a positive control on the template itself: if llama scores ~1.0 here,
# the bare "Answer:" prompt is demonstrably answerable.
run_hf llama-3.2-3b "$ROOT/models/Llama-3.2-3B" 6 4k
wait
echo "=== ALL LADDER SHARDS DONE ==="
grep -h "^\[budget\]" "$LOGS"/ibexp-*-ladder*.log || true
ls -la "$OUT"/*ladder* || true

#!/usr/bin/env bash
# Batch-resolved decode timing + per-batch peak memory for tab:resources, all 9 rows.
#
# Produces the three quantities the published table needs and no existing file carries:
# a per-sequence throughput at a MEASURED concurrency, a batch-16 measurement (the
# paper's own 512/32 divisor, never timed: 0 of 25 files carry a bs16 key), and a peak
# memory attributable to one batch size (eval_efficiency.py resets the peak counter once
# at :409 and reads it at :454 with `for bs in (1,8)` between, so its single Peak GB is
# the max over both batches).
#
# Held identical to the published rows: same weights, same 512-token real-tokenizer
# input, same 64 timed decode steps, warm-up then median of 3. Only the batch sweep and
# the placement of the peak-memory reset differ.
set -euo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# $REPO must come first: the relay branch does `from scripts.eval.relay_utils import ...`,
# which needs the repo ROOT on the path, not just shared_pylibs. Leaving it out is how the
# first sweep produced 7 baseline files and two `ModuleNotFoundError: No module named
# 'scripts'` failures while the job still reported job_succeeded -- the `|| echo FAILED`
# in the wrappers below swallows the exit code, so job status cannot be the authority here.
export PYTHONPATH=$REPO:$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

OUT=$REPO/results/benchmarks/efficiency_total
LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

# Clear prior rows: the completeness guard at the bottom counts *.json, and a file left over
# from a partial sweep would satisfy that count without having been measured this run.
# Archive rather than delete so the first sweep's seven baseline rows stay comparable.
if compgen -G "$OUT/*.json" > /dev/null; then
  STAMP=$(date +%Y%m%d-%H%M%S)
  mkdir -p "$OUT/prev-$STAMP"
  mv "$OUT"/*.json "$OUT/prev-$STAMP"/
  echo "archived $(ls -1 "$OUT/prev-$STAMP" | wc -l) prior row(s) -> $OUT/prev-$STAMP"
fi

BATCHES=1,8,16
GEN=512
E=scripts/eval/eval_efficiency_total.py
FAILED=0

hf () {  # name model_dir
  echo "=== hf $1"
  if python3 $E --impl hf --model_dir "$2" --batches $BATCHES --gen_len $GEN \
      --output "$OUT/$1.json"; then :; else echo "FAILED: $1"; FAILED=$((FAILED+1)); fi
}
relay () {  # name ckpt_dir
  echo "=== relay $1"
  if python3 $E --impl relay --ckpt_dir "$2" --batches $BATCHES --gen_len $GEN \
      --output "$OUT/$1.json"; then :; else echo "FAILED: $1"; FAILED=$((FAILED+1)); fi
}

# The seven baselines. Every path below was cross-checked against the `model_dir` recorded
# in that model's own capability run_summary.json, 7/7 MATCH -- not guessed from the name.
# Two first guesses were wrong and are recorded here so they are not re-introduced:
# `models/mamba-2.8b-hf` and `models/rwkv7-world3` do not exist.
hf qwen2.5-3b      "$ROOT/models/Qwen2.5-3B"
hf llama-3.2-3b    "$ROOT/models/Llama-3.2-3B"
hf phi-2           "$ROOT/models/phi-2"
hf gemma-2-2b      "$ROOT/models/gemma-2-2b"
hf mamba-2.8b      "$ROOT/models/mamba-2.8b"
hf rwkv6-world     "$ROOT/models/rwkv6-world-3b"
hf rwkv7-world3    "$ROOT/models/RWKV7-Goose-World3-2.9B-HF"

# Our two arms. ckpt paths taken from the ppl.json of the same-named result dirs, which
# DO record `ckpt`; the efficiency files do not, so this run writes it into its output.
relay champ-dynamic-30k \
  "$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000"
relay global-dit-50k \
  "$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000"

echo "efficiency_total sweep finished -> $OUT"
ls -1 "$OUT"

# Job status is NOT the authority: the wrappers above catch each failure so the sweep
# completes all nine rows, which means a partial sweep still exits 0 unless we check here.
EXPECT=9
GOT=$(ls -1 "$OUT"/*.json 2>/dev/null | wc -l)
echo "rows written: $GOT / $EXPECT   (failures this sweep: $FAILED)"
if [[ "$FAILED" -ne 0 || "$GOT" -lt "$EXPECT" ]]; then
  echo "SWEEP INCOMPLETE" >&2
  exit 4
fi
echo "SWEEP COMPLETE"

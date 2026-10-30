#!/bin/bash
# RULER expansion: complete the 13-task x 3-length grid (binding-capacity curve).
#
# Purpose. The capacity-vs-bindings law says our score degrades with the number of
# discrete items that must be held in the recurrent state at once, and is ~independent of
# context length. Existing evidence is 3 gold-count points (1 / 3 / 10) at 4k and 16k.
# This driver fills EVERY remaining (task, length) cell so the curve has all 13 tasks at
# all three cached lengths, and so 8192 -- never used before -- becomes length-axis
# evidence on the exact tasks the law was built from.
#
# n = 100 for every new cell. Rationale: the already-published cells store 200 rows each
# with row ids 0..199 taken as a parquet prefix, so their n=100 value is recoverable
# EXACTLY from the archived predictions with no GPU time (verified: |delta| <= 0.04 on
# every cell except qa_1, which moves +0.04/+0.11 because 100 rows of a 3-gold QA task is
# a coarse grid). Halving n therefore costs nothing in comparability and buys the full
# 39-cell grid instead of 21 cells, for ~15 GPU-hours instead of ~21.
#
# Protocol is inherited verbatim from the published cells (checked against the `config`
# block of results/benchmarks/longctx/champ-dynamic-30k__ruler.json, not from prose):
#   --max_input_tokens 16384 --max_new_tokens 128 --plan_steps 100 --cfg_scale 2.0
#   --blend 0.7 --diffusion_sampler ddpm --seed 42
# Per-row `max_new_tokens` from the parquet still caps each task below 128 (qa 32, vt 30,
# fwe 50, cwe 120) -- that happens inside the runner and must not be overridden here.
#
# Sharding: 8 GPUs in one container. champ (the slowest arm, ~7.3 s/sample) gets 4 shards,
# global (~5.1 s/sample) gets 3, and the two attention baselines (~1.3 s/sample) share the
# last GPU sequentially. Shards are balanced by n_samples x length so wall clock is ~2.5 h
# rather than the ~7 h a single-GPU champ pass would take.
#
# mamba-2.8b and rwkv6 are deliberately EXCLUDED: measured at 10-40 s/sample they would
# cost ~10 GPU-h each for the 30 new cells alone, and neither is needed to show the law is
# specific to us. phi-2 (2048-token positional window) and gemma-2-2b (8192) are excluded
# at 8192/16384 for the same reason the README already forbids publishing their 16k RULER
# cells: those are truncation artefacts, not capabilities.
set -uo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning
export HF_HOME=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# ${PYTHONPATH:+:$PYTHONPATH}, never a bare $PYTHONPATH: PYTHONPATH is UNSET in this image
# and under `set -u` a bare reference aborts the shell before any log redirect exists,
# which kills the job with no log at all.
export PYTHONPATH=/inspire/hdd/global_user/zhangjiaquan-253108540222/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

DATA=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/datasets--simonjegou--ruler/snapshots/24adceac8a0e6532936e8d721cd9e9084d2e4686
OUT=$REPO/results/benchmarks/ruler_expand
LOGD=$REPO/logs/qz
mkdir -p "$OUT" "$LOGD"
cd "$REPO"

NSAMP=100
M=/inspire/hdd/global_user/zhangjiaquan-253108540222/models
CK=$REPO/outputs_dynamic_basis
CHAMP=$CK/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000
GLOBAL=$CK/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000

# --- cell inventory -------------------------------------------------------------------
# Already measured at n=200 (do NOT rerun; re-derived at n=100 from archived rows):
#   4096: niah_single_1 niah_multikey_1 vt cwe fwe qa_1     16384: niah_single_1 vt cwe
NEW_4096="4096/niah_multikey_2,4096/niah_multikey_3,4096/niah_multiquery,4096/niah_multivalue,4096/niah_single_2,4096/niah_single_3,4096/qa_2"
NEW_8192_A="8192/cwe,8192/fwe,8192/niah_multikey_1"
NEW_8192_B="8192/niah_multikey_2,8192/niah_multikey_3,8192/niah_multiquery,8192/niah_multivalue,8192/niah_single_1"
NEW_8192_C="8192/niah_single_2,8192/niah_single_3,8192/qa_1,8192/qa_2,8192/vt"
NEW_16384="16384/fwe,16384/niah_multikey_1,16384/niah_multikey_2,16384/niah_multikey_3,16384/niah_multiquery,16384/niah_multivalue,16384/niah_single_2,16384/niah_single_3,16384/qa_1,16384/qa_2"
ALL_NEW="$NEW_4096,$NEW_8192_A,$NEW_8192_B,$NEW_8192_C,$NEW_16384"   # 30 cells

run_relay() {   # shard_name gpu ckpt arm expected_step datasets
  local name=$1 gpu=$2 ckpt=$3 arm=$4 step=$5 ds=$6
  echo "[launch] $name on GPU $gpu (relay arm=$arm step=$step)"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/eval/run_open_context_bench.py \
      --suite ruler --datasets "$ds" --data_dir "$DATA" \
      --ckpt_dir "$ckpt" --arm "$arm" --expected_step "$step" \
      --output "$OUT/${name}__ruler_expand.json" \
      --max_samples "$NSAMP" --max_input_tokens 16384 --max_new_tokens 128 \
      --plan_steps 100 --cfg_scale 2.0 --blend 0.7 --diffusion_sampler ddpm \
      --seed 42 --device cuda \
      > "$LOGD/ruler-expand-${name}.log" 2>&1
  echo "[done] $name exit=$?"
}

run_hf() {      # shard_name gpu model_dir datasets
  local name=$1 gpu=$2 path=$3 ds=$4
  echo "[launch] $name on GPU $gpu (hf baseline)"
  # --decode hf is correct for full-attention transformers and matches the published
  # cells (their `config` predates the flag, whose default is hf). The `loop` path is
  # required only for fla recurrent models (rwkv7/gla), which this expansion does not run.
  CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/eval/run_open_context_bench.py \
      --suite ruler --datasets "$ds" --data_dir "$DATA" \
      --model_dir "$path" --arm raw --decode hf \
      --output "$OUT/${name}__ruler_expand.json" \
      --max_samples "$NSAMP" --max_input_tokens 16384 --max_new_tokens 128 \
      --seed 42 --device cuda \
      > "$LOGD/ruler-expand-${name}.log" 2>&1
  echo "[done] $name exit=$?"
}

echo "=== RULER expansion: 30 new cells x 4 models, n=$NSAMP, started $(date -u) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

run_relay champ-4096-8192a 0 "$CHAMP"  dynamic 30000 "$NEW_4096,$NEW_8192_A"   &
run_relay champ-8192b      1 "$CHAMP"  dynamic 30000 "$NEW_8192_B"             &
run_relay champ-8192c      2 "$CHAMP"  dynamic 30000 "$NEW_8192_C"             &
run_relay champ-16384      3 "$CHAMP"  dynamic 30000 "$NEW_16384"              &
run_relay global-4096-8192a 4 "$GLOBAL" global 50000 "$NEW_4096,$NEW_8192_A"   &
run_relay global-8192bc    5 "$GLOBAL" global  50000 "$NEW_8192_B,$NEW_8192_C" &
run_relay global-16384     6 "$GLOBAL" global  50000 "$NEW_16384"              &
(
  # Both attention baselines on one GPU, sequentially: together they are ~2.5 h, the same
  # as the slowest relay shard, so nothing is gained by contending for a single device.
  run_hf llama-3.2-3b 7 "$M/Llama-3.2-3B" "$ALL_NEW"
  run_hf qwen2.5-3b   7 "$M/Qwen2.5-3B"   "$ALL_NEW"
) &
wait
echo "=== all shards finished $(date -u) ==="

echo "=== per-shard cell scores ==="
python3 - <<'PY'
import json, glob, os
base = '/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/results/benchmarks/ruler_expand'
for f in sorted(glob.glob(os.path.join(base, '*__ruler_expand.json'))):
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f, 'UNREADABLE', e); continue
    print('==', os.path.basename(f).replace('__ruler_expand.json', ''),
          'elapsed_h=%.2f' % (d.get('elapsed_seconds', 0) / 3600))
    for k, v in d['aggregates'].items():
        print('   %-26s n=%-4s score=%.4f' % (k, v.get('n'), v.get('score', -1)))
PY
echo "=== ALL DONE ==="

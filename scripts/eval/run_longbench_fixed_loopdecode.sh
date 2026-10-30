#!/usr/bin/env bash
# LongBench for the two fla recurrent baselines through the CORRECT decode path.
#
# Why this is a separate run and not a re-do of the sweep: for rwkv7/gla,
# `model.generate()` is wrong even after patch_fla_recurrent_generate. Measured token
# agreement against a hand-written single-token loop on 20 bAbILong qa1/4k prompts
# (results/benchmarks/falsify_zero/rwkv7-fingerprint.json, gla-fingerprint.json):
#
#   rwkv7  patched_vs_loop  2/20 at every prefix length 1..32;  score unpatched 0% / patched 5% / loop 40%
#   gla    patched_vs_loop 14/20 at prefix 32;                  score unpatched 5% / patched 20% / loop 20%
#
# So the sweep's rwkv7 row (0.5 F1 on hotpotqa, 48/50 rows exactly 0.000, repetition
# collapse) is OUR artefact, not RWKV-7's capability, and must not be published as a result.
# The loop is also the shape LACESRuntime/GlobalRuntime use for our own arms, so this is the
# protocol-matched comparison rather than a handicap.
#
# Everything else -- datasets, sample count, seed, truncation budget, scorer -- is identical
# to run_longbench_fixed_sweep.sh, so `_loop` vs the sweep isolates the decode path alone.
set -euo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# A bare $PYTHONPATH under `set -u` kills the job with no log at all.
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

OUT=$REPO/results/benchmarks/longctx_fix
LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

DATASETS=hotpotqa,2wikimqa,musique,narrativeqa,multifieldqa_en,passage_retrieval_en,gov_report,qmsum,multi_news
COMMON="--suite longbench --datasets $DATASETS \
  --data_dir $ROOT/longbench_local/extract/data \
  --longbench_config_dir $ROOT/longbench_local/pinned_spec \
  --max_samples 50 --max_input_tokens 32768 --max_new_tokens 512 --seed 42 --device cuda \
  --arm raw --decode loop"

CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/run_open_context_bench.py $COMMON \
  --model_dir "$ROOT/models/RWKV7-Goose-World3-2.9B-HF" \
  --output "$OUT/rwkv7-world3-loop__longbench.json" \
  > "$LOGS/lbfix-rwkv7-world3-loop-longbench.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 scripts/eval/run_open_context_bench.py $COMMON \
  --model_dir "$ROOT/hf_cache/hub/models--fla-hub--gla-1.3B-100B/snapshots/46b15820a4df269e99aed9d709e017677c15d24b" \
  --output "$OUT/gla-1.3b-loop__longbench.json" \
  > "$LOGS/lbfix-gla-1.3b-loop-longbench.log" 2>&1 &

wait
echo "=== LOOP-DECODE RUN DONE ==="
grep -h "^\[budget\]" "$LOGS"/lbfix-rwkv7-world3-loop-longbench.log \
                      "$LOGS"/lbfix-gla-1.3b-loop-longbench.log || true

#!/usr/bin/env bash
# Falsification sweep for the near-zero long-context baseline rows.
# One 8-GPU node; each GPU runs one independent probe config, all writing their own JSON.
#
# The question per model: is the published ~0 a capability limit or OUR artefact?
#   phi-2        -> positional overflow (window 2048, fed 3710). Re-run inside the window.
#   rwkv7/gla    -> fla generate bug. Fingerprint (unpatched vs patched vs loop), then re-run.
#   gemma-2-2b   -> window 8192, fed 15k at the 16k cell; also never got its BOS token.
#   qwen2.5-3b   -> positive control: must reproduce the published number.
set -uo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning
MODELS=/inspire/hdd/global_user/zhangjiaquan-253108540222/models
BABI=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/datasets--RMT-team--babilong/snapshots/ee0d588794c7ac098062ee0d247c733d62e94fe2/data
GLA=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/models--fla-hub--gla-1.3B-100B/snapshots/46b15820a4df269e99aed9d709e017677c15d24b
RWKV=$MODELS/RWKV7-Goose-World3-2.9B-HF
OUT=$REPO/results/benchmarks/falsify_zero
LOG=$REPO/logs/qz/falsify

export HF_HOME=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# A bare $PYTHONPATH under `set -u` kills the job with no log.
export PYTHONPATH=/inspire/hdd/global_user/zhangjiaquan-253108540222/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT" "$LOG"
cd "$REPO"
P=scripts/eval/falsify_zero_baselines.py

run() {  # run <gpu> <tag> <args...>
  local gpu=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python3 "$P" "$@" \
      --output "$OUT/${tag}.json" > "$LOG/${tag}.log" 2>&1 &
  echo "  launched gpu=$gpu tag=$tag pid=$!"
}

echo "=== launching falsification probes ==="

# ---- GPU 0: phi-2 inside its real 2048 window (reserve 32 for generation => 2016) --------
# The published cell fed 3710 tokens into a 2048-position table. This is the same task with
# less haystack: a legitimate, reportable protocol difference, not a like-for-like number.
run 0 phi2-window --mode rerun --model_dir "$MODELS/phi-2" --data_dir "$BABI" \
    --datasets qa1/0k,qa1/1k,qa1/2k,qa1/4k --budget window --decode hf --max_samples 100

# ---- GPU 1: phi-2 unchanged budget, as the reproduction control ------------------------
run 1 phi2-repro --mode rerun --model_dir "$MODELS/phi-2" --data_dir "$BABI" \
    --datasets qa1/4k --budget 16384 --decode hf --max_samples 100

# ---- GPU 2: rwkv7 fingerprint -- is the fla bug active in the published run? ------------
run 2 rwkv7-fingerprint --mode fingerprint --model_dir "$RWKV" --data_dir "$BABI" \
    --dataset qa1/4k --budget_int 16384 --max_samples 20

# ---- GPU 3: rwkv7 re-run through the correct decode path (single-token loop) ------------
run 3 rwkv7-loop --mode rerun --model_dir "$RWKV" --data_dir "$BABI" \
    --datasets qa1/4k,qa1/8k,qa1/16k,qa2/4k --budget 16384 --decode loop --max_samples 100

# ---- GPU 4: gla fingerprint ------------------------------------------------------------
run 4 gla-fingerprint --mode fingerprint --model_dir "$GLA" --data_dir "$BABI" \
    --dataset qa1/4k --budget_int 16384 --max_samples 20

# ---- GPU 5: gla re-run through the correct decode path, WITH its BOS token --------------
run 5 gla-loop-bos --mode rerun --model_dir "$GLA" --data_dir "$BABI" \
    --datasets qa1/4k,qa1/8k,qa1/16k,qa2/4k --budget 16384 --decode loop --add_bos \
    --max_samples 100

# ---- GPU 6: gemma-2-2b inside its real 8192 window, with its BOS token ------------------
# Published 16k cell fed ~15k tokens to an 8192-window model => 0.0. Two artefacts at once.
run 6 gemma-window-bos --mode rerun --model_dir "$MODELS/gemma-2-2b" --data_dir "$BABI" \
    --datasets qa1/4k,qa1/8k --budget window --decode hf --add_bos --max_samples 100

# ---- GPU 7: positive control + gemma BOS-only isolation --------------------------------
(
  CUDA_VISIBLE_DEVICES=7 python3 "$P" --mode rerun --model_dir "$MODELS/Qwen2.5-3B" \
      --data_dir "$BABI" --datasets qa1/4k,qa1/16k --budget 16384 --decode hf \
      --max_samples 100 --output "$OUT/qwen-control.json" > "$LOG/qwen-control.log" 2>&1
  CUDA_VISIBLE_DEVICES=7 python3 "$P" --mode rerun --model_dir "$MODELS/gemma-2-2b" \
      --data_dir "$BABI" --datasets qa1/4k --budget 16384 --decode hf --add_bos \
      --max_samples 100 --output "$OUT/gemma-bos-only.json" > "$LOG/gemma-bos-only.log" 2>&1
) &
echo "  launched gpu=7 tag=qwen-control+gemma-bos-only pid=$!"

wait
echo "=== all probes finished ==="
for f in "$OUT"/*.json; do
  echo "--- $f"
  python3 -c "
import json,sys
d=json.load(open('$f'))
if d.get('mode')=='rerun':
    print('   budget=',d['effective_budget'],'ceiling=',d['positional_ceiling'])
    for k,v in d['aggregates'].items(): print(f'   {k}: {v[\"score\"]*100:.1f}% (n={v[\"n\"]})')
else:
    print('   scores:', d['scores_pct'])
    print('   agreement:', json.dumps(d['agreement_by_prefix_len']))
" 2>&1
done

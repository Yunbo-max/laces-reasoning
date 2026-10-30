#!/usr/bin/env bash
# llama-3.2-3b BOS test -- closing the one gap in the falsification report.
#
# Why this run exists. My falsification sweep found gemma-2-2b's entire +40pp came from the
# BOS token that `add_special_tokens=False` silently drops. I checked that defect only on the
# near-zero models (gemma, gla) because the brief was about zeros. A peer agent pointed out
# llama-3.2-3b drops BOS too, and I verified it empirically:
#     llama-3.2-3b  tokenizer_config add_bos_token=True, attr False, DROPS id 128000
# llama is our STRONGEST baseline (60/50/44/26), so if its row moves the ranking moves --
# and it would move AGAINST us. That makes it a required check, not an optional one.
#
# This is a CLEANER experiment than gemma's: max_position_embeddings=131072 (rope llama3,
# original 8192 x factor 32), so at a 16384 budget there is NO positional ceiling and NO
# truncation confound. The only variable is the BOS token.
#
# GPU 0-3: the four published cells WITH BOS restored.
# GPU 4-7: the same cells WITHOUT BOS = reproduction control. These must land on the
#          published 60.0/50.0/44.0/26.0, otherwise the comparison is meaningless and any
#          delta I report is measuring something other than the BOS token.
set -uo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning
LLAMA=/inspire/hdd/global_user/zhangjiaquan-253108540222/models/llama-3.2-3b
BABI=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/datasets--RMT-team--babilong/snapshots/ee0d588794c7ac098062ee0d247c733d62e94fe2/data
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

echo "=== llama-3.2-3b BOS test: 4 cells x {bos, no-bos control} ==="

# One cell per GPU so each of the 8 configs is independent; --decode hf because llama is a
# plain transformer, the fla recurrent-cache bug cannot apply to it.
i=0
for spec in qa1/4k qa1/8k qa1/16k qa2/4k; do
  tag=${spec/\//_}
  run "$i" "llama-bos-${tag}" --mode rerun --model_dir "$LLAMA" --data_dir "$BABI" \
      --datasets "$spec" --budget 16384 --decode hf --add_bos --max_samples 100
  i=$((i + 1))
done
for spec in qa1/4k qa1/8k qa1/16k qa2/4k; do
  tag=${spec/\//_}
  run "$i" "llama-nobos-${tag}" --mode rerun --model_dir "$LLAMA" --data_dir "$BABI" \
      --datasets "$spec" --budget 16384 --decode hf --max_samples 100
  i=$((i + 1))
done

wait
echo "=== done ==="
python3 - <<'PY'
import json, pathlib
out = pathlib.Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/"
                   "results/benchmarks/falsify_zero")
published = {"qa1_4k": 60.0, "qa1_8k": 50.0, "qa1_16k": 44.0, "qa2_4k": 26.0}
print(f"{'cell':9} {'published':>9} {'no-bos ctl':>10} {'with-bos':>9} {'delta':>7} {'ctl_ok':>7}")
for cell, pub in published.items():
    spec = cell.replace("_", "/")
    tag = spec.replace("/", "_")
    def score(kind):
        f = out / f"llama-{kind}-{tag}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        agg = d.get("aggregates", {})
        return next(iter(agg.values()))["score"] * 100 if agg else None
    nb, wb = score("nobos"), score("bos")
    if nb is None or wb is None:
        print(f"{cell:9} {pub:9.1f} {'MISSING':>10}")
        continue
    ok = abs(nb - pub) < 1e-9
    print(f"{cell:9} {pub:9.1f} {nb:10.1f} {wb:9.1f} {wb - nb:+7.1f} {str(ok):>7}")
PY

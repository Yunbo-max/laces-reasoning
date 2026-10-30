#!/usr/bin/env bash
# mamba-2.8b, ONE RULER 16384 cell, ONE GPU. Takes the dataset as $1.
#
# WHY THIS EXISTS. The 8-GPU driver (run_ruler_mamba_sharded.sh) put all three 16384 cells in
# one job. That job (job-aa1c7ca5) was STOPPED EXTERNALLY at 12:19:59 UTC at 3.01 h of its 12 h
# cap -- not a timeout, not a crash: the three lane logs end on clean `score=0.000` lines with no
# traceback, within 51 s of each other. run_open_context_bench.py writes its JSON only once, at
# cell end, and has no --offset/--resume flag, so 404 completed rows (123+131+150) were lost with
# zero salvage. One cell per job means the next external kill costs one cell, not three.
#
# The six 4096 cells all landed before that kill and are already final
# (results/benchmarks/longctx_fix/mamba-2.8b__ruler_4k.json, 36.37, 1200 rows, gated merge exit 0).
# This script only rebuilds the three missing 16384 cells.
#
# DECODE: mamba is NOT an fla model. It is transformers-native MambaForCausalLM; the loader prints
# "model_type='mamba' is native to transformers; fla not imported" and "fla recurrent-generate
# patch applied: False". The fla 0.5.0 + transformers 5.3.0 recurrent-cache bug therefore cannot
# reach it, so the hf decode path is correct here and is what the landed 4k shards used
# (decode_path='hf', decode_is_trustworthy=True). Only rwkv7/gla/gated_deltanet need --decode loop.
# Verified from the previous run's own stamps, not assumed.
#
# PROTOCOL: byte-identical to the 4k shards and to the 8 models already in the table -- same
# --max_samples 200, --max_input_tokens 16384, --max_new_tokens 128, --seed 42, same snapshot,
# same --arm raw. Only the work split differs, so the numbers stay comparable.
#
# BUDGET: effective_input_budget 16384, truncation_reason "cli_cap_only (recurrent model: no
# positional ceiling)". Mamba's measured 16k seen fractions are niah_single_1 1.0000 (0/200
# pinned), cwe 1.0000 (0/200), vt 0.9642 (200/200) -- the best-conditioned 16k profile in that
# column after Llama.
#
# RATE: 86.7 s/row measured over the full 09:44-11:50 window of the killed job => ~4.8 h for 200
# rows. Cap is set to 11 h for headroom on a shared node.
set -euo pipefail

DS="${1:?usage: $0 <niah_single_1|vt|cwe>}"

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

OUT=$REPO/results/benchmarks/longctx_mamba_ruler_shards
LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

DATA=$ROOT/hf_cache/hub/datasets--simonjegou--ruler/snapshots/24adceac8a0e6532936e8d721cd9e9084d2e4686
MODEL=$ROOT/models/mamba-2.8b

TAG="16384-${DS}"
python3 scripts/eval/run_open_context_bench.py \
  --suite ruler --data_dir "$DATA" --model_dir "$MODEL" --arm raw \
  --max_samples 200 --max_input_tokens 16384 --max_new_tokens 128 --seed 42 --device cuda \
  --datasets "16384/${DS}" \
  --output "$OUT/mamba-2.8b__${TAG}__ruler.json" \
  > "$LOGS/rulermamba2-${TAG}.log" 2>&1

# NOTE: do NOT print "exit=$?" here. Under `set -e` this line is only reached when the run
# succeeded, so $? is always 0 and the stamp would be meaningless -- exactly the kind of
# always-true success marker that has already misled this project once. Job status is the
# authority: if the python call fails, set -e aborts and the job is marked failed.
echo "cell $TAG completed -> $OUT/mamba-2.8b__${TAG}__ruler.json"
ls -l "$OUT/mamba-2.8b__${TAG}__ruler.json"
python3 - <<EOF
import json
d = json.load(open("$OUT/mamba-2.8b__${TAG}__ruler.json"))
ag = d.get("aggregates") or {}
print("STAMPS decode_path=", d.get("decode_path"), "trust=", d.get("decode_is_trustworthy"))
print("CELLS", {k: (v.get("score") if isinstance(v, dict) else v) for k, v in ag.items()})
print("ROWS", len(d.get("rows") or []))
EOF

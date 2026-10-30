#!/usr/bin/env bash
# Re-run the TWO cells hit by the stop-token resolution defect: the rwkv7-world3 baseline and
# OUR OWN global-dit-50k arm. Both LongBench, both at their published protocol.
#
# WHY. `greedy_loop` (baselines) and the relay runtimes (our arms) were both handed only
# `tokenizer.eos_token_id`. RWKV7-Goose-World3-2.9B-HF -- which is the baseline AND the frozen
# decoder inside our relay checkpoints -- declares three different eos values:
#
#   tokenizer.eos_token_id  = 65530   <- what was used; this id NEVER appears in any output
#   generation_config        = 0      <- '<|rwkv_tokenizer_end_of_text|>', the real one
#   config.json              = 2      <- '!'
#
# So the stop check could not fire and all 450 rows ran to their full budget. Measured id-0
# occurrences INSIDE published generations, and the offline rescore of the truncated prefix:
#
#   cell                      rows  id0   changed        macro          delta
#   rwkv7-world3 (baseline)    450  141   82 (67u/15d)   0.0955->0.1197  +0.0242
#   global-dit-50k (OUR ARM)   450   79   58 (55u/ 3d)   0.0955->0.1198  +0.0243
#   champ-dynamic-30k (ours)   450    0    0             0.0901->0.0901  +0.0000   <- not rerun
#
# champ-dynamic already stops on its own (33 distinct lengths, 0 rows contain id 0), so it is
# unaffected and is NOT re-run. gla is unaffected for a different reason: its tokenizer eos
# '</s>' resolves to id 2, matching its config and generation_config, so its loop had the right
# id all along (and it emits it in 0/2950 rows).
#
# The other four suites are provably unaffected and are NOT re-run: RULER, bAbILong and
# InfiniteBench use containment/recall metrics, so trailing junk cannot remove a match. Verified
# empirically rather than assumed, by rescoring every row of both arms: RULER 0/1800 rows change
# (despite id 0 in 390 global-dit / 671 rwkv7 rows), bAbILong 0/400, InfiniteBench 0/250.
#
# DIRECTION -- read this before quoting any number. The defect hit a competitor AND our own arm
# by almost exactly the same amount (+0.0242 vs +0.0243), so the rwkv7-vs-global-dit LongBench
# comparison is very nearly unchanged: 0.0955 vs 0.0955 before, ~0.1197 vs ~0.1198 after. Fixing
# it does NOT hand us a win; the honest statement is that both arms were understated and the gap
# was always a wash. Anyone tempted to re-run only the baseline would produce a biased half-fix.
#
# PROTOCOL. Every axis is copied verbatim from each published cell's own `config` block so the
# only difference is the stop ids: max_samples 50, max_input_tokens 32768, max_new_tokens 512,
# seed 42, same 9 datasets, same data_dir and pinned_spec. The baseline keeps `--arm raw
# --decode loop`; our arm keeps `--arm global --expected_step 50000` with no --decode (the relay
# path ignores it). Expected in both logs: `[load] resolved stop ids: [0, 2, 65530]`.
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

# Staging dir, exactly as for the first pass. Nothing is copied to results/benchmarks/longctx/
# until the gate script has verified completeness, decode_path and the resolved stop ids.
OUT=$REPO/results/benchmarks/longctx_eosfix
LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

DATASETS=hotpotqa,2wikimqa,musique,narrativeqa,multifieldqa_en,passage_retrieval_en,gov_report,qmsum,multi_news
COMMON="--suite longbench --datasets $DATASETS \
  --data_dir $ROOT/longbench_local/extract/data \
  --longbench_config_dir $ROOT/longbench_local/pinned_spec \
  --max_samples 50 --max_input_tokens 32768 --max_new_tokens 512 --seed 42 --device cuda"

CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/run_open_context_bench.py $COMMON \
  --arm raw --decode loop \
  --model_dir "$ROOT/models/RWKV7-Goose-World3-2.9B-HF" \
  --output "$OUT/rwkv7-world3__longbench.json" \
  > "$LOGS/eosfix-rwkv7-world3-longbench.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 scripts/eval/run_open_context_bench.py $COMMON \
  --arm global --expected_step 50000 \
  --ckpt_dir "$REPO/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000" \
  --output "$OUT/global-dit-50k__longbench.json" \
  > "$LOGS/eosfix-global-dit-50k-longbench.log" 2>&1 &

wait
echo "=== EOS-FIX RE-RUN DONE ==="
# The provenance axes that live only in the log, plus the new fourth one.
for L in eosfix-rwkv7-world3-longbench eosfix-global-dit-50k-longbench; do
  echo "--- $L ---"
  grep -h "^\[load\] resolved stop ids" "$LOGS/$L.log" || echo "MISSING stop-ids banner"
  grep -h "^\[budget\]" "$LOGS/$L.log" || true
  grep -h "patch applied" "$LOGS/$L.log" || true
done

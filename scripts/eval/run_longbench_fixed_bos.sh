#!/usr/bin/env bash
# Wave 3: re-run the three models whose tokenizer prepends BOS, now that the prepend actually
# fires.
#
# Why these three and only these three: truncate_ids must tokenize with
# add_special_tokens=False (the LongBench template already carries the task framing, and
# letting the tokenizer inject specials would shift the middle-truncation offsets). That also
# drops the leading BOS. Measured at an unchanged budget, the single token is worth +40pp on
# bAbILong qa1/4k for gemma (13.0 -> 53.0,
# results/benchmarks/falsify_zero/gemma-bos-only.json), so the earlier cells for these models
# are our artefact, not their capability.
#
# The first version of the fix was silently inert: it read `add_bos_token` off the loaded
# tokenizer, which reports False on this machine's snapshots even where tokenizer_config.json
# says `"add_bos_token": true`. Measured 2026-09-21 -- gemma-2-2b (GemmaTokenizer) config true
# -> attr False; gla-1.3B (LlamaTokenizer) config true -> attr False; llama-3.2-3B
# (TokenizersBackend) no key at all -> attr False but DOES prepend 128000. So the flag read
# returned False for all three models it existed to catch. tokenizer_wants_bos now probes
# add_special_tokens True-vs-False instead, which is verified to agree with HF's own
# tokenization for all 8 baselines. qwen2.5-{3b,1.5b}, phi-2, rwkv7 and mamba genuinely have
# no leading BOS and are unaffected -- re-running them would only burn GPU hours.
#
# Everything else -- datasets, sample count, seed, max_new_tokens, budget policy, scorer -- is
# identical to run_longbench_fixed_sweep.sh, so the BOS token is the only difference.
#
# gla-1.3b additionally uses --decode loop, matching run_longbench_fixed_loopdecode.sh, so its
# cell stays comparable to the rwkv7 cell we publish. (For gla the decode path is worth little
# on its own -- hotpotqa 5.8 hf vs 6.3 loop -- but mixing paths inside one table is not
# defensible.)
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
  --arm raw"

CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/run_open_context_bench.py $COMMON \
  --model_dir "$ROOT/models/gemma-2-2b" \
  --output "$OUT/gemma-2-2b-bos__longbench.json" \
  > "$LOGS/lbfix-gemma-2-2b-bos-longbench.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 scripts/eval/run_open_context_bench.py $COMMON \
  --model_dir "$ROOT/models/Llama-3.2-3B" \
  --output "$OUT/llama-3.2-3b-bos__longbench.json" \
  > "$LOGS/lbfix-llama-3.2-3b-bos-longbench.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 scripts/eval/run_open_context_bench.py $COMMON --decode loop \
  --model_dir "$ROOT/hf_cache/hub/models--fla-hub--gla-1.3B-100B/snapshots/46b15820a4df269e99aed9d709e017677c15d24b" \
  --output "$OUT/gla-1.3b-bos-loop__longbench.json" \
  > "$LOGS/lbfix-gla-1.3b-bos-loop-longbench.log" 2>&1 &

wait
echo "=== BOS RE-RUN DONE ==="
grep -h "^\[budget\]" "$LOGS"/lbfix-gemma-2-2b-bos-longbench.log \
                      "$LOGS"/lbfix-llama-3.2-3b-bos-longbench.log \
                      "$LOGS"/lbfix-gla-1.3b-bos-loop-longbench.log || true

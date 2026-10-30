#!/usr/bin/env bash
# mamba-2.8b: the missing LongBench cell (printed `--` in tab:longcontext) plus the bAbILong
# denominator top-up (its published cells are 50 items where every other row has 100).
#
# WHY SHARDED, ONE DATASET PER GPU. This image has no mamba fast kernels (mamba_ssm and
# causal_conv1d both ABSENT -- re-verified on the login node today), and transformers' mambapy
# pscan fallback is gated on `self.training`, so it never engages at eval. mamba therefore runs a
# Python sequential scan. Measured on this exact model/GPU from its own published bAbILong run:
# 7894 s over 1,517,243 token-steps = 5.203 ms per token-step. That cross-checks against the
# independent InfiniteBench observation of ~2.9 min/row at 32768 tokens (174 s / 33280 steps =
# 5.23 ms), so the unit cost is trusted, not guessed.
#
# LongBench at the matched protocol is 450 rows summing 5,458,592 input tokens (taken from the
# rwkv7-world3 cell, same rows) plus 450x512 generated = 5.69M token-steps ~ 8.2 GPU-h serially.
# run_open_context_bench.py writes its JSON exactly ONCE, after the item loop
# (`output_path.write_text` at the end), so a serial run that overruns the cap yields NOTHING at
# all. One dataset per lane bounds the loss to a single dataset and makes the wall clock the
# slowest single dataset (narrativeqa, est 1.82 h) instead of the sum.
#
# PROTOCOL -- copied field by field from the peers' own `config` blocks so the only difference is
# the work split. LongBench: same 9 datasets, --max_samples 50, --max_input_tokens 32768,
# --max_new_tokens 512, --seed 42, same --data_dir and --longbench_config_dir. bAbILong: same 4
# datasets, --max_samples 100 (the cache's hard cap, and exactly what every other row has),
# --max_input_tokens 16384, --max_new_tokens 32.
#
# DECODE: `hf` (the default), and that is the CORRECT path here, not an oversight. The fla 0.5.0
# + transformers 5.3.0 recurrent-cache bug that forced `--decode loop` on rwkv7/gla/
# gated_deltanet cannot touch mamba: mamba is native to transformers
# (`model_type: mamba` / `MambaForCausalLM`), so the loader never imports fla for it. Passed
# explicitly so the recorded config states the choice rather than inheriting it.
#
# BUDGET: mamba is recurrent, so there is no positional ceiling and
# effective_input_budget == max_input_tokens == 32768, matching the recurrent peers
# (rwkv7-world3, gla-1.3b, global-dit-50k) exactly. phi-2 and pythia-2.8b are the rows bound by
# their own windows; mamba is not one of them.
#
# WHERE OUTPUT GOES: staging directories only. Nothing is copied into
# results/benchmarks/longctx/, because which file feeds a printed table is a deliberate decision
# made against each file's own provenance tier, never by a job silently landing a newer file in
# the canonical directory.
set -uo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# A bare $PYTHONPATH under `set -u` kills the job with no log at all.
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MODEL=$ROOT/models/mamba-2.8b
LB_DATA=$ROOT/longbench_local/extract/data
LB_SPEC=$ROOT/longbench_local/pinned_spec
BL_DATA=$HF_HOME/hub/datasets--RMT-team--babilong/snapshots/ee0d588794c7ac098062ee0d247c733d62e94fe2/data

LB_SHARDS=$REPO/results/benchmarks/longctx_mamba_longbench_shards
BL_SHARDS=$REPO/results/benchmarks/longctx_mamba_babilong_shards
MERGED=$REPO/results/benchmarks/longctx_mambafill
LOGS=$REPO/logs/qz
mkdir -p "$LB_SHARDS" "$BL_SHARDS" "$MERGED" "$LOGS"
cd "$REPO"

echo "=== mamba-2.8b fill: LongBench (9 datasets x 50) + bAbILong top-up (4 datasets x 100) ==="
echo "=== 8 lanes; est longest lane 2.22 h (babilong qa1/16k); est total 12.6 GPU-h ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8

lb () {  # lb <gpu> <dataset>
  python scripts/eval/run_open_context_bench.py \
    --suite longbench --datasets "$2" \
    --data_dir "$LB_DATA" --longbench_config_dir "$LB_SPEC" \
    --model_dir "$MODEL" --decode hf \
    --max_samples 50 --max_input_tokens 32768 --max_new_tokens 512 --seed 42 \
    --output "$LB_SHARDS/mamba-2.8b__$2__longbench.json" \
    > "$LOGS/mambafill-lb-$2.log" 2>&1
  echo "lane$1 longbench/$2 exit=$?"
}

bl () {  # bl <gpu> <dataset-with-slash> <tag>
  python scripts/eval/run_open_context_bench.py \
    --suite babilong --datasets "$2" \
    --data_dir "$BL_DATA" \
    --model_dir "$MODEL" --decode hf \
    --max_samples 100 --max_input_tokens 16384 --max_new_tokens 32 --seed 42 \
    --output "$BL_SHARDS/mamba-2.8b__$3__babilong.json" \
    > "$LOGS/mambafill-bl-$3.log" 2>&1
  echo "lane$1 babilong/$2 exit=$?"
}

# Lane packing is longest-processing-time-first over the 13 measured per-dataset estimates, so
# the wall clock is the largest single item (babilong qa1/16k, 2.22 h) rather than any sum.
( export CUDA_VISIBLE_DEVICES=0; bl 0 qa1/16k qa1-16k ) &                       # 2.22 h
( export CUDA_VISIBLE_DEVICES=1; lb 1 narrativeqa ) &                           # 1.82 h
( export CUDA_VISIBLE_DEVICES=2; lb 2 musique ) &                               # 1.18 h
( export CUDA_VISIBLE_DEVICES=3; bl 3 qa1/8k qa1-8k; lb 3 multi_news ) &        # 1.33 h
( export CUDA_VISIBLE_DEVICES=4; lb 4 qmsum; bl 4 qa2/4k qa2-4k ) &             # 1.59 h
( export CUDA_VISIBLE_DEVICES=5; lb 5 hotpotqa; bl 5 qa1/4k qa1-4k ) &          # 1.53 h
( export CUDA_VISIBLE_DEVICES=6; lb 6 passage_retrieval_en; lb 6 2wikimqa ) &   # 1.50 h
( export CUDA_VISIBLE_DEVICES=7; lb 7 gov_report; lb 7 multifieldqa_en ) &      # 1.41 h
wait
echo "=== ALL LANES DONE -- per-shard validation before any merge ==="

python3 - <<'PYEOF'
import glob, json, os

ROOT = '/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning'
SPECS = [
    ('longbench', f'{ROOT}/results/benchmarks/longctx_mamba_longbench_shards', 9, 50, 32768, 512,
     f'{ROOT}/results/benchmarks/longctx/rwkv7-world3__longbench.json'),
    ('babilong', f'{ROOT}/results/benchmarks/longctx_mamba_babilong_shards', 4, 100, 16384, 32,
     f'{ROOT}/results/benchmarks/longctx/rwkv7-world3__babilong.json'),
]

for suite, shard_dir, want_cells, want_n, want_in, want_new, peer_path in SPECS:
    print(f'\n===== {suite}: shards in {shard_dir} =====')
    paths = sorted(glob.glob(os.path.join(shard_dir, '*.json')))
    if not paths:
        print('  NO SHARDS -- every lane for this suite failed or was cut. Nothing to merge.')
        continue
    cells = 0
    for path in paths:
        blob = json.load(open(path))
        cfg, agg, rows = blob.get('config', {}), blob.get('aggregates', {}), blob.get('rows', [])
        cells += len(agg)
        tokens = [r.get('input_tokens') for r in rows if r.get('input_tokens') is not None]
        # Truncation fingerprint needs BOTH conjuncts: constant AND equal to the computed cap.
        # A constant that is not the cap is a property of the data, not of the harness.
        cap = cfg.get('effective_input_budget')
        constant = bool(tokens) and len(set(tokens)) == 1
        at_cap = constant and tokens[0] == cap
        # The observable consequence of a working stop rule is short rows, because the runner
        # breaks before appending; searching for the id proves nothing.
        gen = [r.get('generated_tokens') for r in rows if r.get('generated_tokens') is not None]
        stop_fires = bool(gen) and min(gen) < want_new
        print(f'  {os.path.basename(path):52s} cells={len(agg)} rows={len(rows):4d} '
              f'n={[v.get("n") for v in agg.values()]} '
              f'in_tok max={max(tokens) if tokens else "?"} cap={cap} '
              f'clamped={"YES" if at_cap else "no"} stop_ids={cfg.get("stop_ids")} '
              f'stop_fires={stop_fires} decode={cfg.get("decode")}')
        for field, want in (('max_samples', want_n), ('max_input_tokens', want_in),
                            ('max_new_tokens', want_new), ('seed', 42)):
            if cfg.get(field) != want:
                print(f'    *** PROTOCOL MISMATCH {field}={cfg.get(field)!r} want {want!r}')
        for dataset, entry in sorted(agg.items()):
            if entry.get('n') != want_n:
                print(f'    *** SHORT CELL {dataset}: n={entry.get("n")} want {want_n}')
    print(f'  merged-cell count would be {cells}/{want_cells}')

    # Row-identity control: the shards must cover the SAME items the peers were scored on, or the
    # merged row is not comparable no matter how well each shard validates on its own.
    if os.path.exists(peer_path):
        peer = json.load(open(peer_path))
        peer_ids = {(r.get('dataset'), r.get('id')) for r in peer.get('rows', [])}
        mine = set()
        for path in paths:
            mine |= {(r.get('dataset'), r.get('id'))
                     for r in json.load(open(path)).get('rows', [])}
        shared = peer_ids & mine
        print(f'  ROW CONTROL vs {os.path.basename(peer_path)}: peer={len(peer_ids)} '
              f'mine={len(mine)} shared={len(shared)} mine_only={len(mine - peer_ids)}')
        if mine and not shared:
            print('    *** ZERO shared items: the shards are not the peers\' rows. Do NOT compare.')
PYEOF

echo "=== MERGE (gated: refuses to write a short file at the publish path) ==="
python3 scripts/eval/merge_mamba_shards.py \
  --shard_dir "$LB_SHARDS" --glob 'mamba-2.8b__*__longbench.json' \
  --expect_cells 9 --expect_rows_per_cell 50 \
  --output "$MERGED/mamba-2.8b__longbench.json"
echo "longbench merge exit=$?"

python3 scripts/eval/merge_mamba_shards.py \
  --shard_dir "$BL_SHARDS" --glob 'mamba-2.8b__*__babilong.json' \
  --expect_cells 4 --expect_rows_per_cell 100 \
  --output "$MERGED/mamba-2.8b__babilong.json"
echo "babilong merge exit=$?"

echo "=== RESULT (staging only; nothing was copied into results/benchmarks/longctx/) ==="
python3 - <<'PYEOF'
import json, os

ROOT = '/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning'
MERGED = f'{ROOT}/results/benchmarks/longctx_mambafill'
PEERS = {
    'longbench': ['rwkv7-world3', 'gla-1.3b', 'global-dit-50k', 'champ-dynamic-30k',
                  'qwen2.5-3b', 'llama-3.2-3b', 'phi-2'],
    'babilong': ['rwkv7-world3', 'gla-1.3b', 'global-dit-50k', 'champ-dynamic-30k'],
}

def macro(path):
    agg = json.load(open(path)).get('aggregates', {})
    scores = [v['score'] for v in agg.values() if isinstance(v, dict) and 'score' in v]
    return (100 * sum(scores) / len(scores), len(scores)) if scores else (None, 0)

for suite, peers in PEERS.items():
    path = f'{MERGED}/mamba-2.8b__{suite}.json'
    if not os.path.exists(path):
        print(f'\n{suite}: NOT WRITTEN (a gate refused it; see the merge output above)')
        continue
    value, cells = macro(path)
    print(f'\n{suite}: mamba-2.8b macro over {cells} cells = {value:.2f}')
    blob = json.load(open(path))
    for dataset, entry in sorted(blob['aggregates'].items()):
        print(f'    {dataset:24s} n={entry.get("n"):4d}  {100 * entry["score"]:6.2f}')
    print('  peers on the same suite, from results/benchmarks/longctx/ (context, not a claim):')
    for peer in peers:
        peer_path = f'{ROOT}/results/benchmarks/longctx/{peer}__{suite}.json'
        if os.path.exists(peer_path):
            peer_value, peer_cells = macro(peer_path)
            print(f'    {peer:22s} macro over {peer_cells} cells = {peer_value:6.2f}')
    print(f'  file: {path}')
PYEOF
echo "=== MAMBA FILL COMPLETE ==="
exit 0

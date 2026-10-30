#!/usr/bin/env bash
# Why mamba-2.8b scores exactly 0.000 on every LongBench row: length limit, or harness defect?
#
# The two hypotheses make different predictions and the existing evidence does not separate
# them, so this probe crosses prompt length with the decode path.
#
# What is already known (all from files on disk, no new compute):
#   * weights are healthy       -- wikitext2 PPL 13.04, pg19 10.56 (results/benchmarks/ppl/)
#   * short prompts are fine    -- a 6-token piqa prompt yields fluent English
#                                  (results/benchmarks/baseline-mamba-2.8b/piqa.jsonl)
#   * 3.7k is still fluent      -- bAbILong qa1/4k: "She is in the garden, and she is very
#                                  happy." Wrong answer, but not degenerate. qa1_4k = 2.0
#   * 7.5k has collapsed        -- MemEval (mean 7540 tok) predicts ",,,,"  and
#                                  "vs, vs, vs, vs, vs, ..." on 0/50 scoring rows
#   * LongBench is ~9-11k       -- 49/49 hotpotqa rows exactly 0.000
#
# So fluency dies somewhere between 3.7k and 7.5k. That is consistent with BOTH hypotheses:
#   H1 (capability) mamba-2.8b was trained at 2048 on the Pile; its recurrent state genuinely
#      degrades far past that, so the collapse is the model's and the row is publishable as a
#      real (if unflattering) result.
#   H2 (harness) something in our decode path corrupts long sequences -- exactly what happened
#      to rwkv7, where model.generate() disagreed with a hand-written single-token loop on
#      18/20 tokens and cost 12x on hotpotqa.
#
# H2 is not idle speculation here: this model runs the SEQUENTIAL FALLBACK kernel (the log says
# selective_scan_fn / causal_conv1d_fn are all None, no mamba-ssm and no causal-conv1d
# installed), which is a different code path from the one the published numbers were produced
# with, and it is the path where a long-sequence numerical or cache bug would hide.
#
# Discriminator: `--decode loop` vs `--decode hf` at 2048 / 4096 / 8192.
#   * both paths collapse at the same length  => H1, the state is the limit. Publishable.
#   * loop stays fluent where hf collapses    => H2, our artefact. Not publishable as a score.
# Note mamba is NOT an fla model (log: "fla recurrent-generate patch applied: False"), so the
# known fla/transformers incompatibility cannot be the explanation -- which is precisely why
# this needs its own measurement rather than an assumption either way.
#
# 5 rows x 3 lengths x 2 paths on one GPU. Keeps hotpotqa/seed/scorer identical to the sweep.
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

OUT=$REPO/results/benchmarks/falsify_zero
LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

for LEN in 2048 4096 8192; do
  for DEC in hf loop; do
    python3 scripts/eval/run_open_context_bench.py \
      --suite longbench --datasets hotpotqa \
      --data_dir $ROOT/longbench_local/extract/data \
      --longbench_config_dir $ROOT/longbench_local/pinned_spec \
      --max_samples 5 --max_input_tokens "$LEN" --max_new_tokens 64 --seed 42 --device cuda \
      --arm raw --decode "$DEC" \
      --model_dir "$ROOT/models/mamba-2.8b" \
      --output "$OUT/mamba-len$LEN-$DEC.json" \
      > "$LOGS/mamba-probe-len$LEN-$DEC.log" 2>&1
    echo "done len=$LEN decode=$DEC"
  done
done

echo "=== MAMBA LENGTH x DECODE PROBE DONE ==="
python3 - <<'PY'
import json, pathlib
out = pathlib.Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning"
                   "/results/benchmarks/falsify_zero")
print(f"{'len':>6s} {'decode':>6s} {'score':>7s} {'distinct_ids':>13s}  first_prediction")
for length in (2048, 4096, 8192):
    for dec in ("hf", "loop"):
        p = out / f"mamba-len{length}-{dec}.json"
        if not p.exists():
            print(f"{length:>6d} {dec:>6s}   (missing)"); continue
        d = json.loads(p.read_text())
        rows = d["rows"]
        # distinct-id count is the degeneracy fingerprint: fluent text has many, a collapsed
        # row has 1-2. Averaged so one lucky row cannot hide a collapse.
        distinct = sum(len(set(r.get("prediction_token_ids") or [])) for r in rows) / len(rows)
        score = 100 * d["aggregates"]["hotpotqa"]["score"]
        print(f"{length:>6d} {dec:>6s} {score:7.1f} {distinct:13.1f}  "
              f"{(rows[0].get('prediction') or '')[:60]!r}")
PY

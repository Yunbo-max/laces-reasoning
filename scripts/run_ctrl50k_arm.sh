#!/bin/bash
# Controlled fixed-vs-dynamic writer pair: one arm, chunked / biRWKV / 2.9B / 50k.
#
# Usage: run_ctrl50k_arm.sh <fixed|dynlowrank>
#
# Both arms are launched with THIS SAME SCRIPT and the same config template, so
# the only difference between them is the single $WRITER substitution. No Hydra
# overrides are passed: everything lives in the config file, so the recipe is
# fully recorded on disk and the pair cannot drift via a forgotten CLI flag.
# (The historical 50k "pair" drifted exactly that way -- its two arms got
# different resume paths from CLI overrides.)

set -euo pipefail

WRITER="${1:?usage: run_ctrl50k_arm.sh <fixed|dynlowrank>}"
case "$WRITER" in
  fixed|dynlowrank) ;;
  *) echo "FATAL: writer must be 'fixed' or 'dynlowrank', got '$WRITER'" >&2; exit 2 ;;
esac

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning
cd "$REPO"

# Offline: compute nodes have no network.
export HF_HOME=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# MUST be written with the ${VAR:+:$VAR} guard -- a bare $PYTHONPATH under
# `set -u` kills the job before python starts, with no log at all.
export PYTHONPATH=/inspire/hdd/global_user/zhangjiaquan-253108540222/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="rwkv_relay_2.9B_chunked_birwkv_${WRITER}_r32_ctrl50k"

echo "=========================================================="
echo " ctrl50k arm      : $WRITER"
echo " config           : configs/${CONFIG}.yaml"
echo " started          : $(date -Is)"
echo " host             : $(hostname)"
echo " git HEAD         : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo " git dirty files  : $(git status --porcelain 2>/dev/null | wc -l)"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# Gate 1: the parameter-set / writer-routing assertion. Refuse to burn GPU hours
# on a misconfigured writer.
echo "[gate] preflight assertions ..."
python3 scripts/preflight_ctrl50k_pair.py
echo "[gate] preflight OK"

# Gate 2: data must be present locally (nothing can be downloaded here).
TOKDIR=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/owt_rwkv_tokens/train
if [ ! -d "$TOKDIR" ]; then echo "FATAL: token dir missing: $TOKDIR" >&2; exit 3; fi
echo "[gate] token shards present"

exec python3 -u train_state_hijacking_dit.py --config-name "$CONFIG"

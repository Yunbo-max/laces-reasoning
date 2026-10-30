#!/usr/bin/env bash
# 13.3B dynamic-basis (dynlowrank r32), joint S1+S2 recipe, BiRWKV denoiser.
#
# Uses the explicit per-mode configs, which declare the denoiser and n_basis=32:
#   chunked -> rwkv_relay_13.3B_dynamic_basis_dynlowrank_r32_chunked_birwkv
#   global  -> rwkv_relay_13.3B_dynamic_basis_dynlowrank_r32_global_birwkv
#
# One stage-2 job that trains S1 and S2 together (s2_unfreeze_s1). Outputs go to
# SAVE_DIR (default outputs_dynamic_basis_nb32) so these runs never collide with
# the n_basis=16 pair launched earlier from the shared config.
set -euo pipefail

REPO_ROOT="/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning"
cd "${REPO_ROOT}"

MODE="${MODE:?set MODE=chunked or MODE=global}"
NPROC="${NPROC:-8}"
BATCH="${BATCH:-4}"                 # per GPU; x8 = global 32
STEPS="${STEPS:-50000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
MASTER_PORT="${MASTER_PORT:-29511}"
TAG="${TAG:-nb32}"
SAVE_DIR="${SAVE_DIR:-./outputs_dynamic_basis_${TAG}}"

# Resume target MUST carry a nonzero state_scale, otherwise the dynlowrank writer
# receives exactly zero gradient on the first backward (states = U V^T * scale) and
# the trainer's require_s1_writer_grad guard aborts. The pure S0 checkpoint has
# state_scale = 0.0 and no S1, so it cannot be used here. This 512-geometry 13.3B S1
# checkpoint has state_scale = 0.146.
RESUME_CKPT="${RESUME_CKPT:-/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/outputs_relay/owt512-traj32x16-13.3B-basis32-prefix-suffix-blend0p5-s1-rwkv/step_00050000}"

case "${MODE}" in
  chunked) CONFIG="rwkv_relay_13.3B_dynamic_basis_dynlowrank_r32_chunked_birwkv" ;;
  global)  CONFIG="rwkv_relay_13.3B_dynamic_basis_dynlowrank_r32_global_birwkv" ;;
  *) echo "FATAL: MODE must be chunked or global, got '${MODE}'" >&2; exit 2 ;;
esac

RUN_NAME="laces-13.3B-dynlowrank-r32-${MODE}-birwkv-${TAG}-${STEPS}"
LOG_DIR="${REPO_ROOT}/validation_logs_qz"
mkdir -p "${LOG_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC - 1)))"

echo "=========================================="
echo "13.3B dynamic-basis (dynlowrank) + BiRWKV  [${MODE}]"
echo "  config=${CONFIG}"
echo "  joint S1+S2, stage=2, s2_unfreeze_s1"
echo "  NPROC=${NPROC} BATCH=${BATCH}/gpu STEPS=${STEPS}"
echo "  resume=${RESUME_CKPT}"
echo "  save_dir=${SAVE_DIR}"
echo "=========================================="
[[ -f "${RESUME_CKPT}/model.pt" ]] || { echo "FATAL: resume checkpoint missing: ${RESUME_CKPT}/model.pt" >&2; exit 2; }

torchrun --nnodes=1 --nproc_per_node="${NPROC}" --node_rank=0 \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
  train_state_hijacking_dit.py \
  --config-name "${CONFIG}" \
  "training.stage=2" \
  "training.resume=${RESUME_CKPT}" \
  "training.num_train_steps=${STEPS}" \
  "training.train_batch_size=${BATCH}" \
  "training.save_every_n_steps=${SAVE_EVERY}" \
  "logging.run_name=${RUN_NAME}" \
  "logging.save_dir=${SAVE_DIR}" \
  > "${LOG_DIR}/${RUN_NAME}.log" 2>&1

echo "[done] ${REPO_ROOT}/${SAVE_DIR#./}/${RUN_NAME}"

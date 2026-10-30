#!/usr/bin/env bash
# Injection-repair finetune: make the relay's injected states non-destructive.
#
# Diagnosis this responds to: the frozen backbone already knows the answers
# (raw MMLU-Redux 42.9%) and the relay *destroys* that (36.9% trajectory,
# 26.0% single-z). So the target is not new knowledge, it is a write that does
# not damage the backbone. Two levers, both existing knobs:
#
#   loss.s2_coadapt_ce_loss_weight  0.5 -> 3.0
#       The coadapt CE scores the true continuation under the *injected* model.
#       Raising it directly penalises injection that makes next-token prediction
#       worse -- i.e. it pays the relay to be gentle. This is the repair term.
#
#   optimizer.lr  1e-4 -> 2e-5
#       A repair pass on an already-converged writer, not fresh training. A high
#       LR here would just re-break what we are trying to preserve.
#
# Everything else mirrors training/run_prefix_suffix_trajectory.sh so the run is
# architecture-identical to champ-dynamic (chunked 32x16, BiRWKV, ddpm, basis16).
#
# Data: s2_traj_sft_512_mmlupro_focused (444k, math/multi-step reasoning corpora).
# The name is misleading -- it contains no MMLU-Pro items (see its manifest:
# nemotron_post_math / openthoughts3 / openr1_mot / nemotron_math_v4), so training
# on it does not contaminate any evaluated benchmark.
#
# Run with torchrun --nproc_per_node=8.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/s2_traj_sft_512_mmlupro_focused}"
BASE_CKPT="${BASE_CKPT:-outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000}"
RUN_NAME="${RUN_NAME:-laces-2.9B-dynlowrank-r32-injection-repair}"
STEPS="${STEPS:-40000}"
BATCH="${BATCH:-4}"
LR="${LR:-2.0e-5}"
COADAPT_W="${COADAPT_W:-3.0}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
NPROC="${NPROC:-8}"

for p in "$DATA_DIR" "$BASE_CKPT"; do
  [[ -e "$p" ]] || { echo "FATAL: missing $p" >&2; exit 1; }
done

echo "repair finetune"
echo "  data      : $DATA_DIR"
echo "  resume    : $BASE_CKPT"
echo "  run_name  : $RUN_NAME"
echo "  steps=$STEPS batch=$BATCH lr=$LR coadapt_ce_w=$COADAPT_W save_every=$SAVE_EVERY nproc=$NPROC"

exec torchrun --nproc_per_node="$NPROC" train_state_hijacking_dit.py \
  --config-name rwkv_relay_2.9B_dynamic_basis_dynlowrank_r32 \
  data.token_dir="$DATA_DIR" \
  data.max_length=512 \
  model.n_basis=16 \
  model.trajectory_chunk_size=32 \
  model.trajectory_horizon=16 \
  model.trajectory_s1_mode=independent \
  model.trajectory_denoiser_type=birwkv \
  model.trajectory_state_hidden=256 \
  model.trajectory_state_depth=4 \
  model.trajectory_state_blend=0.7 \
  training.stage=2 \
  training.gen_type=ddpm \
  training.num_train_steps="$STEPS" \
  training.resume="$BASE_CKPT" \
  training.train_batch_size="$BATCH" \
  training.save_every_n_steps="$SAVE_EVERY" \
  training.s2_unfreeze_s0=false \
  +training.prefix_suffix_trajectory_s2=true \
  +training.cfg_drop_prob=0.1 \
  +training.prefix_suffix_min_prefix=32 \
  +training.prefix_suffix_min_suffix=32 \
  loss.s2_coadapt_ce_loss_weight="$COADAPT_W" \
  loss.trajectory_delta_loss_weight=0.0 \
  optimizer.lr="$LR" \
  logging.run_name="$RUN_NAME" \
  logging.save_dir=./outputs_dynamic_basis

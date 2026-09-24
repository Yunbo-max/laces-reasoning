#!/usr/bin/env bash
# DiT width/depth ablation, CLEAN (no resume, random-init planner).
#
# One rung per invocation: run_rung.sh <hidden> <depth>
#
# Why no `training.resume`: the 2026-09-17 attempt passed
#   training.resume=outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000
# which is a dit_hidden=768 checkpoint. Feeding it to a 512 or 1024 model makes
# _compatible_trainable_state() drop the whole `trajectory_dit.*` root (473 keys,
# all 768-dim) by guilt-by-association, so the denoiser would be random while the
# encoder/S1 were inherited -- different init per rung, not an ablation. Both those
# runs also died at startup with zero checkpoints, so nothing is being discarded here.
# `training.resume` defaults to null in the config; omitting the override is the clean path.
#
# Comparability: these three rungs are comparable to EACH OTHER only. Every existing
# 50k global number is resume-derived from step_00030000 and will be higher. Do not
# mix these rows into that table.
set -euo pipefail

H="${1:?usage: run_rung.sh <dit_hidden> <dit_depth>}"
D="${2:?usage: run_rung.sh <dit_hidden> <dit_depth>}"

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning
RWKV="$REPO/models/RWKV7-Goose-World3-2.9B-HF-fla-v2"
RUN="laces-2.9B-dynlowrank-r32-global-dit${H}x${D}-b16-50k-scratch"

cd "$REPO"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

python train_state_hijacking_dit.py \
  --config-name rwkv_relay_2.9B_dynamic_basis_dynlowrank_r32_global \
  model.rwkv_local_path="$RWKV" \
  model.dit_hidden="$H" \
  model.dit_depth="$D" \
  model.dit_num_heads=4 \
  training.num_train_steps=50000 \
  training.train_batch_size=16 \
  training.save_every_n_steps=5000 \
  logging.run_name="$RUN"

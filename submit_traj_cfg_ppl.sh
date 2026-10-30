#!/bin/bash
# Submit traj_cfg PPL tasks for chunked models (champ-fixed-512, champ-dynamic)
# These were previously evaluated with prefix_cfg and got corrupted (66k PPL)
# Now they will use score_relay_traj (chunk-wise trajectory injection)

set -e

WORKSPACE_ID="ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
PROJECT="project-632c8db8-4530-413a-ada5-df91774a7e09"
SPEC_H100="79fe954a-be92-4772-ac0b-94ad8a79b7bb"
PRIORITY=6

REPO="/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning"

# Task: champ-fixed-512 traj_cfg PPL
echo "Submitting: laces-ppl3-champ-fixed-512-traj_cfg"
qz train CreateJob \
  --workspace-id "$WORKSPACE_ID" \
  --project-id "$PROJECT" \
  --spec-id "$SPEC_H100" \
  --priority "$PRIORITY" \
  --name "laces-ppl3-champ-fixed-512-traj_cfg" \
  --image-uri "docker.sii.shaipower.online/inspire-studio/relay2:v2" \
  --command "
cd '$REPO'
export HF_HUB_OFFLINE=1
export HF_HOME=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
python scripts/eval/eval_lm_ppl.py \
  --impl relay \
  --ckpt_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/outputs_fixed_basis/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000 \
  --mode traj_cfg \
  --corpora wikitext2,wikitext103,pg19,lm1b,ptb,wmt14,xsum \
  --max_files 100 \
  --split 256 \
  --steps 100 \
  --cfg_scale 1.0 \
  --output results/benchmarks/ppl/champ-fixed-512_traj_cfg.json \
  2>&1 | tee /inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/qz_eval_logs/ppl3-champ-fixed-512-traj_cfg.log
" \
  2>&1 | tee -a /tmp/qz_submit.log

# Task: champ-dynamic traj_cfg PPL
echo "Submitting: laces-ppl3-champ-dynamic-traj_cfg"
qz train CreateJob \
  --workspace-id "$WORKSPACE_ID" \
  --project-id "$PROJECT" \
  --spec-id "$SPEC_H100" \
  --priority "$PRIORITY" \
  --name "laces-ppl3-champ-dynamic-traj_cfg" \
  --image-uri "docker.sii.shaipower.online/inspire-studio/relay2:v2" \
  --command "
cd '$REPO'
export HF_HUB_OFFLINE=1
export HF_HOME=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
python scripts/eval/eval_lm_ppl.py \
  --impl relay \
  --ckpt_dir /inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000 \
  --mode traj_cfg \
  --corpora wikitext2,wikitext103,pg19,lm1b,ptb,wmt14,xsum \
  --max_files 100 \
  --split 256 \
  --steps 100 \
  --cfg_scale 1.0 \
  --output results/benchmarks/ppl/champ-dynamic_traj_cfg.json \
  2>&1 | tee /inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/qz_eval_logs/ppl3-champ-dynamic-traj_cfg.log
" \
  2>&1 | tee -a /tmp/qz_submit.log

echo "Jobs submitted. Check /tmp/qz_submit.log for details."

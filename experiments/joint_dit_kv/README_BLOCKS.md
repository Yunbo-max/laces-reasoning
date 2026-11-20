# Single-Prompt Block-wise Latent Diffusion

**Status: experimental implementation; NOT trained and NOT shown to beat LatentMAS.**
This is the single-prompt research direction. The older README.md and run.py
remain a separate four-private-evidence joint-fusion experiment.

## Architecture: block-internal bidirectional, cross-block causal

We reuse the blockwise autoregressive factorization/attention-mask structure
of BD3-LMs (Arriola et al., ICLR 2025). The official BD3-LM code uses
DISCRETE token diffusion: https://github.com/kuleshov-group/bd3lms
Our implementation uses continuous rectified flow over learned latents.
The original BD3 discrete denoising sampler cannot be dropped into this code
without changing the model's latent objective.

    input single prompt q
         |
    shared QueryEncoder(q) -> c(q)
         |
    Block 1: joint bidirectional denoise of S latent slots from noise
         |
       STOP? --yes--> final KV writer
         |
    Block 2: joint bidirectional denoise of NEW S slots from noise,
             conditioning on c(q) and CLEAN previously generated Block 1
         |
       STOP? --yes--> final KV writer
         |
    ... up to max_blocks
         |
    concat ALL completed block slots -> trained Virtual KV Writer
         |
    frozen Qwen(q, past_key_values=virtual_KV) -> answer

Within a block, all positions attend to each other. Earlier blocks never
attend to current/future blocks. The current block can attend to all
previously generated blocks. This blockwise causal mask is tested in
tests/test_blocks.py.

A block is NOT an Agent with its own model or separate private observation.
It is a *group of latent reasoning steps*. Training uses real intermediate
reasoning-step strings from GSM8K, not artificially repeated copies of the
same final answer.

Config (starting point):
- Qwen2.5-0.5B-Instruct FROZEN (single 3090 pilot); 1.5B later on H100
- 2 reasoning streams per block, 4 slots each (S=8); latent width 128
- 4 blocks maximum -> at most 32 virtual KV slots
- 4-layer shared Transformer block denoiser, rectified-flow Euler 2 steps
- learned STOP/CONTINUE head evaluated AFTER each completed block
- frozen Qwen remains autoregressive, no MTP

Important: The STOP head is a learned binary classifier, NOT a special
continuous-latent EOS token. Qwen's own final-output EOS is separate.

## Training stages and losses

1. AE: individual real reasoning-step -> learned query encoder -> P latent
   slots -> KV writer -> frozen Qwen reconstruction CE. Update encoder+writer.
2. Block: freeze encoder/receiver. Teacher-force CLEAN earlier target blocks,
   noise only the current target block, and train flow-velocity MSE. For
   RECEIVER CE, sample all preceding/current blocks FROM NOISE, autoregressively,
   synthesise their combined KV, and teacher-force only the final answer. STOP
   BCE is trained on sampled (not oracle-clean) blocks, with the real number
   of step blocks as the target.

    L = L_block_flow_MSE + lambda_ce * L_answer_CE
        + lambda_stop * L_stop_BCE

Gold reasoning steps provide training targets, not generation conditions.
The flow-MSE term uses teacher-forced history whereas the receiver-CE rollout
uses sampled history: a future self-forcing experiment is recommended.

## Prepare data

Official GSM8K *train* only, SHA256 95%/5% train/dev split. The original
official GSM8K test data is NEVER used for training or checkpoint selection.
Each rationale line is one step; append a final-answer step. Raw intermediate
calculator annotations are removed from the training text. The converter
saves train.jsonl/dev.jsonl/manifest.json.

    pip install datasets
    python -m experiments.joint_dit_kv.prepare_gsm8k \
      --out data/gsm8k_block_train --holdout-percent 5

For a tiny integration smoke (NOT a research benchmark):

    python -m experiments.joint_dit_kv.prepare_gsm8k \
      --make-toy --out data/gsm8k_block_smoke --train-toy 256 --dev-toy 64

## Execute

    python -m pytest -q experiments/joint_dit_kv/tests

AE stage:

    python -m experiments.joint_dit_kv.run_blocks \
      --command train --stage ae \
      --train data/gsm8k_block_train/train.jsonl \
      --dev data/gsm8k_block_train/dev.jsonl \
      --out runs/block_ae --steps 400 --accum 1

Joint block training:

    python -m experiments.joint_dit_kv.run_blocks \
      --command train --stage block --init-ae runs/block_ae/best.pt \
      --train data/gsm8k_block_train/train.jsonl \
      --dev data/gsm8k_block_train/dev.jsonl \
      --streams-per-block 2 --max-blocks 4 --flow-steps 2 \
      --out runs/block_joint --steps 1200 --accum 1

Adaptive STOP evaluation:

    python -m experiments.joint_dit_kv.run_blocks \
      --command eval --checkpoint runs/block_joint/best.pt \
      --data data/gsm8k_block_train/dev.jsonl \
      --out runs/eval_adaptive

Fixed-length matched controls (use 1, 2, and 4):

    python -m experiments.joint_dit_kv.run_blocks \
      --command eval --checkpoint runs/block_joint/best.pt \
      --data data/gsm8k_block_train/dev.jsonl \
      --force-blocks 2 --out runs/eval_B2

Compare accuracy, end-to-end latency, number of blocks, and number of
virtual KV slots for adaptive vs fixed. Every STOP probability is recorded
along with the final answer.

## Relation to earlier experiment and LatentMAS

- This route uses one prompt q. It does NOT read the earlier HotpotQA data's
  four separate private evidence strings.
- The old four-Agent fully joint DiT uses bidirectional attention across
  ALL slots simultaneously. The blockwise version is bidirectional only
  within a block, strictly autoregressive across completed blocks.
- This approach is centralized latent reasoning, not decentralized Agents.
- Official LatentMAS works via sequential latent/KV relay and is training-free.
  For an apples-to-apples baseline, use the original LatentMAS on the same
  single questions with a matching frozen Qwen receiver; the existing
  run_latentmas.py in this repository expects private evidence and must not
  be used as-is for single-prompt claims.
- The current implementation uses straightforward per-block decoding and
  does not yet reproduce BD3's two-forward-pass optimization or cross-block
  denoiser KV-cache acceleration. The *final receiver* KV is built once
  from the resulting latent blocks.
- For a paper, add same-backbone raw Qwen, official LatentMAS, independent
  non-interacting step latents, deterministic joint attention, shuffled
  latent controls, and paired sealed GSM8K test evaluation with multiple seeds.

**No GPU training or benchmark superiority is asserted by this code.**

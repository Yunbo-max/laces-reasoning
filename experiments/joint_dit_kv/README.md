# Centralized Joint DiT -> virtual KV -> frozen Qwen (LatentMAS study)

**Status: implementation + untrained experiment harness. No accuracy/latency win is claimed.**
This experimental branch does **not** replace the established LaLaLanD/S0/S1/S2 code
and does not modify the main branch.

## Research question

Can **joint bidirectional generation of multiple agent-conditioned latent slots**
outperform **sequential latent-KV relay** (official LatentMAS) on **matched
multi-source evidence tasks**, when the decoder is the same frozen Qwen?

This is **centralized joint reasoning**, not decentralized execution:
the shared denoiser sees all local-agent condition slots. There is one final receiver.

## Architecture / loss

1. Each agent's private evidence is encoded by a *shared* query-based encoder
   (P slots x d). Frozen **receiver token embeddings**, trainable positional
   embeddings + query attention. No mean-pooled 32-D Gaussian VAE.
2. Stage AE: use evidence/answer *token reconstruction CE* through a shared
   learned virtual-KV writer plus frozen Qwen. This trains decodability and
   exact symbol retention. No KL regularizer in this prototype.
3. Stage Joint: input = [agent A..M evidence slots, question slots].
   The target clean trajectory is [agent evidence slots, answer slots], where
   **gold-answer slots appear only as training targets**. The DiT prior sees
   clean context slots and *noisy* trajectory slots. A full-attention encoder
   permits symmetric interactions among all agents and the answer slot.
   We use **rectified-flow matching MSE**, not DDPM epsilon prediction.
4. **For Receiver CE, unroll flow sampling from pure Gaussian noise**,
   write resulting slots into frozen Qwen's per-layer virtual KV cache,
   and teacher-force the answer. Gold-answer latents are not used for the
   receiver's prediction.
5. The writer maps P latent slots to **P synthetic post-RoPE KV positions** per
   transformer layer. The receiver's real question follows the virtual prefix;
   its first-answer-token logit is included in CE. Later tokens update KV
   normally. Qwen, RoPE and attention cache semantics remain fixed.

Stages:
  AE:          L = CE(recover exact local evidence or answer)
  Joint:       L = a * flow_MSE + b * receiver_CE + c * evidence_AE_CE
  Independent: L = b * receiver_CE + c * evidence_AE_CE
(the independent baseline uses the **same trained AE checkpoint** and
concatenates local encodings; it does not use a joint denoiser).

The source implementation is in model.py, data.py and run.py.

## Environment

Use Python 3.10+, PyTorch 2.5+ and a compatible Transformers 4.x release.
For reproducibility pin exact versions in the experiment manifest once validated
on your target server. HF model weights need to be locally available or download
access enabled. Do not use a larger model than GPU memory allows.

Example:

    python -m pip install "torch>=2.5" "transformers>=4.45,<5" pytest

**3090 24GB**: first test Qwen2.5-0.5B-Instruct with batch=1
(grad_accum=1) and 2 flow steps.
**H100**: then try the same protocol with Qwen2.5-1.5B-Instruct.
A frozen model still requires backward activations for KV writer training.
If memory is insufficient, reduce agent/context lengths or use a smaller LM.
No training time estimate or memory bound is asserted.

## Step 1: CPU contract tests

    python -m pytest -q experiments/joint_dit_kv/tests/test_contracts.py

test_small_qwen_cache_grad_through_writer checks a tiny random Qwen2 for:
virtual KV cache contract, receiver first token loss and differentiability
back to encoder-independent DiT/writer, with frozen Qwen weights.
A test pass does not establish end-task capability.

## Step 2: self-contained synthetic smoke dataset

    python -m experiments.joint_dit_kv.data \
      --make-toy data/joint_dit_smoke --train 256 --dev 64

The 4 evidence pieces form a 3-hop key/locker/color join with a distractor.
**Smoke data is not a publishable benchmark or a basis for beating LatentMAS.**

Unified real-data JSONL format, one item per line:
    {"id":"001","question":"...","agents":["A private...","B private...", "..."],"answer":"...","choices":[]}

The data reader can also read HiddenBench's native benchmark.json array
(shared_information/hidden_information/correct_answer), but that benchmark
should be used only as evaluation unless a separately licensed, nonoverlapping
training split is provided. Training/dev exact question overlap is prohibited.

## Step 3: train token-reconstructing AE

    python -m experiments.joint_dit_kv.run \
      --command train --stage ae \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --train data/joint_dit_smoke/train.jsonl \
      --dev data/joint_dit_smoke/dev.jsonl \
      --out runs/joint_ae --steps 400 --eval-every 100 \
      --grad-accum 1 --eval-limit 12

Review runs/joint_ae/dev_latest.json (evidence token CE), then inspect
whether exact numbers/operators survive reconstruction and whether a shuffled
condition changes the receiver's predictions.

## Step 4: two independently trained variants from EXACTLY the same AE

Joint bidirectional DiT:

    python -m experiments.joint_dit_kv.run \
      --command train --stage joint --init-ae runs/joint_ae/best.pt \
      --train data/joint_dit_smoke/train.jsonl \
      --dev data/joint_dit_smoke/dev.jsonl \
      --out runs/joint_dit --steps 1200 --eval-every 200 \
      --sample-steps 2 --grad-accum 1 --eval-limit 12

Independent encoder + KV writer, no joint denoising:

    python -m experiments.joint_dit_kv.run \
      --command train --stage independent --init-ae runs/joint_ae/best.pt \
      --train data/joint_dit_smoke/train.jsonl \
      --dev data/joint_dit_smoke/dev.jsonl \
      --out runs/independent --steps 1200 --eval-every 200 \
      --grad-accum 1 --eval-limit 12

For a fair paper ablation, match optimization steps, total training examples,
backbone, eval prompts and sampling budget; note that independent has fewer
trainable operations. Include an equally sized deterministic joint-attention
baseline in the full study before attributing gains to diffusion.

## Step 5: paired eval (same IDs, source, Qwen and sampling)

    python -m experiments.joint_dit_kv.run --command eval \
      --strategy joint --checkpoint runs/joint_dit/best.pt \
      --data data/joint_dit_smoke/dev.jsonl --out runs/eval_joint \
      --temperature 0.7 --top-p 0.9 --seed 42

    python -m experiments.joint_dit_kv.run --command eval \
      --strategy independent --checkpoint runs/independent/best.pt \
      --data data/joint_dit_smoke/dev.jsonl --out runs/eval_independent \
      --temperature 0.7 --top-p 0.9 --seed 42

    python -m experiments.joint_dit_kv.run --command eval \
      --strategy full_text --model Qwen/Qwen2.5-0.5B-Instruct \
      --data data/joint_dit_smoke/dev.jsonl --out runs/eval_fulltext \
      --temperature 0.7 --top-p 0.9 --seed 42

    python -m experiments.joint_dit_kv.run --command eval \
      --strategy question_only --model Qwen/Qwen2.5-0.5B-Instruct \
      --data data/joint_dit_smoke/dev.jsonl --out runs/eval_questiononly \
      --temperature 0.7 --top-p 0.9 --seed 42

These return individual JSONL predictions and summary files. The built-in
normalizer is a conservative smoke metric, **not** a substitute for official
GSM8K/MBPP/HiddenBench/MASBench evaluation scripts.

## Step 6: OFFICIAL LatentMAS baseline, not independent fusion mislabeled

Obtain the official release at https://github.com/Gen-Verse/LatentMAS.
Its Python dependencies, including optional plotting packages, must be installed
in an environment compatible with its pinned transformers implementation.
Our adapter loads its unchanged models.py/ModelWrapper via importlib and calls
generate_latent_batch sequentially with cumulative KV:

    git clone https://github.com/Gen-Verse/LatentMAS.git LatentMAS
    python -m experiments.joint_dit_kv.run_latentmas \
      --source LatentMAS --data data/joint_dit_smoke/dev.jsonl \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --latent-steps 10 --max-new-tokens 32 \
      --temperature 0.7 --top-p 0.9 --seed 42 \
      --out runs/eval_latentmas/predictions.jsonl

IMPORTANT: LatentMAS is **training-free**, while ours uses learned AE + prior +
writer. Report all training compute and amortized inference cost, including
multi-step flow. Compare both under identical receiver and evaluation data.
Official LatentMAS code currently samples the final answer stochastically;
therefore the evaluation examples above set matching receiver temperature/top_p.
Check its version-specific cache handling before accepting benchmark numbers.
KV-resident bytes reported by the official adapter are a **cache inventory**,
not measured data-transfer bandwidth.

Our joint system is currently a single-process **centralized** prototype:
the latent_payload_bytes field is a theoretical bf16 representation size,
**not** actual network traffic or decentralized communication savings.
For every model, measure complete end-to-end time including evidence encoding,
diffusion, KV construction and text decoding. Keep benchmark sealed tests
separate from training, model selection and hyperparameter sweeps.

## Minimum publication-critical follow-ups

- Confirm tiny-Qwen KV cache integration on the exact pinned transformers
  release, plus receiver gradients, train-to-inference first-token parity,
  and no leaked gold-answer latents.
- Replace synthetic-smoke-only pretraining with a separate distributed-evidence
  training corpus. Keep benchmark examples/near duplicates out of training.
- Add **trained deterministic joint attention** and **shuffle / zero / swap
  each agent's evidence** controls under the same checkpoint and evaluation loop.
- Compare using *official* benchmark metrics, per-question paired results,
  uncertainty, total GPU-hours, transmitted bytes vs resident KV size and
  latency decomposition. A method beating LatentMAS on a small toy is not
  enough to claim a research advance.


## Real training data: HotpotQA distributed documents

For a non-toy **training** corpus, the official HotpotQA distractor **train**
split can be converted to four local-agent document messages. The script uses
question-word-overlap retrieval and only the sentence text for local evidence.
Supporting fact labels are used for a *coverage audit only*, never for
selecting which documents an agent sees.

Install datasets and convert either its official training JSON (optional
--local-json) or the Hugging Face Parquet-backed training split:

    python -m pip install datasets
    python -m experiments.joint_dit_kv.prepare_hotpot \
      --out data/hotpot_distributed_train --agents 4 \
      --sentences-per-agent 3 --holdout-percent 5

Use the generated data/hotpot_distributed_train/train.jsonl and dev.jsonl
in the AE/joint/independent commands above. For longer evidence in AE,
start AE with --max-text-tokens 256, then stage 2 loads the same config.
The converter does not touch the official HotpotQA validation/test split;
do not use benchmark scores for model selection. HotpotQA is licensed
CC BY-SA 4.0; honor attribution and terms.

Before scientific comparison: add official metric (answer EM/F1), comparable
answer-format prompts across official LatentMAS, document coverage stratified
by agent, three paired seeds and a matched deterministic joint-attention
model. The minimal synthetic-smoke accuracy is not a publishable result.

## Causal evidence controls

After training a joint checkpoint, remove or swap exactly one agent's
private evidence using the SAME checkpoint and output scorer:

    python -m experiments.joint_dit_kv.run --command eval \
      --strategy joint --checkpoint runs/joint_dit/best.pt \
      --data data/joint_dit_smoke/dev.jsonl --ablate-agent 1 \
      --out runs/eval_ablate1

    python -m experiments.joint_dit_kv.run --command eval \
      --strategy joint --checkpoint runs/joint_dit/best.pt \
      --data data/joint_dit_smoke/dev.jsonl --swap-agent 1 \
      --out runs/eval_swap1

Indices are 0-based. Swap takes evidence from the next paired evaluation
example, not another agent of the same example.

"""Blockwise causal / bidirectional mask and tiny-Qwen KV integration contracts."""
import torch
from torch import nn
import pytest

from experiments.joint_dit_kv.block_model import (
    BlockConfig, block_causal_mask, BlockFlow, BlockJointSystem,
)
from experiments.joint_dit_kv.prepare_gsm8k import split_gsm8k_answer, split_for_id, make_toy
from experiments.joint_dit_kv.run_blocks import read_steps, math_exact


def test_attention_block_diagonal_and_strict_causality():
    # 2 prompt slots, block1 two positions, current block two positions.
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    mask = block_causal_mask(groups)
    assert not mask[0, 1] and not mask[1, 0]  # prompt bidirectional
    assert not mask[2, 3] and not mask[3, 2]  # block1 bidirectional
    assert not mask[4, 5] and not mask[5, 4]  # current block bidirectional
    assert mask[0, 4] and mask[1, 5]         # no future -> prompt
    assert mask[2, 4] and mask[3, 5]         # no future -> old block
    assert not mask[4, 0] and not mask[5, 2]  # current sees prompt + old


def test_flow_per_block_shapes_conditioning_and_gradients():
    torch.manual_seed(5)
    cfg = BlockConfig(latent_dim=32, slots=2, streams_per_block=2,
                      heads=4, dit_layers=1, sample_steps=2, max_blocks=3)
    model = BlockFlow(cfg)
    prompt = torch.randn(1, 2, 32)
    previous = torch.randn(1, 1, 4, 32)
    clean = torch.randn(1, 4, 32)
    loss = model.loss_one_block(clean, prompt, previous)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.denoiser.out[-1].weight.grad.abs().sum() > 0
    sampled = model.sample_one_block(prompt, previous)
    stop = model.stop_logit(sampled, prompt)
    assert sampled.shape == (1, 4, 32)
    assert stop.shape == (1,)


def test_gsm8k_split_and_single_prompt_data(tmp_path):
    steps, answer = split_gsm8k_answer(
        "Let x = 2+3 = <<2+3=5>>5.\nThen multiply by 2 to get 10.\n#### 10")
    assert steps == [
        "Let x = 2+3 = 5.", "Then multiply by 2 to get 10.", "Final answer: 10"
    ]
    assert answer == "10"
    assert split_for_id("question1", 5) == split_for_id("question1", 5)
    make_toy(tmp_path, train=4, dev=2)
    assert len(read_steps(str(tmp_path / "train.jsonl"))) == 4
    assert math_exact("The answer is 12", "12")
    assert not math_exact("The answer is 11", "12")


def test_tiny_qwen_autoregressive_blocks_kv_and_stop_grad():
    transformers = pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    class TinyTokenizer:
        eos_token_id = 1
        def __call__(self, s, add_special_tokens=False, truncation=False,
                     max_length=96, return_tensors="pt"):
            vals = [2 + ord(c) % 100 for c in s][:max_length]
            return {"input_ids": torch.tensor([vals], dtype=torch.long)}
        def decode(self, ids, skip_special_tokens=True):
            return "2"

    config = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=256, rms_norm_eps=1e-6)
    receiver = Qwen2ForCausalLM(config)
    cfg = BlockConfig(latent_dim=32, slots=2, streams_per_block=2,
                      max_agents=2, max_blocks=3, heads=4, dit_layers=1,
                      sample_steps=2, max_text_tokens=64)
    system = BlockJointSystem(receiver, TinyTokenizer(), cfg)
    device = torch.device("cpu")
    q = "What is 1 + 1?"
    steps = ["Add 1 and 1.", "Final answer: 2."]
    loss, metrics = system.block_losses(q, steps, "2", device)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["target_blocks"] == 1
    assert system.writer.net[-1].weight.grad is not None
    assert system.flow.denoiser.out[-1].weight.grad is not None
    assert system.flow.stop[-1].weight.grad is not None
    assert all(x.grad is None for x in receiver.parameters())
    sample, logit = system.sample_blocks(system.prompt_latents(q, device),
                                         force_blocks=2)
    assert sample.shape == (1, 2, 4, 32)
    assert logit.shape == (1, 2)

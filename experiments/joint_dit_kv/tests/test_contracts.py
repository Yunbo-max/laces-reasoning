"""CPU-friendly mechanism/unit tests. Tiny-Qwen integration is optional if transformers installed."""
import pytest
import torch
from torch import nn

from experiments.joint_dit_kv.data import Example, check_disjoint, exact_match, make_toy, load_examples
from experiments.joint_dit_kv.model import JointConfig, JointFlow, QueryEncoder, KVWriter


def test_flow_bidirectional_shapes_and_gradients():
    torch.manual_seed(4)
    cfg = JointConfig(latent_dim=32, slots=2, max_agents=4, heads=4,
                      dit_layers=1, sample_steps=2)
    f = JointFlow(cfg)
    condition = torch.randn(1, 4, 2, 32)  # three agents plus answer slot
    target = torch.randn_like(condition)
    loss = f.training_loss(target, condition)
    loss.backward()
    assert loss.isfinite()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in f.parameters())
    z = f.sample(condition)
    assert z.shape == condition.shape and z.isfinite().all()


def test_query_encoder_reconstruction_gradient():
    cfg = JointConfig(latent_dim=32, slots=2, heads=4, max_text_tokens=24)
    embedding = nn.Embedding(80, 40).requires_grad_(False)
    encoder = QueryEncoder(40, cfg)
    z = encoder(torch.tensor([[3, 4, 5, 8, 9]]), embedding)
    z.square().sum().backward()
    assert z.shape == (1, 2, 32)
    assert embedding.weight.grad is None
    assert encoder.proj.weight.grad.abs().sum() > 0


def test_kv_writer_shapes():
    class Decoder:
        num_hidden_layers = 3
        num_key_value_heads = 2
        head_dim = 8
        num_attention_heads = 4
        hidden_size = 32

    writer = KVWriter(JointConfig(latent_dim=32, slots=2), Decoder())
    z = torch.randn(1, 6, 32)
    out = writer(z)
    assert out.shape == (3, 2, 1, 2, 6, 8)
    out.square().sum().backward()
    assert writer.net[-1].weight.grad is not None


def test_data_and_leakage_guard(tmp_path):
    make_toy(tmp_path, train_count=8, dev_count=4)
    train = load_examples(tmp_path / "train.jsonl")
    dev = load_examples(tmp_path / "dev.jsonl")
    check_disjoint(train, dev)
    with pytest.raises(ValueError):
        check_disjoint(train, train)
    assert len(train[0].agents) == 4
    assert exact_match("blue", "blue", ["blue", "green"])
    assert not exact_match("blue green", "blue", ["blue", "green"])
    assert not exact_match("a blue sky", "blue")


def test_small_qwen_cache_grad_through_writer():
    transformers = pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from experiments.joint_dit_kv.model import JointSystem

    class TinyTokenizer:
        eos_token_id = 1
        def __call__(self, s, add_special_tokens=False, truncation=False,
                     max_length=96, return_tensors="pt"):
            vals = [2 + ord(c) % 100 for c in s][:max_length]
            return {"input_ids": torch.tensor([vals], dtype=torch.long)}
        def decode(self, token_ids, skip_special_tokens=True):
            return "generated"

    config = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=128,
        rms_norm_eps=1e-6,
    )
    llm = Qwen2ForCausalLM(config)
    system = JointSystem(llm, TinyTokenizer(), JointConfig(
        latent_dim=32, slots=2, max_agents=4, heads=4, dit_layers=1,
        max_text_tokens=48, sample_steps=2
    ))
    c = system.encode_conditions(["key=A", "A maps to blue"], "What color?", torch.device("cpu"))
    latent = system.flow.sample(c)
    loss = system.receiver_loss(latent, "What color?", "blue", torch.device("cpu"))
    loss.backward()
    assert torch.isfinite(loss)
    assert system.writer.net[-1].weight.grad is not None
    assert system.flow.net.out[-1].weight.grad is not None
    assert all(p.grad is None for p in llm.parameters())


def test_hotpot_conversion_uses_no_support_labels():
    from experiments.joint_dit_kv.prepare_hotpot import build_example
    sample = {
        "id": "hotpot-1",
        "question": "Where was the Alpha telescope built?",
        "answer": "Cambridge",
        "context": {
            "title": ["Alpha telescope", "Building", "Random", "Other"],
            "sentences": [
                ["The Alpha telescope was created by Alice."],
                ["Alice built it in Cambridge."],
                ["A random city was mentioned."],
                ["The invention was famous."],
            ],
        },
        "supporting_facts": {"title": ["Alpha telescope", "Building"], "sent_id": [0, 0]},
    }
    before = build_example(sample, 3, 2)
    sample["supporting_facts"] = {"title": ["Other"], "sent_id": [0]}
    after = build_example(sample, 3, 2)
    assert before[0]["agents"] == after[0]["agents"]
    assert before[0]["question"] == after[0]["question"]
    assert before[1:] != after[1:]

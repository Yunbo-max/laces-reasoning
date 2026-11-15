"""Official LatentMAS sequential-relay adapter on the exact SAME JSONL examples.

This deliberately imports the official ModelWrapper instead of mislabeling independent
latent concatenation as "LatentMAS". Official wrapper uses stochastic text generation;
record that fact when comparing to greedy joint-KV evaluations.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from .data import load_examples, exact_match


def load_official_wrapper(source: str):
    path = Path(source).resolve() / "models.py"
    if not path.is_file():
        raise FileNotFoundError("Clone https://github.com/Gen-Verse/LatentMAS first: " + str(path))
    spec = importlib.util.spec_from_file_location("_latentmas_official_models", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import official LatentMAS/models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ModelWrapper, module._past_length


def kv_bytes(cache):
    if cache is None:
        return 0
    # Legacy tuple caches in official LatentMAS, or modern DynamicCache layers.
    if hasattr(cache, "layers"):
        total = 0
        for layer in cache.layers:
            for name in ("keys", "values"):
                x = getattr(layer, name, None)
                if torch.is_tensor(x):
                    total += x.numel() * x.element_size()
        return total
    if isinstance(cache, (tuple, list)):
        return sum(x.numel() * x.element_size()
                   for layer in cache for x in layer[:2] if torch.is_tensor(x))
    return 0


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="LatentMAS", help="Checkout of Gen-Verse/LatentMAS")
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--latent-steps", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-agents", type=int, default=8)
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    Wrapper, past_length = load_official_wrapper(args.source)
    opts = SimpleNamespace(
        latent_space_realign=False, device=device, device2=device, use_vllm=False,
        use_second_HF_model=False, tensor_parallel_size=1,
        gpu_memory_utilization=0.9, enable_prefix_caching=False,
        method="latent_mas", task="custom", max_new_tokens=args.max_new_tokens,
        think=False, latent_only=False, sequential_info_only=False,
    )
    model = Wrapper(args.model, device, use_vllm=False, args=opts)
    rows = []
    for idx, ex in enumerate(load_examples(args.data, args.max_agents)):
        torch.manual_seed(args.seed + idx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        past = None
        # LatentMAS has a sequential order: each successor receives predecessor's KV.
        for i, evidence in enumerate(ex.agents):
            prompt = ("Question: " + ex.question + "\n" +
                      "You are Agent " + str(i + 1) + ". Private evidence: " + evidence +
                      "\nConsider this evidence without outputting a final answer.")
            enc = model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            past = model.generate_latent_batch(
                enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                latent_steps=args.latent_steps, past_key_values=past,
            )
        prompt = "Question: " + ex.question + "\nAnswer:"
        enc = model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        replies, _ = model.generate_text_batch(
            enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            past_key_values=past,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        predicted = replies[0]
        rows.append({
            "id": ex.id, "strategy": "latentmas_official_sequential",
            "prediction": predicted, "gold": ex.answer,
            "correct": exact_match(predicted, ex.answer, ex.choices),
            "time_ms": (time.perf_counter() - start) * 1000,
            "cached_tokens": int(past_length(past)),
            "kv_resident_bytes": kv_bytes(past),
            "model": args.model, "temperature": args.temperature,
            "top_p": args.top_p, "latent_steps": args.latent_steps,
        })
        if (idx + 1) % 10 == 0:
            print("LatentMAS completed " + str(idx + 1), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "n": len(rows), "accuracy": sum(r["correct"] for r in rows)/max(1,len(rows)),
        "mean_time_ms": sum(r["time_ms"] for r in rows)/max(1,len(rows)),
        "mean_kv_resident_bytes": sum(r["kv_resident_bytes"] for r in rows)/max(1,len(rows)),
        "note": "Official sequential LatentMAS; stochastic receiver sampling by its API. "
                "KV resident bytes are NOT measured network traffic.",
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

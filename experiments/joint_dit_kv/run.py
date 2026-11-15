"""Train and evaluate Centralized Joint DiT -> Virtual KV -> Frozen Qwen.

Examples:
  python -m experiments.joint_dit_kv.data --make-toy data/joint_dit_smoke
  python -m experiments.joint_dit_kv.run --command train --stage ae --train data/joint_dit_smoke/train.jsonl --dev data/joint_dit_smoke/dev.jsonl --out runs/ae
  python -m experiments.joint_dit_kv.run --command train --stage joint --init-ae runs/ae/last.pt --train data/joint_dit_smoke/train.jsonl --dev data/joint_dit_smoke/dev.jsonl --out runs/joint
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .data import load_examples, check_disjoint, exact_match
from .model import JointConfig, JointSystem


def load_frozen_model(model_id: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Eager attention makes virtual-prefix cache contracts easy to audit.
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    receiver = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="eager"
    ).to(device)
    receiver.eval().requires_grad_(False)
    return receiver, tokenizer


def load_weights(system: JointSystem, ckpt: dict):
    source = ckpt["trainable"]
    absent, unexpected = system.load_state_dict(source, strict=False)
    if unexpected or any(not key.startswith("receiver.") for key in absent):
        raise ValueError("Checkpoint trainable-key mismatch: " + str((absent[:5], unexpected[:5])))


def make_system(args, device, ckpt=None):
    cfg = (JointConfig(**ckpt["config"]) if ckpt is not None else
           JointConfig(latent_dim=args.dim, slots=args.slots, max_agents=args.max_agents,
                       max_text_tokens=args.max_text_tokens, heads=args.heads,
                       dit_layers=args.layers, sample_steps=(args.sample_steps or 4)))
    if ckpt is not None and args.sample_steps is not None:
        cfg.sample_steps = args.sample_steps
    model_id = ckpt["model_id"] if ckpt is not None else args.model
    receiver, tokenizer = load_frozen_model(model_id, device)
    system = JointSystem(receiver, tokenizer, cfg).to(device)
    if ckpt is not None:
        load_weights(system, ckpt)
    return system, model_id


def save_checkpoint(system, model_id, stage, path, step, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 1, "model_id": model_id, "stage": stage,
        "config": asdict(system.cfg), "step": step,
        "trainable": {k: v.cpu() for k, v in system.compact_state_dict().items()},
        "metadata": extra or {},
    }, path)


def raw_generate(system, prompt: str, device, new_tokens: int,
                 temperature: float = 0.0, top_p: float = 0.9):
    tokens = system.tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                              truncation=True, max_length=2048)["input_ids"].to(device)
    kwargs = {"input_ids": tokens, "max_new_tokens": new_tokens,
              "pad_token_id": system.tokenizer.eos_token_id}
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kwargs.update(do_sample=False)
    ids = system.receiver.generate(**kwargs)
    return system.tokenizer.decode(ids[0, tokens.size(1):], skip_special_tokens=True).strip()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(system, data, device, strategy, max_new, out_file=None, limit=None,
             seed=42, temperature: float = 0.0, top_p: float = 0.9,
             ablate_agent: int = -1, swap_agent: int = -1):
    system.eval()
    counts = []
    rows = []
    eval_set = data if limit is None else data[:limit]
    for ix, example in enumerate(eval_set):
        torch.manual_seed(seed + ix)
        agent_texts = list(example.agents)
        if 0 <= ablate_agent < len(agent_texts):
            agent_texts[ablate_agent] = "No private evidence available."
        if 0 <= swap_agent < len(agent_texts):
            if len(eval_set) < 2:
                raise ValueError("swapped-agent control requires >=2 examples")
            donor = eval_set[(ix + 1) % len(eval_set)].agents
            if len(donor) <= swap_agent:
                raise ValueError("Agent count mismatch in swapped evidence control")
            agent_texts[swap_agent] = donor[swap_agent]
        sync(device)
        t0 = time.perf_counter()
        if strategy in ("joint", "independent", "mean"):
            conditions = system.encode_conditions(agent_texts, example.question, device)
            latents = system.latent(conditions, strategy=strategy)
            answer = system.generate(latents, example.question, device, max_new_tokens=max_new,
                                     temperature=temperature, top_p=top_p)
            # This is a virtual KV prefix; count transmitted latent content separately.
            latent_bytes = len(agent_texts) * system.cfg.slots * system.cfg.latent_dim * 2
            input_tokens = sum(len(system.tokenizer.encode(s, add_special_tokens=False))
                               for s in agent_texts)
        elif strategy == "full_text":
            prompt = ("Evidence:\n" +
                      "\n".join("Agent " + str(i+1) + ": " + a for i, a in enumerate(agent_texts)) +
                      "\nQuestion: " + example.question + "\nAnswer:")
            answer = raw_generate(system, prompt, device, max_new,
                                  temperature=temperature, top_p=top_p)
            latent_bytes = 0
            input_tokens = len(system.tokenizer.encode(prompt, add_special_tokens=False))
        elif strategy == "question_only":
            prompt = "Question: " + example.question + "\nAnswer:"
            answer = raw_generate(system, prompt, device, max_new)
            latent_bytes = 0
            input_tokens = len(system.tokenizer.encode(prompt, add_special_tokens=False))
        else:
            raise ValueError("Unsupported strategy " + strategy)
        sync(device)
        elapsed = (time.perf_counter() - t0) * 1000
        correct = exact_match(answer, example.answer, example.choices)
        counts.append(int(correct))
        rows.append({
            "id": example.id, "strategy": strategy, "prediction": answer,
            "gold": example.answer, "correct": correct, "time_ms": elapsed,
            "latent_payload_bytes": latent_bytes,
            "local_evidence_tokens": input_tokens,
            "ablate_agent_index": ablate_agent,
            "swapped_agent_index": swap_agent,
        })
    result = {
        "strategy": strategy, "n": len(rows),
        "accuracy": sum(counts) / max(1, len(counts)),
        "latency_mean_ms": sum(r["time_ms"] for r in rows) / max(1, len(rows)),
        "latent_payload_bytes_per_example": (sum(r["latent_payload_bytes"] for r in rows)
                                             / max(1, len(rows))),
        "note": "Prototype conservative exact-match scorer; use benchmark official metric for papers",
        "ablate_agent_index": ablate_agent,
        "swapped_agent_index": swap_agent,
    }
    if out_file:
        path = Path(out_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        path.with_suffix(".summary.json").write_text(json.dumps(result, indent=2))
    return result


def train(args, device):
    if args.stage != "ae" and not args.init_ae:
        raise ValueError("Joint/independent training requires a trained --init-ae checkpoint")
    base = torch.load(args.init_ae, map_location="cpu", weights_only=False) if args.init_ae else None
    system, model_id = make_system(args, device, base)
    train_data = load_examples(args.train, system.cfg.max_agents)
    dev_data = load_examples(args.dev, system.cfg.max_agents)
    check_disjoint(train_data, dev_data)
    if not train_data or not dev_data:
        raise ValueError("Empty train or dev data")
    if args.train == args.dev:
        raise ValueError("Training and dev files must differ")

    if args.stage == "ae":
        system.flow.requires_grad_(False)
        stage_modules = (system.encoder, system.writer)
    else:
        system.encoder.requires_grad_(False)  # fixed target manifold after AE stage
        if args.stage == "independent":
            system.flow.requires_grad_(False)
            stage_modules = (system.writer,)
        else:
            system.flow.requires_grad_(True)
            stage_modules = (system.flow, system.writer)
    parameters = [p for module in stage_modules for p in module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    system.train()
    rng = random.Random(args.seed)
    best_accuracy = None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(
        {"args": vars(args), "model": model_id, "config": asdict(system.cfg)}, indent=2
    ))

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        batch_losses = []
        for _ in range(args.grad_accum):
            ex = rng.choice(train_data)
            device = next(system.writer.parameters()).device
            if args.stage == "ae":
                # Random agent evidence; same AE can be reused for answer targets.
                target = rng.choice(ex.agents + [ex.answer])
                loss = system.ae_loss(target, device)
            else:
                conditions = system.encode_conditions(ex.agents, ex.question, device)
                if args.stage == "joint":
                    clean = system.gold_targets(conditions, ex.answer, device)
                    flow_loss = system.flow.training_loss(clean, conditions)
                    predicted = system.flow.sample(conditions)
                else:
                    flow_loss = torch.zeros((), device=device)
                    predicted = conditions
                receiver_ce = system.receiver_loss(predicted, ex.question, ex.answer, device)
                # Keep the writer's acquired autoencoding behavior when fine-tuning it.
                evidence = rng.choice(ex.agents)
                loss = (args.diff_weight * flow_loss + args.receiver_weight * receiver_ce +
                        args.ae_weight * system.ae_loss(evidence, device))
            (loss / args.grad_accum).backward()
            batch_losses.append(float(loss.detach()))
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"step": step, "stage": args.stage,
                              "loss": sum(batch_losses)/len(batch_losses),
                              "grad_norm": float(grad_norm)}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            save_checkpoint(system, model_id, args.stage, out / "last.pt", step)
            if args.stage == "ae":
                system.eval()
                with torch.no_grad():
                    scores = [float(system.ae_loss(rng.choice(x.agents), device))
                              for x in dev_data[:args.eval_limit]]
                metric = {"ae_dev_ce": sum(scores)/len(scores)}
                # AE-only phase selects by reconstruction CE.
                score = -metric["ae_dev_ce"]
            else:
                with torch.no_grad():
                    metric = evaluate(system, dev_data, device,
                                      "joint" if args.stage == "joint" else "independent",
                                      args.max_new_tokens, limit=args.eval_limit, seed=args.seed)
                score = metric["accuracy"]
            if best_accuracy is None or score > best_accuracy:
                best_accuracy = score
                save_checkpoint(system, model_id, args.stage, out / "best.pt", step, metric)
            (out / "dev_latest.json").write_text(json.dumps({"step": step, **metric}, indent=2))
            print("DEV " + json.dumps({"step": step, **metric}), flush=True)
            system.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", choices=["train", "eval"], required=True)
    ap.add_argument("--stage", choices=["ae", "joint", "independent"], default="joint")
    ap.add_argument("--strategy", choices=["joint", "independent", "mean", "full_text", "question_only"],
                    default="joint")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train", default="")
    ap.add_argument("--dev", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--init-ae", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out", default="runs/joint_dit")
    ap.add_argument("--prediction-file", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--max-agents", type=int, default=8)
    ap.add_argument("--max-text-tokens", type=int, default=96)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--sample-steps", type=int, default=None,
                    help="AE default=4; override pretrained config in stage2 if set")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--diff-weight", type=float, default=1.0)
    ap.add_argument("--receiver-weight", type=float, default=1.0)
    ap.add_argument("--ae-weight", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-limit", type=int, default=24)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Final eval sampling; checkpoint selection is greedy")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--ablate-agent", type=int, default=-1,
                    help="Zero one agent evidence during eval; 0-indexed")
    ap.add_argument("--swap-agent", type=int, default=-1,
                    help="Swap one agent evidence with next example during eval; 0-indexed")
    args = ap.parse_args()
    if args.eval_every < 1 or args.grad_accum < 1:
        ap.error("eval-every and grad-accum must be positive")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if args.command == "train":
        if not (args.train and args.dev):
            ap.error("train requires --train and --dev")
        train(args, device)
    else:
        if not args.data:
            ap.error("eval requires --data")
        ckpt = (torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                if args.checkpoint else None)
        if args.strategy in ("joint", "independent", "mean") and ckpt is None:
            ap.error("latent strategies require --checkpoint; random writers are not baselines")
        system, _ = make_system(args, device, ckpt)
        dataset = load_examples(args.data, system.cfg.max_agents)
        with torch.no_grad():
            result = evaluate(system, dataset, device, args.strategy, args.max_new_tokens,
                              out_file=args.prediction_file or str(Path(args.out)/"predictions.jsonl"),
                              seed=args.seed, temperature=args.temperature,
                              top_p=args.top_p, ablate_agent=args.ablate_agent,
                              swap_agent=args.swap_agent)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Single-prompt, block-autoregressive latent diffusion -> virtual KV -> frozen Qwen.

Training:
  stage ae:    exact step-token reconstruction CE via frozen Qwen and KV writer
  stage block: teacher-forced causal flow MSE + predicted-block receiver CE + STOP BCE

No MTP, no 64K pretraining. Blocks are AR; individual block slots are bidirectional.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .block_model import BlockConfig, BlockJointSystem
from .run import load_frozen_model


def read_steps(path: str) -> list[dict]:
    result = []
    seen = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("question") or not row.get("answer") or not row.get("steps"):
            raise ValueError("Expected single-prompt rows with question, answer, steps")
        if not isinstance(row["steps"], list) or not all(isinstance(s, str) for s in row["steps"]):
            raise ValueError("steps must be an array of text reasoning steps")
        identifier = str(row.get("id", row["question"]))
        if identifier in seen:
            raise ValueError("Duplicate example ID " + identifier)
        seen.add(identifier)
        result.append(row)
    return result


def numeric_answer(text: str):
    box = re.findall(r"\\boxed\{([^}]*)\}", str(text))
    if box:
        text = box[-1]
    digits = re.findall(r"(?<![a-zA-Z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?", str(text))
    if not digits:
        return None
    return digits[-1].replace(",", "")


def math_exact(pred: str, gold: str):
    a, b = numeric_answer(pred), numeric_answer(gold)
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except ValueError:
        return a == b


def load_checkpoint(system: BlockJointSystem, checkpoint: dict, only_ae: bool = False):
    weights = checkpoint["trainable"]
    if only_ae:
        weights = {k: v for k, v in weights.items()
                   if k.startswith("encoder.") or k.startswith("writer.")}
        if not weights:
            raise ValueError("No pretrained AE or KV writer found in checkpoint")
    missing, unexpected = system.load_state_dict(weights, strict=False)
    if unexpected:
        raise ValueError("Unexpected keys in checkpoint: " + repr(unexpected[:8]))
    if not only_ae and any(not x.startswith("receiver.") for x in missing):
        raise ValueError("Missing trainable keys: " + repr(missing[:8]))


def setup(args, device, checkpoint=None):
    if checkpoint and checkpoint.get("kind") == "blockwise":
        cfg = BlockConfig(**checkpoint["config"])
        name = checkpoint["model_id"]
    else:
        cfg = BlockConfig(latent_dim=args.dim, slots=args.slots,
                          max_agents=args.streams_per_block, max_text_tokens=args.max_text_tokens,
                          heads=args.heads, dit_layers=args.layers,
                          sample_steps=args.flow_steps,
                          streams_per_block=args.streams_per_block,
                          max_blocks=args.max_blocks, stop_threshold=args.stop_threshold)
        name = args.model
    if args.flow_steps_override:
        cfg.sample_steps = args.flow_steps_override
    decoder, tokenizer = load_frozen_model(name, device)
    system = BlockJointSystem(decoder, tokenizer, cfg).to(device)
    if checkpoint and checkpoint.get("kind") == "blockwise":
        load_checkpoint(system, checkpoint)
    return system, name


def save(system, model_name, stage, folder, step, metadata=None, name="last.pt"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "blockwise", "version": 1, "stage": stage,
        "model_id": model_name, "config": asdict(system.cfg), "step": step,
        "trainable": {k: v.detach().cpu() for k, v in
                      system.compact_state_dict().items()},
        "metadata": metadata or {},
    }
    torch.save(payload, folder / name)


@torch.no_grad()
def evaluate(system: BlockJointSystem, data: list[dict], device, max_new=48,
             limit=None, force_blocks=None, output=None, seed=42):
    system.eval()
    rows = []
    for i, x in enumerate(data[:limit] if limit else data):
        torch.manual_seed(seed + i)
        if device.type == "cuda":
            torch.cuda.synchronize()
        import time
        start = time.perf_counter()
        prediction, extra = system.answer(x["question"], device,
                                          max_new_tokens=max_new,
                                          force_blocks=force_blocks)
        if device.type == "cuda":
            torch.cuda.synchronize()
        rows.append({
            "id": x.get("id", i), "prompt": x["question"],
            "pred": prediction, "gold": x["answer"],
            "correct": math_exact(prediction, x["answer"]),
            "time_ms": (time.perf_counter() - start) * 1000, **extra,
        })
    summary = {
        "n": len(rows), "accuracy": sum(r["correct"] for r in rows)/max(len(rows), 1),
        "mean_blocks": sum(r["num_blocks"] for r in rows)/max(len(rows), 1),
        "mean_ms": sum(r["time_ms"] for r in rows)/max(len(rows), 1),
        "forced_blocks": force_blocks,
    }
    if output:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        p.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def train(args, device):
    trainset, devset = read_steps(args.train), read_steps(args.dev)
    if not trainset or not devset:
        raise ValueError("Empty training or validation split")
    ids = {x.get("id", x["question"]) for x in trainset}
    if ids.intersection(x.get("id", x["question"]) for x in devset):
        raise ValueError("Train/dev ID overlap")
    if set(x["question"] for x in trainset).intersection(x["question"] for x in devset):
        raise ValueError("Train/dev identical question overlap")
    system, model_name = setup(args, device)
    if args.stage == "block":
        if not args.init_ae:
            raise ValueError("--init-ae is required for block training")
        prev = torch.load(args.init_ae, map_location="cpu", weights_only=False)
        if prev.get("model_id") != model_name:
            raise ValueError("AE and block receiver model identifiers differ")
        if prev.get("kind") == "blockwise":
            if prev["config"]["latent_dim"] != system.cfg.latent_dim:
                raise ValueError("AE latent width differs")
        load_checkpoint(system, prev, only_ae=True)

    if args.stage == "ae":
        system.flow.requires_grad_(False)
        parameters = list(system.encoder.parameters()) + list(system.writer.parameters())
    else:
        system.encoder.requires_grad_(False)
        parameters = list(system.flow.parameters()) + list(system.writer.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    rng = random.Random(args.seed)
    best = None
    for step in range(1, args.steps+1):
        system.train()
        optimizer.zero_grad(set_to_none=True)
        parts = []
        for _ in range(args.accum):
            row = rng.choice(trainset)
            if args.stage == "ae":
                # Each step has one local latent; train exact per-step reconstruction.
                text = rng.choice(row["steps"])
                loss = system.ae_loss(text, device)
                metrics = {}
            else:
                loss, metrics = system.block_losses(
                    row["question"], row["steps"], row["answer"], device,
                    ce_weight=args.ce_weight, stop_weight=args.stop_weight)
            (loss/args.accum).backward()
            parts.append(float(loss.detach()))
        grad = torch.nn.utils.clip_grad_norm_(parameters, args.clip)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({
                "step": step, "stage": args.stage,
                "train_loss": sum(parts)/len(parts), "grad_norm": float(grad),
                **{k:float(v) for k,v in metrics.items()},
            }), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            save(system, model_name, args.stage, out, step)
            system.eval()
            with torch.no_grad():
                if args.stage == "ae":
                    scores = [float(system.ae_loss(row["steps"][0], device))
                              for row in devset[:args.eval_limit]]
                    metric = {"ae_dev_ce": sum(scores)/len(scores)}
                    objective = -metric["ae_dev_ce"]
                else:
                    metric = evaluate(system, devset, device, max_new=args.max_new,
                                      limit=args.eval_limit, seed=args.seed)
                    objective = metric["accuracy"]
            if best is None or objective > best:
                best = objective
                save(system, model_name, args.stage, out, step, metric, "best.pt")
            (out / "dev_latest.json").write_text(
                json.dumps({"step":step,**metric},indent=2))
            print("DEV "+json.dumps({"step":step,**metric}),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", choices=("train","eval"), required=True)
    parser.add_argument("--stage", choices=("ae","block"), default="block")
    parser.add_argument("--train",default="")
    parser.add_argument("--dev",default="")
    parser.add_argument("--data",default="")
    parser.add_argument("--init-ae",default="")
    parser.add_argument("--checkpoint",default="")
    parser.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out",default="runs/block_joint")
    parser.add_argument("--dim",type=int,default=128)
    parser.add_argument("--slots",type=int,default=4)
    parser.add_argument("--streams-per-block",type=int,default=2)
    parser.add_argument("--max-blocks",type=int,default=4)
    parser.add_argument("--stop-threshold",type=float,default=0.5)
    parser.add_argument("--max-text-tokens",type=int,default=128)
    parser.add_argument("--heads",type=int,default=4)
    parser.add_argument("--layers",type=int,default=4)
    parser.add_argument("--flow-steps",type=int,default=2)
    parser.add_argument("--flow-steps-override",type=int,default=0)
    parser.add_argument("--steps",type=int,default=1000)
    parser.add_argument("--accum",type=int,default=1)
    parser.add_argument("--lr",type=float,default=2e-4)
    parser.add_argument("--ce-weight",type=float,default=1.0)
    parser.add_argument("--stop-weight",type=float,default=0.2)
    parser.add_argument("--clip",type=float,default=1.0)
    parser.add_argument("--eval-every",type=int,default=200)
    parser.add_argument("--eval-limit",type=int,default=32)
    parser.add_argument("--log-every",type=int,default=20)
    parser.add_argument("--max-new",type=int,default=48)
    parser.add_argument("--force-blocks",type=int,default=0)
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args()
    if args.accum < 1 or args.eval_every < 1 or args.max_blocks < 1:
        parser.error("Invalid accumulation / eval interval / max blocks")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if args.command=="train":
        if not args.train or not args.dev:
            parser.error("--train and --dev required")
        train(args, device)
    else:
        if not args.data or not args.checkpoint:
            parser.error("--data and --checkpoint required")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if ckpt.get("kind") != "blockwise":
            parser.error("Blockwise evaluation requires trained blockwise checkpoint")
        system,_ = setup(args, device, ckpt)
        data = read_steps(args.data)
        result = evaluate(system, data, device, max_new=args.max_new,
                          force_blocks=args.force_blocks or None,
                          output=str(Path(args.out)/"predictions.jsonl"),
                          seed=args.seed)
        print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

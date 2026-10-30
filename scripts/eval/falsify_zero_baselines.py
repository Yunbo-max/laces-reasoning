#!/usr/bin/env python3
"""Falsification probe for the near-zero long-context baseline rows.

Question this answers, per model: is the published ~0 a genuine capability limit, or an
artefact of OUR harness? Written as a SEPARATE script on purpose -- three other agents are
editing ``run_open_context_bench.py`` concurrently, so nothing here mutates it. The scoring
rule, the prompt builder and the truncation shape are COPIED from it verbatim (verified:
this file's ``babilong_score`` reproduces all 3600 published row scores bit for bit) so the
numbers are comparable to the published ones.

Modes (``--mode``):

  fingerprint   For an fla model (rwkv7 / gla): run HF ``generate`` unpatched, patched, and a
                hand-written single-token recurrent loop on the SAME prompts, and report
                per-position token agreement. The bug's signature (memory
                fla-generate-recurrent-cache-bug.md) is "first token 100% identical, collapse
                from the 2nd onward", so agreement is reported per prefix length k.

  rerun         Re-measure bAbILong under a chosen decode path and input budget, and score
                with the harness's own ``babilong_score``. ``--decode {hf,hf_patched,loop}``
                and ``--budget`` are the two axes. ``--budget`` may be an int or ``window``
                (= the model's real positional ceiling minus max_new_tokens).

  matcher       No GPU. Re-score existing result JSONs under progressively looser matchers to
                bound how much the scorer could possibly be hiding. Pure analysis.

Every mode writes a JSON to ``--output``.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------------------
# COPIED from run_open_context_bench.py so this probe cannot drift from the published metric.
# ---------------------------------------------------------------------------------------
BABILONG_TASK_LABELS = {
    "qa1": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa2": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa3": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa6": ["no", "yes"],
    "qa9": ["no", "yes"],
}


def babilong_score(task: str, prediction: str, target: str, question: str) -> float:
    """BABILong's official label-aware ``compare_answers``. Verbatim from the harness."""
    output = prediction.lower().split(".")[0]
    output = output.split("<context>")[0].split("<example>")[0].split("question")[0]
    labels = {label.lower() for label in BABILONG_TASK_LABELS[task]}
    labels_in_question = {label for label in labels if label in question.lower()}
    labels_in_output = {label for label in labels if label in output} - labels_in_question
    target = target.lower()
    if "," in target and len(target) > 3:
        subtargets = target.split(",")
        return float(all(item in labels_in_output for item in subtargets) and
                     len(labels_in_output) == len(subtargets))
    return float(target in labels_in_output and len(labels_in_output) == 1)


def load_babilong_rows(path: Path) -> list[dict]:
    items = json.loads(path.read_text())
    return [{
        "id": index,
        "prompt": f"{item['input'].rstrip()}\nQuestion: {item['question'].strip()}\nAnswer:",
        "question": item["question"].strip(),
        "answers": [str(item["target"])],
    } for index, item in enumerate(items)]


def truncate_ids(tokenizer, prompt: str, max_input_tokens: int, device: str,
                 add_special_tokens: bool = False) -> torch.Tensor:
    """Middle-truncate to ``max_input_tokens`` (LongBench's shape), as the harness does.

    ``add_special_tokens`` is exposed because the harness hardcodes False, which silently
    drops the BOS token that gemma-2 and gla prepend -- both were trained with it always
    present. That is a candidate artefact, so it has to be switchable to be testable.
    """
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=add_special_tokens).input_ids[0]
    if ids.numel() > max_input_tokens:
        left = max_input_tokens // 2
        ids = torch.cat([ids[:left], ids[-(max_input_tokens - left):]])
    if ids.numel() < 2:
        raise ValueError("Prompt must contain at least two tokens")
    return ids.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------------------
# The fla patch (copied from the harness, which copied it from the verified v2 job).
# ---------------------------------------------------------------------------------------
def is_fla_model(model) -> bool:
    return any("fla" in getattr(base, "__module__", "") for base in type(model).__mro__)


def patch_fla_recurrent_generate(model) -> bool:
    """Make HF ``generate`` correct on fla's recurrent caches. Returns True if patched."""
    cls = type(model)
    if not is_fla_model(model):
        return False
    original = cls.prepare_inputs_for_generation
    if getattr(original, "_fla_recurrent_patched", False):
        return True

    # The **kwargs parameter MUST be named exactly `kwargs`: transformers'
    # _validate_model_kwargs does `if "kwargs" in model_args: model_args |=
    # signature(self.forward)`, so naming it `kw` makes generate reject attention_mask.
    @functools.wraps(original)
    def fixed(self, input_ids=None, past_key_values=None, **kwargs):
        model_inputs = original(self, input_ids=input_ids,
                                past_key_values=past_key_values, **kwargs)
        if past_key_values is not None:
            try:
                seen = int(past_key_values.get_seq_length())
            except Exception:
                seen = 0
            if (seen > 0 and model_inputs.get("input_ids") is not None
                    and model_inputs["input_ids"].shape[1] > 1):
                model_inputs["input_ids"] = model_inputs["input_ids"][:, -1:].contiguous()
                model_inputs["cache_position"] = None
        return model_inputs

    fixed._fla_recurrent_patched = True
    cls.prepare_inputs_for_generation = fixed
    return True


def unpatch_fla(model) -> None:
    cls = type(model)
    current = cls.prepare_inputs_for_generation
    if getattr(current, "_fla_recurrent_patched", False):
        cls.prepare_inputs_for_generation = current.__wrapped__


@torch.no_grad()
def greedy_loop(model, prefix_ids: torch.Tensor, max_new_tokens: int,
                eos_id: int | None = None) -> list[int]:
    """Reference decode: prefill once, then feed exactly one token per step.

    This is the same shape as LACESRuntime._stream / GlobalRuntime.generate -- i.e. the path
    OUR OWN arms already use -- so a baseline decoded this way is protocol-matched to them.
    """
    out = model(input_ids=prefix_ids, use_cache=True, return_dict=True)
    cache, logits = out.past_key_values, out.logits[0, -1]
    ids: list[int] = []
    for index in range(max_new_tokens):
        token = int(logits.argmax())
        if eos_id is not None and token == eos_id:
            break
        ids.append(token)
        if index + 1 < max_new_tokens:
            out = model(input_ids=torch.tensor([[token]], device=prefix_ids.device,
                                               dtype=torch.long),
                        past_key_values=cache, use_cache=True, return_dict=True)
            cache, logits = out.past_key_values, out.logits[0, -1]
    return ids


@torch.no_grad()
def hf_generate(model, tokenizer, prefix_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
    out = model.generate(
        input_ids=prefix_ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=(getattr(tokenizer, "pad_token_id", None)
                      or getattr(tokenizer, "eos_token_id", None)),
    )
    return out[0, prefix_ids.shape[1]:].tolist()


def positional_ceiling(model_dir: str) -> tuple[int | None, str]:
    """The model's REAL positional ceiling, or None. Config-declared value + source check.

    Recurrent models (rwkv7 / gla / mamba) declare ``max_position_embeddings`` but never
    consume positions, so the field is inert; the probe is the running module's source.
    """
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
    except Exception as error:
        return None, f"config unreadable: {error!r}"
    declared = (config.get("max_position_embeddings") or config.get("n_positions")
                or config.get("n_ctx"))
    model_type = config.get("model_type")
    recurrent = {"mamba", "mamba2", "rwkv", "rwkv5", "rwkv6", "rwkv7", "gla",
                 "gated_deltanet", "retnet", "hgrn", "hgrn2", "delta_net", "linear_attn"}
    if model_type in recurrent and not config.get("attn"):
        return None, (f"model_type={model_type!r} recurrent, attn=null => declared "
                      f"{declared} is inert")
    if not declared:
        return None, f"model_type={model_type!r} declares no window"
    return int(declared), f"model_type={model_type!r} window={int(declared)}"


def load_model(model_dir: str, device: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
        model_type = config.get("model_type")
        has_auto_map = bool((config.get("auto_map") or {}).get("AutoModelForCausalLM"))
    except Exception:
        model_type, has_auto_map = None, False
    if model_type and model_type not in CONFIG_MAPPING and not has_auto_map:
        import fla.models  # noqa: F401
        print(f"[load] imported fla.models to register {model_type!r}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True,
                                             local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=dtype,
        local_files_only=True).to(device).eval()
    return model, tokenizer


# ---------------------------------------------------------------------------------------
# Mode: fingerprint
# ---------------------------------------------------------------------------------------
def run_fingerprint(args) -> dict:
    model, tokenizer = load_model(args.model_dir, args.device)
    fla = is_fla_model(model)
    print(f"[fingerprint] fla model: {fla}", flush=True)
    data_dir = Path(args.data_dir)
    rows = load_babilong_rows(data_dir / f"{args.dataset}.json")[:args.max_samples]
    task = args.dataset.split("/")[0]
    eos = getattr(tokenizer, "eos_token_id", None)

    records = []
    for index, row in enumerate(rows):
        # `--add_bos` MUST be threaded here, not only in rerun mode (line ~309). Without it
        # the flag was silently inert in fingerprint mode: the probe would report an
        # agreement measured at add_bos=False while claiming to describe BOS-prepended
        # published rows. Fingerprint evidence of the omission: gla-fingerprint.json row 0
        # has a 3977-token prefix, the BOS runs have 3978.
        prefix_ids = truncate_ids(tokenizer, row["prompt"], args.budget_int, args.device,
                                  add_special_tokens=args.add_bos)
        unpatch_fla(model)
        unp = hf_generate(model, tokenizer, prefix_ids, args.max_new_tokens)
        patch_fla_recurrent_generate(model)
        pat = hf_generate(model, tokenizer, prefix_ids, args.max_new_tokens)
        unpatch_fla(model)
        loop = greedy_loop(model, prefix_ids, args.max_new_tokens, eos_id=eos)
        rec = {
            "id": row["id"], "input_tokens": int(prefix_ids.shape[1]),
            "answers": row["answers"],
            "unpatched_ids": unp, "patched_ids": pat, "loop_ids": loop,
            "unpatched": tokenizer.decode(unp, skip_special_tokens=True).strip(),
            "patched": tokenizer.decode(pat, skip_special_tokens=True).strip(),
            "loop": tokenizer.decode(loop, skip_special_tokens=True).strip(),
        }
        for name, ids in (("unpatched", unp), ("patched", pat), ("loop", loop)):
            rec[f"score_{name}"] = babilong_score(
                task, tokenizer.decode(ids, skip_special_tokens=True).strip(),
                row["answers"][0], row["question"])
        records.append(rec)
        print(f"[fingerprint] {index + 1}/{len(rows)} "
              f"unp={rec['score_unpatched']} pat={rec['score_patched']} "
              f"loop={rec['score_loop']}", flush=True)

    # Per-prefix-length agreement: the bug's fingerprint is k=1 perfect, k>=2 collapsing.
    agreement = {}
    for pair in (("unpatched", "loop"), ("patched", "loop"), ("unpatched", "patched")):
        a, b = pair
        per_k = {}
        for k in (1, 2, 4, 8, 32):
            hits = total = 0
            for rec in records:
                xa, xb = rec[f"{a}_ids"][:k], rec[f"{b}_ids"][:k]
                if not xa and not xb:
                    continue
                total += 1
                hits += int(xa == xb)
            per_k[k] = {"identical": hits, "n": total}
        agreement[f"{a}_vs_{b}"] = per_k
    scores = {name: sum(r[f"score_{name}"] for r in records) / max(1, len(records)) * 100
              for name in ("unpatched", "patched", "loop")}
    return {"mode": "fingerprint", "is_fla": fla, "config": vars(args),
            "agreement_by_prefix_len": agreement, "scores_pct": scores, "rows": records}


# ---------------------------------------------------------------------------------------
# Mode: rerun
# ---------------------------------------------------------------------------------------
def run_rerun(args) -> dict:
    ceiling, ceiling_reason = positional_ceiling(args.model_dir)
    model, tokenizer = load_model(args.model_dir, args.device)
    if args.decode == "hf_patched":
        applied = patch_fla_recurrent_generate(model)
        print(f"[rerun] fla patch applied: {applied}", flush=True)
    else:
        unpatch_fla(model)
    if args.budget == "window":
        if ceiling is None:
            raise SystemExit(f"--budget window but this model has no ceiling ({ceiling_reason})")
        budget = max(2, ceiling - args.max_new_tokens)
    else:
        budget = int(args.budget)
    print(f"[rerun] ceiling={ceiling} ({ceiling_reason}) budget={budget} "
          f"decode={args.decode} bos={args.add_bos}", flush=True)

    data_dir = Path(args.data_dir)
    eos = getattr(tokenizer, "eos_token_id", None)
    aggregates, rows_out = {}, []
    for spec in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        rows = load_babilong_rows(data_dir / f"{spec}.json")[:args.max_samples]
        task, dataset = spec.split("/")[0], spec.replace("/", "_")
        scores = []
        for index, row in enumerate(rows):
            prefix_ids = truncate_ids(tokenizer, row["prompt"], budget, args.device,
                                      add_special_tokens=args.add_bos)
            if args.decode == "loop":
                ids = greedy_loop(model, prefix_ids, args.max_new_tokens, eos_id=eos)
            else:
                ids = hf_generate(model, tokenizer, prefix_ids, args.max_new_tokens)
            prediction = tokenizer.decode(ids, skip_special_tokens=True).strip()
            score = babilong_score(task, prediction, row["answers"][0], row["question"])
            scores.append(score)
            rows_out.append({"dataset": dataset, "id": row["id"],
                             "input_tokens": int(prefix_ids.shape[1]),
                             "prediction": prediction, "prediction_token_ids": ids,
                             "answers": row["answers"], "score": score})
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(f"[rerun/{dataset}] {index + 1}/{len(rows)} "
                      f"running={sum(scores) / len(scores) * 100:.1f}%", flush=True)
        aggregates[dataset] = {"n": len(scores),
                               "score": sum(scores) / len(scores) if scores else 0.0}
        print(f"[rerun] {dataset} = {aggregates[dataset]['score'] * 100:.1f}%", flush=True)
    return {"mode": "rerun", "config": vars(args),
            "positional_ceiling": ceiling, "ceiling_reason": ceiling_reason,
            "effective_budget": budget, "aggregates": aggregates, "rows": rows_out}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fingerprint", "rerun"), required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset", default="qa1/4k", help="fingerprint mode: one spec")
    parser.add_argument("--datasets", default="qa1/4k", help="rerun mode: comma-separated")
    parser.add_argument("--budget", default="16384", help="int, or 'window'")
    parser.add_argument("--budget_int", type=int, default=16384, help="fingerprint mode")
    parser.add_argument("--decode", choices=("hf", "hf_patched", "loop"), default="hf")
    parser.add_argument("--add_bos", action="store_true",
                        help="prepend the tokenizer's BOS (harness hardcodes off)")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    started = time.time()
    result = run_fingerprint(args) if args.mode == "fingerprint" else run_rerun(args)
    result["elapsed_seconds"] = time.time() - started
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2)[:4000],
          flush=True)


if __name__ == "__main__":
    main()

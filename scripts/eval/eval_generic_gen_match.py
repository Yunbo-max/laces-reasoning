#!/usr/bin/env python3
"""Generic generate-then-match harness for EXTERNAL HF causal LMs (baselines).

Runs the exact same prompts / generation settings as
run_cola_dlm_tasks_prefix_suffix_cfg.py (our models' generative protocol) so that
baseline numbers land in the same table:

  prompt      = few_shot_prefix + prompt     (identical construction)
  decoding    = greedy by default (temperature 0), max_new_tokens, repetition penalty
  output      = same jsonl record schema ({**item, id, prompt, generate,
                ground_truth, choices, prompt_tokens, generated_tokens, method,
                cfg_scale, steps}) so acc_calc / our scorers apply unchanged.

Supported: any HF AutoModelForCausalLM directory (Qwen2.5, Llama-3.2, gemma-2,
phi-2/Phi-3.5, mamba2, RWKV6/7, ...). Use --trust_remote_code for custom
architectures (RWKV, some Mamba releases).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

_TOPK_FILTER = None

def _load_hf_model(model_dir, device, trust_remote_code=True):
    """Load a baseline checkpoint, tolerating fla-hub architectures.

    fla-hub snapshots (e.g. fla-hub/gla-1.3B-100B) declare ``model_type: gla`` but ship no
    remote-code file, so AutoModelForCausalLM only resolves them once the ``fla`` package has
    registered its classes. Importing ``fla.models`` does that, but on a GPU-less host the
    import itself aborts inside triton ("0 active drivers") -- hence three tiers: plain
    AutoModel, then after importing fla, then the class named in ``config.architectures``.
    """
    from transformers import AutoModelForCausalLM
    # Force it on: several of our baselines (RWKV-7/RWKV-6) ship custom modelling code and
    # the callers pass a store_true flag that defaults to False. All checkpoints here are
    # local files under our own control.
    # Honour the dtype the checkpoint declares: mamba-2.8b ships float32 weights, and
    # loading them as bf16 makes the first forward die with
    # "expected mat1 and mat2 to have the same dtype, but got: float != BFloat16".
    _dt = torch.bfloat16
    try:
        _c = json.load(open(os.path.join(str(model_dir), "config.json")))
        _dt = {"float32": torch.float32, "float16": torch.float16,
               "bfloat16": torch.bfloat16}.get(str(_c.get("torch_dtype")))
        _dt = _dt or torch.bfloat16
    except Exception as _e:
        # Do NOT swallow silently: this block previously died on a missing module-level
        # `import os` (NameError) and the bare `pass` left _dt at bfloat16 for every
        # checkpoint, which is what actually broke mamba-2.8b's float32 weights.
        print(f"[load] dtype probe failed ({type(_e).__name__}: {_e}); using {_dt}",
              file=sys.stderr, flush=True)
    kw = dict(trust_remote_code=True, torch_dtype=_dt, local_files_only=True)
    try:
        return AutoModelForCausalLM.from_pretrained(model_dir, **kw).to(device).eval()
    except ValueError as e:
        if "does not recognize this architecture" not in str(e):
            raise
    import json as _json
    import os as _os
    arch = (_json.load(open(_os.path.join(model_dir, "config.json"))).get("architectures")
            or [None])[0]
    try:
        import fla.models as _fm
        if arch and hasattr(_fm, arch):
            return getattr(_fm, arch).from_pretrained(model_dir, **kw).to(device).eval()
        return AutoModelForCausalLM.from_pretrained(model_dir, **kw).to(device).eval()
    except Exception as e2:
        raise RuntimeError(f"cannot load HF checkpoint {model_dir}: {e2}") from e2


_OPTION_MARKER_RE = re.compile(r"\(A\)")


def inline_choices(prompt: str, item: dict) -> str:
    """Inline the item's ``choices`` when the prompt does not already carry them.

    Six tasks in the shared task data (arc_easy, arc_challenge, winogrande, sciq,
    commonsenseqa, piqa) ship a ``choices`` field but build their ``prompt`` WITHOUT
    it, so the model was asked to answer a question whose options it never saw --
    that is free guessing, not option selection, and it depressed every model's score
    on exactly those columns. The other choice tasks (mmlu, obqa, siqa, story_cloze,
    hellaswag, race) already carry a ``(A) ... / (B) ...`` block; the guard below
    detects that block and returns the prompt unchanged, which makes this function
    idempotent: calling it twice, or on an already-fixed prompt, is a no-op.
    """
    choices = item.get("choices") if isinstance(item, dict) else None
    if not isinstance(choices, (list, tuple)) or not choices:
        return prompt
    m = _OPTION_MARKER_RE.search(prompt)
    if m is not None and str(choices[0]).strip().lower() in prompt[m.start():].lower():
        return prompt  # already carries an option list -- never add a second copy
    block = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
    cut = prompt.rfind("Answer:")
    if cut == -1:
        return f"{prompt}\n{block}\nAnswer:"
    return f"{prompt[:cut].rstrip()}\n{block}\nAnswer:"


def build_prompt(task: str, item: dict) -> str:
    """Mirror of run_cola_dlm_tasks_prefix_suffix_cfg.build_prompt: prefer the
    pre-built 'prompt' field (with few_shot_prefix prepended if absent), then make
    sure the options the item carries are actually visible to the model."""
    if isinstance(item.get("prompt"), str) and item["prompt"]:
        prefix = item.get("few_shot_prefix")
        if isinstance(prefix, str) and prefix and not item["prompt"].startswith(prefix):
            return inline_choices(prefix + item["prompt"], item)
        return inline_choices(item["prompt"], item)
    parts = [str(item.get("few_shot_prefix") or ""), str(item.get("context") or ""),
             str(item.get("question") or ""), str(item.get("prompt") or "")]
    return inline_choices("\n".join(p for p in parts if p), item)


def get_ground_truth(item: dict) -> str:
    return str(item.get("ground_truth", item.get("answer", item.get("target", ""))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="local HF model directory")
    p.add_argument("--task_data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", default="mmlu,obqa,race,squad,story_cloze,hellaswag,lambada,siqa")
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--use_chat_template", action="store_true",
                   help="wrap the prompt in the model's chat template (default: raw continuation)")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--tokenizer_dir", default=None,
                   help="load the tokenizer from here instead of --model_dir (e.g. a local\n                        GPT-2 copy for MDLM/SEDD/ELF, whose own tokenizer needs tiktoken)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restore_bos", action="store_true",
                   help="prepend the tokenizer's bos_token_id. The default tokenizer call uses\n"
                        "add_special_tokens=False, which for Gemma/Llama strips the leading BOS\n"
                        "those models expect. Off by default so that every previously published\n"
                        "row stays reproducible from this script unchanged.")
    return p.parse_args()


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, args, device) -> tuple[str, int, int]:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids.to(device)
    if getattr(args, "restore_bos", False):
        bos = getattr(tokenizer, "bos_token_id", None)
        # Only prepend when it is not already there: re-adding it would be a
        # different intervention from the one this flag names.
        if bos is not None and (ids.shape[1] == 0 or int(ids[0, 0]) != bos):
            ids = torch.cat([torch.tensor([[bos]], device=ids.device, dtype=ids.dtype), ids], dim=1)
    prompt_tokens = int(ids.shape[1])
    if args.temperature <= 0:
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    else:
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_ids = out[0, prompt_tokens:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    for special in getattr(tokenizer, "all_special_tokens", ()):
        text = text.replace(special, " ")
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text, prompt_tokens, len(gen_ids)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    print(f"[load] {args.model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir or args.model_dir, trust_remote_code=True, local_files_only=True
    )
    # DCLM (TRI-ML/DCLM-1B, apple/DCLM-7B) ships `model_type: "openlm"` with
    # `architectures: ["OpenLMModel"]`, NO auto_map and no modeling file, so neither
    # transformers nor trust_remote_code can resolve it: from_pretrained raises
    # ValueError("... model type `openlm` but Transformers does not recognize this
    # architecture"). mlfoundations' `open_lm` supplies the classes and self-registers them
    # with the Auto classes on `import open_lm.hf`. Registering HERE, before the CONFIG_MAPPING
    # probe below, is deliberate: it makes `openlm` a known model_type, so that probe takes the
    # "known" branch and does NOT import fla.models -- fla knows nothing about openlm and its
    # blanket re-registration of `mamba` is a documented hazard in this repo.
    # Gated on model_type so every other checkpoint's path is untouched.
    try:
        if json.load(open(os.path.join(str(args.model_dir),
                                       "config.json"))).get("model_type") == "openlm":
            import open_lm.hf  # noqa: F401  registers OpenLMConfig / OpenLMForCausalLM
            print("[load] model_type='openlm': imported open_lm.hf to register "
                  "OpenLMForCausalLM with the Auto classes", flush=True)
    except FileNotFoundError:
        pass
    try:
        # fla-hub checkpoints declare architectures (e.g. model_type `gla`) that only resolve
        # once the fla package registers them; the snapshots ship no remote-code file.
        # Do NOT import it unconditionally: `fla.models` ALSO re-registers the standard
        # `mamba` model_type, so it hijacks mamba-2.8b into fla's own Mamba, whose cached
        # decoding aborts with "Mamba cached decoding only supports a single new token per
        # step." (fla/layers/mamba.py slow_forward). transformers already ships a native
        # `mamba`, so only reach for fla when the checkpoint's model_type is unknown to it.
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        _mtype = None
        try:
            _mtype = json.load(open(os.path.join(str(args.model_dir),
                                                 "config.json"))).get("model_type")
        except Exception:
            pass
        if not (_mtype and _mtype in CONFIG_MAPPING):
            import fla.models  # noqa: F401
    except Exception:
        pass
    model = _load_hf_model(args.model_dir, device, trust_remote_code=args.trust_remote_code)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", "") or tokenizer.unk_token

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {"model_dir": args.model_dir, "max_samples": args.max_samples,
               "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
               "repetition_penalty": args.repetition_penalty,
               "restore_bos": bool(args.restore_bos),
               "protocol": "generate-then-match", "use_chat_template": args.use_chat_template,
               "tasks": {}}

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        path = Path(args.task_data_dir) / f"{task}.jsonl"
        if not path.exists():
            print(f"[{task}] SKIP (no {path})", flush=True)
            continue
        out_path = out_root / f"{task}.jsonl"
        t0 = time.time()
        n = 0
        with path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as fo:
            for i, line in enumerate(f):
                if i >= args.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                item: dict[str, Any] = json.loads(line)
                prompt = build_prompt(task, item)
                if args.use_chat_template and getattr(tokenizer, "chat_template", None):
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                    )
                gen, ptok, gtok = generate_one(model, tokenizer, prompt, args, device)
                rec = dict(item)
                rec.update({"id": item.get("id", i), "prompt": prompt, "generate": gen,
                            "ground_truth": get_ground_truth(item),
                            "choices": item.get("choices", []),
                            "prompt_tokens": ptok, "generated_tokens": gtok,
                            "method": "raw", "cfg_scale": 0.0, "steps": 0,
                            "baseline_model": args.model_dir})
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 20 == 0:
                    print(f"[{task}] {n} samples ({time.time()-t0:.0f}s)", flush=True)
        summary["tasks"][task] = {"n": n, "elapsed_s": round(time.time() - t0, 1)}
        print(f"[{task}] done n={n} in {time.time()-t0:.0f}s -> {out_path}", flush=True)

    (out_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

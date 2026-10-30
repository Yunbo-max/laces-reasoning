#!/usr/bin/env python3
"""STAR-LDM (Stop-Think-AutoRegress) generate-then-match harness.

WHY THIS FILE EXISTS
--------------------
STAR-LDM is *not* an HF causal LM (so `eval_generic_gen_match.py` cannot drive it)
and it is *not* a discrete/flow token denoiser either (so the MDLM/SEDD/Cola-DLM
harnesses do not apply).  It is GPT-2 Large + a **continuous latent-diffusion
planner** over Sentence-T5-XL embeddings (768-d):

    prompt tokens --GPT-2 wte--> embeddings
                                        \
    z_T ~ N(0, I) --(T diffusion steps)--> z_0 in Sentence-T5 space
         each step: z_t -> SoftPromptGenerator -> 8 soft tokens
                    -> concat with prompt embeddings -> GPT-2 (frozen context)
                    -> last-hidden at the 8 soft positions -> ScoreNetHead -> v
         final: z_0 -> SoftPromptGenerator -> 8 soft tokens
                -> GPT-2.generate(inputs_embeds=...) -> text

The generation entry point is `scripts/generate.py`, which is a thin CLI over
`star_ldm.interface.TransfusionGPTInterface`.  This adapter deliberately **reuses
both** (plus the exact `sample_kwargs` assembly from `scripts/generate.py`) rather
than reimplementing the diffusion loop, exactly as the sibling harnesses reuse
official samplers.

PROMPT PROTOCOL
---------------
`build_prompt` is imported from `eval_generic_gen_match` on purpose: the six tasks
(arc_easy/arc_challenge/winogrande/sciq/commonsenseqa/piqa) whose `prompt` field
omits the answer options must be inlined identically to every other baseline, or
the column is not comparable.  Do not fork that function.

OFFLINE REQUIREMENTS (compute nodes have no network)
----------------------------------------------------
`TransfusionGPTInterface.__init__` calls, with no way to override:

    AutoTokenizer.from_pretrained(self.model.gpt2.config._name_or_path)   # args.yaml -> train.lm_name

so the *hub id* named in `args.yaml` (here `gpt2-large`) must resolve from
`$HF_HOME/hub`.  For this checkpoint that entry is materialised at

    $HF_HOME/hub/models--gpt2-large/{refs/main, snapshots/main/* -> /…/models/gpt2-large/*}

Instead of editing the official code, `ensure_local_hub_entries()` below creates
that cache entry (symlinks to an on-disk copy) if it is missing, and also points
the hard-coded `sentence-transformers/sentence-t5-xl` sentence encoder at a local
snapshot.  Nothing in `models/STAR-LDM-github/` is modified.

OUTPUT SCHEMA
-------------
One `<output_dir>/<task>.jsonl` per task; every line is
`{**item, id, prompt, generate, ground_truth, choices, prompt_tokens,
  generated_tokens, method, cfg_scale, steps, baseline_model}`
-- field-for-field identical to `eval_generic_gen_match.py` so
`scripts/eval/score_gen_match.py` consumes it unchanged.  Here
`method="diffusion_star_ldm"`, `cfg_scale=<cls_free_guidance>`,
`steps=<sampling_timesteps>`.

Usage:
  python scripts/eval/eval_starldm_gen_match.py \
      --model_dir /inspire/…/models/STAR-LDM-unpacked \
      --task_data_dir /inspire/…/DiffRwkv/baseline/Cola-DLM/generate_task_data \
      --output_dir results/benchmarks/baseline-star-ldm \
      --tasks mmlu,obqa --max_samples 200 --max_new_tokens 32

  # smoke: generate for N prompts and print the raw text, then exit
  python scripts/eval/eval_starldm_gen_match.py --model_dir … --smoke 2 --tasks mmlu
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

# The official STAR-LDM repo (code, not weights).  Added to sys.path so that
# `import star_ldm…` / `import scripts.generate…` work from any cwd.
DEFAULT_STARLDM_CODE = "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/STAR-LDM-github"

# `from scripts.eval.eval_generic_gen_match import build_prompt` needs THIS repo's
# root on sys.path -- but `python scripts/eval/x.py` only puts `scripts/eval/` there
# (that is how the first QZ smoke run died with "No module named 'scripts.eval'").
# Derive the root from __file__ so the harness never depends on cwd or PYTHONPATH.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

TASKS_DEFAULT = ("mmlu,obqa,siqa,story_cloze,hellaswag,piqa,arc_easy,arc_challenge,"
                 "winogrande,sciq,commonsenseqa,lambada,squad,race")


# ---------------------------------------------------------------------------
# prompt construction -- shared with every other baseline harness
# ---------------------------------------------------------------------------
_GENERIC_HARNESS = Path(_REPO_ROOT) / "scripts" / "eval" / "eval_generic_gen_match.py"
_generic_mod = None


def _generic():
    """Load the shared `eval_generic_gen_match` helpers **by file path**.

    A plain `from scripts.eval.eval_generic_gen_match import build_prompt` is NOT
    safe in this harness, and the first two QZ smoke runs died on exactly that with
    `ModuleNotFoundError: No module named 'scripts.eval'`: the official STAR-LDM
    repo ships its *own* `scripts/` as a regular package (it has an `__init__.py`)
    and `_import_official()` puts that repo root on `sys.path`, so `import scripts`
    binds to STAR-LDM's package -- which has no `eval` subpackage -- instead of
    ours.  Loading by path is immune to that collision and to sys.path order.
    """
    global _generic_mod
    if _generic_mod is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_shared_eval_generic_gen_match", _GENERIC_HARNESS)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _generic_mod = mod
    return _generic_mod


def build_prompt(task: str, item: dict) -> str:
    """The shared implementation (see module docstring) -- exactly one definition."""
    return _generic().build_prompt(task, item)


def get_ground_truth(item: dict) -> str:
    return _generic().get_ground_truth(item)


# ---------------------------------------------------------------------------
# offline plumbing
# ---------------------------------------------------------------------------
def ensure_local_hub_entries(model_dir: str, hf_home: str) -> None:
    """Materialise the offline hub entries that the official loader resolves by id.

    `interface.py` hard-codes `sentence-transformers/sentence-t5-xl` for the
    sentence encoder and reads the LM id from `args.yaml`; both are looked up as
    *hub ids*, so on a network-less node they must exist under `$HF_HOME/hub`.
    This only creates cache entries -- no official file is touched.
    """
    hub = Path(hf_home) / "hub"
    hub.mkdir(parents=True, exist_ok=True)

    candidates = {
        "models--gpt2-large": _find_local_model("gpt2-large"),
        "models--gpt2-medium": _find_local_model("gpt2-medium"),
    }
    for repo_dir, src in candidates.items():
        if src is None:
            continue
        dst = hub / repo_dir
        if (dst / "snapshots" / "main").is_dir():
            continue
        snap = dst / "snapshots" / "main"
        (dst / "blobs").mkdir(parents=True, exist_ok=True)
        (dst / "refs").mkdir(parents=True, exist_ok=True)
        snap.mkdir(parents=True, exist_ok=True)
        (dst / "refs" / "main").write_text("main")
        for fn in os.listdir(src):
            p = os.path.join(src, fn)
            if not os.path.isfile(p) or fn.endswith(".onnx"):
                continue
            tgt = snap / fn
            if not tgt.exists():
                os.symlink(p, tgt)
        print(f"[hub] linked {repo_dir} -> {src}", flush=True)


def _find_local_model(name: str) -> Optional[str]:
    roots = ["/inspire/hdd/global_user/zhangjiaquan-253108540222/models"]
    here = Path(__file__).resolve()
    # …/research/<repo>/scripts/eval/… -> …/research/<repo>/models
    roots.append(str(here.parents[3] / "models"))
    for r in roots:
        p = Path(r) / name
        if (p / "config.json").is_file():
            return str(p)
    return None


def _import_official(code_dir: str):
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from star_ldm.interface import TransfusionGPTInterface  # noqa: E402
    return TransfusionGPTInterface


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def build_sample_kwargs(args, interface) -> dict:
    """Exactly the kwargs `scripts/generate.py` assembles (kept in sync by hand)."""
    return dict(
        sampling_timesteps=args.sampling_timesteps,
        sampler=args.sampler,
        cls_free_guidance=args.cls_free_guidance,
        cls_guidance=args.cls_guidance,
        cls_target=args.cls_target,
        generate_kwargs=dict(
            do_sample=True,
            num_beams=1,
            pad_token_id=interface.tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ),
    )


def _normalise(text: str) -> str:
    """Same whitespace/special-token cleanup as eval_generic_gen_match.generate_one."""
    text = re.sub(r"[ \t]{2,}", " ", text or "")
    return text.strip()


def generate_one(interface, tokenizer, prompt: str, sample_kwargs: dict) -> tuple[str, int, int]:
    """Returns (continuation_text, prompt_tokens, generated_tokens).

    `TransfusionGPTInterface.generate` returns the decoded continuation only (the
    soft prompt is concatenated to the *prompt embeddings* and GPT-2 is called with
    `inputs_embeds`, so the returned ids never include the prompt).  That matches
    the `generate` field of the other baselines.
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_tokens = int(ids.shape[1])
    text = interface.generate([prompt], **sample_kwargs)[0]
    text = _normalise(text)
    # The official path decodes ids without exposing them; approximate the count the
    # same way the scorer-relevant siblings do (re-encode the continuation).
    generated_tokens = len(tokenizer(text, add_special_tokens=False).input_ids) if text else 0
    return text, prompt_tokens, generated_tokens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="STAR-LDM checkpoint directory (must contain model.pt + args.yaml)")
    p.add_argument("--code_dir", default=DEFAULT_STARLDM_CODE,
                   help="the official STAR-LDM repo (star_ldm package)")
    p.add_argument("--task_data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", default=TASKS_DEFAULT)
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=32,
                   help="GPT-2 continuation length (official generate.py default: 64)")
    # --- official diffusion/sampling knobs (names follow scripts/generate.py) ---
    p.add_argument("--sampling_timesteps", type=int, default=50)
    p.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--cls_free_guidance", type=float, default=1.0,
                   help="1.0 = conditional only (official default)")
    p.add_argument("--cls_guidance", type=float, default=0.0,
                   help="classifier guidance scale (needs --classifier_path; 0 disables)")
    p.add_argument("--cls_target", type=float, default=None)
    p.add_argument("--classifier_path", default=None)
    # --- GPT-2 token sampling ---
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    # --- misc ---
    p.add_argument("--hf_home", default=os.environ.get(
        "HF_HOME", "/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", type=int, default=0,
                   help="generate for N prompts and print, then stop")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ensure_local_hub_entries(args.model_dir, os.environ["HF_HOME"])

    TransfusionGPTInterface = _import_official(args.code_dir)

    print(f"[load] {args.model_dir}", flush=True)
    interface = TransfusionGPTInterface(
        model_path=args.model_dir,
        device=args.device,
        classifier_path=args.classifier_path,
    )
    tokenizer = interface.tokenizer
    print(f"[load] ok. device={interface.device} eos_id={tokenizer.eos_token_id}", flush=True)

    sample_kwargs = build_sample_kwargs(args, interface)
    print("[sampling] " + json.dumps({k: v for k, v in sample_kwargs.items()
                                      if k != "generate_kwargs"}), flush=True)
    print("[generate_kwargs] " + json.dumps(sample_kwargs["generate_kwargs"]), flush=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_root = Path(args.output_dir)

    # ---------------- smoke ----------------
    if args.smoke > 0:
        for task in tasks:
            path = Path(args.task_data_dir) / f"{task}.jsonl"
            if not path.exists():
                print(f"[{task}] SKIP (no {path})", flush=True)
                continue
            print(f"\n===== SMOKE {task} =====", flush=True)
            with path.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= args.smoke:
                        break
                    item = json.loads(line)
                    prompt = build_prompt(task, item)
                    t0 = time.time()
                    gen, ptok, gtok = generate_one(interface, tokenizer, prompt, sample_kwargs)
                    dt = time.time() - t0
                    print(f"--- sample {i} ({dt:.1f}s, prompt_tokens={ptok}, gen_tokens={gtok}) ---",
                          flush=True)
                    print(f"PROMPT[:400]: {prompt[:400]!r}", flush=True)
                    print(f"GENERATE: {gen!r}", flush=True)
                    print(f"GROUND_TRUTH: {get_ground_truth(item)!r}", flush=True)
            break
        return

    # ---------------- full run ----------------
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {"model_dir": args.model_dir, "max_samples": args.max_samples,
               "max_new_tokens": args.max_new_tokens,
               "sampling_timesteps": args.sampling_timesteps, "sampler": args.sampler,
               "cls_free_guidance": args.cls_free_guidance, "top_p": args.top_p,
               "repetition_penalty": args.repetition_penalty,
               "protocol": "generate-then-match", "tasks": {}}

    for task in tasks:
        path = Path(args.task_data_dir) / f"{task}.jsonl"
        if not path.exists():
            print(f"[{task}] SKIP (no {path})", flush=True)
            continue
        out_path = out_root / f"{task}.jsonl"
        t0 = time.time()
        n = 0
        with path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as fo:
            for i, line in enumerate(f):
                if args.max_samples >= 0 and i >= args.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                item: dict[str, Any] = json.loads(line)
                prompt = build_prompt(task, item)
                gen, ptok, gtok = generate_one(interface, tokenizer, prompt, sample_kwargs)
                rec = dict(item)
                rec.update({"id": item.get("id", i), "prompt": prompt, "generate": gen,
                            "ground_truth": get_ground_truth(item),
                            "choices": item.get("choices", []),
                            "prompt_tokens": ptok, "generated_tokens": gtok,
                            "method": "diffusion_star_ldm",
                            "cfg_scale": args.cls_free_guidance,
                            "steps": args.sampling_timesteps,
                            "baseline_model": args.model_dir})
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fo.flush()
                n += 1
                if n % 20 == 0:
                    print(f"[{task}] {n} samples ({time.time()-t0:.0f}s)", flush=True)
        summary["tasks"][task] = {"n": n, "elapsed_s": round(time.time() - t0, 1)}
        print(f"[{task}] done n={n} in {time.time()-t0:.0f}s -> {out_path}", flush=True)

    (out_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

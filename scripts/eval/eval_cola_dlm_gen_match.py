#!/usr/bin/env python3
"""Generate-then-match harness for **CoLA-DLM** (Continuous Latent Diffusion LM).

CoLA-DLM (ByteDance-Seed, arXiv:2605.06548) is *not* a HuggingFace
``AutoModelForCausalLM``: it ships two bespoke ``PreTrainedModel`` subclasses
(``ColaDiTModel`` = block-causal latent Diffusion Transformer prior,
``ColaTextVAEModel`` = Text VAE with a 16-d continuous latent) and its own
sampler in ``cola_dlm.inference.generate_task_repaint_inference``.  Feeding it to
``eval_generic_gen_match.py`` fails at load time with
``Unrecognized model ... no model_type`` (there is no ``auto_map`` either), so
this adapter is the CoLA-DLM entry point of the same baseline table.

It follows ``eval_diffusion_lm_gen_match.py`` exactly where that is possible:

  prompts   ``build_prompt`` is copied verbatim from ``eval_generic_gen_match.py``
            (``few_shot_prefix + prompt``), so CoLA-DLM sees the *same* prompt
            string as every other row of the table.  The string is handed to
            CoLA's official sampler as the ``question`` field with
            ``task_name="raw"``, whose template branch is ``return question``
            (verified in ``inference.py:217``).  CoLA's own per-task templates are
            *not* used: our ``generate_task_data`` items carry the pre-built
            ``prompt`` (few-shot prefix already inlined) and no ``question`` /
            ``context`` fields, so the official templates would emit
            ``Question: \n(A)...`` with an empty question.  For the 8 benchmark
            tasks the pre-built prompt is byte-identical to what the official
            template emits (the jsonl was split out of it), so this is not a
            deviation in prompt content.
  decoding  the official sampler loop is called unchanged (prefix encode ->
            block-wise flow-matching transport with CFG -> VAE decode).  We only
            *observe* it: a tokenizer proxy records the exact number of generated
            tokens and decodes without special tokens, and the text then gets the
            same whitespace cleanup as ``eval_generic_gen_match.py``.
  output    one ``<output_dir>/<task>.jsonl`` per task with the identical record
            schema (``{**item, id, prompt, generate, ground_truth, choices,
            prompt_tokens, generated_tokens, method, cfg_scale, steps,
            baseline_model}``), so ``score_gen_match.py`` applies unchanged.

Offline: every load is ``local_files_only=True`` (and the QZ command sets
``HF_HUB_OFFLINE=1``); nothing is fetched from the Hub.

Requires a CUDA device.  The official sampler hard-codes
``autocast("cuda", bfloat16)`` and explicit ``.to(torch.bfloat16)`` casts, which
torch silently *disables* on a CPU-only host — the bf16 DiT input then meets fp32
weights and dies with ``mat1 and mat2 must have the same dtype``.  That is the
official code's property, not this adapter's; do not bother with ``--device cpu``.

``--model_dir`` note: ``/models/Cola-DLM`` (capitalised) is a *truncated* copy of
the release — DiT shard 2 is short by 460,405,120 bytes and the VAE by 24,840,857 —
so it fails inside ``from_pretrained`` with ``SafetensorError ...
MetadataIncompleteBuffer``.  Point at ``/models/cola-dlm`` (the complete copy); the
``check_weights`` preflight below rejects the truncated one in two seconds.

Sanity check for a fresh GPU job: ``--smoke 2`` generates for the first 2 prompts
only and prints prompt / ground truth / generation to the log.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any

import torch

TASKS_DEFAULT = "mmlu,obqa,race,story_cloze,hellaswag,siqa,lambada,squad"

# Candidate locations of the `cola_dlm` python package (the *code*, not the
# weights).  First one that imports wins; override with --code_dir.
COLA_CODE_CANDIDATES = [
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/Cola-DLM-github",
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/baseline/Cola-DLM",
]

# Token ids of the released tokenizer (models/cola-dlm/tokenizer.json, vocab 100278).
PAD_TOKEN_ID = 100277
EOS_TOKEN_ID = 100257
IM_END_TOKEN_ID = 100265


# ---------------------------------------------------------------------------
# Prompt / answer helpers — copied verbatim from eval_generic_gen_match.py so the
# prompts and the ground-truth field stay identical across the whole table.
# ---------------------------------------------------------------------------


_OPTION_MARKER_RE = re.compile(r"\(A\)")


def inline_choices(prompt: str, item: dict) -> str:
    """Inline the item's ``choices`` when the prompt does not already carry them.

    Same defect and same guard as eval_generic_gen_match.inline_choices: the six
    tasks arc_easy/arc_challenge/winogrande/sciq/commonsenseqa/piqa ship ``choices``
    but no option list in ``prompt``. mmlu/obqa/siqa/story_cloze/hellaswag/race
    already carry a ``(A) ... / (B) ...`` block and are returned untouched, so this
    is idempotent.
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
    """Mirror of eval_generic_gen_match.build_prompt: prefer the pre-built 'prompt'
    field (with few_shot_prefix prepended if absent), then inline the choices when
    the prompt omits them."""
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


# ---------------------------------------------------------------------------
# Model code / weight discovery and validation
# ---------------------------------------------------------------------------


def resolve_code_dir(model_dir: str, override: str | None) -> str:
    """Return a directory that contains the importable `cola_dlm` package."""
    env = os.environ.get("COLA_DLM_CODE_DIR") or None
    cands = [override, env, model_dir, *COLA_CODE_CANDIDATES]
    for c in cands:
        if not c:
            continue
        p = Path(c)
        if (p / "cola_dlm" / "inference.py").exists() and (p / "cola_dlm" / "modeling_cola_dit.py").exists():
            return str(p)
    raise SystemExit(
        "cannot find the CoLA-DLM python package (needs cola_dlm/{inference,modeling_cola_dit}.py). "
        f"Tried: {[c for c in cands if c]}. Pass --code_dir explicitly."
    )


def check_safetensors_file(path: Path) -> str | None:
    """Return a human-readable problem string if a .safetensors file is truncated."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            return f"{path} is only {size} bytes (not a safetensors file)"
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or 8 + header_len > size or header_len > 100_000_000:
            return (f"{path}: header length {header_len} does not fit in {size:,} bytes "
                    f"(git-lfs pointer or truncated file?)")
        hdr = fh.read(header_len)
    try:
        header = json.loads(hdr.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return f"{path}: header not parseable ({type(e).__name__})"
    end = max((h["data_offsets"][1] for k, h in header.items() if k != "__metadata__"), default=0)
    need = 8 + header_len + end
    if size < need:
        return (f"{path} is TRUNCATED: {size:,} bytes on disk, {need:,} required "
                f"(short by {need - size:,})")
    return None


def check_weights(dit_dir: Path, vae_dir: Path) -> None:
    """Fail fast with a precise message instead of dying 20 minutes into a GPU job.

    Learned the hard way: /models/Cola-DLM is a truncated copy of the release
    (DiT shard 2 short by 460,405,120 bytes, VAE short by 24,840,857 bytes), whose
    only symptom is ``SafetensorError: Error while deserializing header:
    MetadataIncompleteBuffer`` from deep inside ``from_pretrained``.
    """
    problems: list[str] = []
    for d, name in ((dit_dir, "DiT"), (vae_dir, "VAE")):
        if not d.is_dir():
            problems.append(f"{name} directory missing: {d}")
            continue
        if not (d / "config.json").exists():
            problems.append(f"{name} config.json missing in {d}")
        shards = sorted(d.glob("*.safetensors"))
        if not shards:
            problems.append(f"{name} has no *.safetensors in {d}")
        for s in shards:
            msg = check_safetensors_file(s)
            if msg:
                problems.append(msg)
    if problems:
        raise SystemExit(
            "[cola-dlm] unusable checkpoint:\n  - " + "\n  - ".join(problems)
            + "\n(known-good complete copy: "
              "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/cola-dlm)"
        )


# ---------------------------------------------------------------------------
# Tokenizer proxy: exact generated-token counts + special-token-free text
# ---------------------------------------------------------------------------


class TokenizerProxy:
    """Thin wrapper around ``tokenizers.Tokenizer`` for the official sampler.

    ``generate_task_repaint_inference`` uses exactly two tokenizer calls
    (``encode(prompt).ids`` and ``decode_batch(ids, skip_special_tokens=False)``).
    Wrapping them lets us record the *exact* generated length per sample and drop
    the ``<|pad|>`` / ``<|endoftext|>`` / ``<|im_end|>`` markers, which is what
    ``eval_generic_gen_match.py`` passes to the scorer as ``generate``.
    """

    def __init__(self, tok):
        self._tok = tok
        self.gen_lengths: list[int] = []

    def encode(self, text: str):
        return self._tok.encode(text)

    def decode_batch(self, batch, skip_special_tokens: bool = False):
        self.gen_lengths = [len(b) for b in batch]
        return self._tok.decode_batch(batch, skip_special_tokens=True)

    def __getattr__(self, name):
        return getattr(self._tok, name)


def clean_generation(text: str) -> str:
    """Same post-processing as eval_generic_gen_match.generate_one."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def flush_kv(dit, vae) -> None:
    """Drop per-sample K/V caches (the official sampler does this at the end of a
    successful call; an exception mid-loop would otherwise leave stale cache)."""
    try:
        for block in dit.blocks:
            block.set_kv_cache(False)
        vae.set_kv_cache(False)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Preflight
#
# This adapter cannot execute a single diffusion step on a GPU-less host (the
# official loop hard-codes ``autocast("cuda", bfloat16)``, which torch *disables*
# on a CPU-only machine, so the explicitly-bf16 DiT input meets fp32 weights and
# dies with "mat1 and mat2 must have the same dtype").  Everything the wrapper
# assumes about the released cola_dlm API is therefore asserted up front, before
# the 9 GB of weights are read, so a signature change fails in two seconds instead
# of 20 minutes into a GPU job.
# ---------------------------------------------------------------------------

SAMPLER_KWARGS = (
    "dit", "vae", "tokenizer", "prompts", "task_name", "device", "T", "timestep_num",
    "guidance_scale", "max_new_tokens", "temperature", "top_k", "top_p",
    "repetition_penalty", "pad_token_id", "eos_token_id", "im_end_token_id",
)


def preflight(code_dir: str) -> None:
    from cola_dlm import apply_prompt_template, generate_task_repaint_inference

    params = inspect.signature(generate_task_repaint_inference).parameters
    unknown = [k for k in SAMPLER_KWARGS if k not in params]
    if unknown:
        raise SystemExit(
            f"[cola-dlm] {code_dir}: generate_task_repaint_inference has no parameter(s) "
            f"{unknown}; the released API changed and the adapter must be updated "
            f"(signature: {list(params)})"
        )
    # task_name="raw" is not a CoLA task, so apply_prompt_template falls through to
    # `return question` -- the prompt the model sees is then exactly build_prompt's
    # output.  Asserted, not assumed.
    sentinel = "SENTINEL_PROMPT_42"
    if apply_prompt_template("raw", "", sentinel, "", None) != sentinel:
        raise SystemExit(
            "[cola-dlm] apply_prompt_template('raw', ...) no longer returns the question "
            "verbatim; the shared prompt protocol would silently change"
        )
    print(f"[preflight] sampler signature {len(params)} params + raw-prompt branch OK ({code_dir})", flush=True)


def check_model_surface(dit, vae) -> None:
    """The attributes generate_task_repaint_inference touches on the two models."""
    problems = []
    for obj, name, attrs in (
        (dit, "dit", ("block_size", "blocks", "config")),
        (vae, "vae", ("scaling_factor", "shifting_factor", "patch_size", "latent_dim",
                      "set_kv_cache", "encode", "decode")),
    ):
        for a in attrs:
            if not hasattr(obj, a):
                problems.append(f"{name}.{a}")
    try:
        if len(dit.blocks) and not hasattr(dit.blocks[0], "set_kv_cache"):
            problems.append("dit.blocks[0].set_kv_cache")
    except Exception:  # noqa: BLE001
        pass
    if problems:
        raise SystemExit(f"[cola-dlm] model objects are missing {problems}; the adapter targets "
                         f"the released cola_dlm API")
    print(f"[preflight] model surface OK (dit.blocks={len(dit.blocks)}, vae latent_dim="
          f"{vae.latent_dim})", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True,
                   help="CoLA-DLM root directory: contains cola_dlm/cola_dit, cola_dlm/cola_vae, tokenizer.json")
    p.add_argument("--code_dir", default=None,
                   help="directory containing the importable cola_dlm package (auto-detected if omitted)")
    p.add_argument("--dit_path", default=None, help="override the DiT weight dir (default <model_dir>/cola_dlm/cola_dit)")
    p.add_argument("--vae_path", default=None, help="override the VAE weight dir (default <model_dir>/cola_dlm/cola_vae)")
    p.add_argument("--tokenizer_path", default=None, help="override tokenizer.json (default <model_dir>/tokenizer.json)")
    p.add_argument("--task_data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", default=TASKS_DEFAULT)
    p.add_argument("--max_samples", type=int, default=100, help="0 or negative = all samples (official CLI convention)")
    p.add_argument("--max_new_tokens", type=int, default=32,
                   help="CoLA-DLM generates in whole blocks of 16 tokens, so this is rounded up "
                        "to a multiple of 16 by its own sampler")
    p.add_argument("--steps", "--timestep_num", dest="steps", type=int, default=16,
                   help="flow-matching denoising steps (official default: 16)")
    p.add_argument("--guidance_scale", type=float, default=7.0, help="CFG scale (official default: 7.0)")
    p.add_argument("--batch_size", type=int, default=20, help="samples per sampler call (official default: 20)")
    p.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--pad_token_id", type=int, default=PAD_TOKEN_ID)
    p.add_argument("--eos_token_id", type=int, default=EOS_TOKEN_ID)
    p.add_argument("--im_end_token_id", type=int, default=IM_END_TOKEN_ID)
    p.add_argument("--noise_seed", type=int, default=66,
                   help="COLA_INFER_PER_SAMPLE_NOISE_SEED: per-sample noise keyed by (seed, id, step) so "
                        "results do not depend on batch composition (official run_benchmark.sh uses 66); "
                        "-1 = use the global RNG")
    p.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16", "float16"],
                   help="weight dtype; default = the dtype declared in the checkpoint config (fp32)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", type=int, default=0,
                   help="generate for the first N prompts only and print prompt/GT/generation, then stop")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sampler wrapper
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_cola_sampler(dit, vae, tok_proxy, prompts: list[dict], args, device) -> list[dict]:
    """One call of the official CoLA-DLM sampler on a batch of `prompts`.

    ``task_name="raw"`` is not one of CoLA's task names, so
    ``apply_prompt_template`` returns the ``question`` field verbatim — i.e. the
    exact prompt built by ``build_prompt``.
    """
    from cola_dlm import generate_task_repaint_inference

    kwargs: dict[str, Any] = {
        "dit": dit,
        "vae": vae,
        "tokenizer": tok_proxy,
        "prompts": prompts,
        "task_name": "raw",
        "device": device,
        "T": 1000.0,
        "timestep_num": args.steps,
        "guidance_scale": args.guidance_scale,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": args.pad_token_id,
        "eos_token_id": args.eos_token_id,
        "im_end_token_id": args.im_end_token_id,
    }
    assert set(kwargs) == set(SAMPLER_KWARGS), "sampler kwargs out of sync with the preflight list"
    return generate_task_repaint_inference(**kwargs)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.noise_seed is not None and args.noise_seed >= 0:
        os.environ["COLA_INFER_PER_SAMPLE_NOISE_SEED"] = str(args.noise_seed)
    elif args.noise_seed is not None and args.noise_seed < 0:
        os.environ.pop("COLA_INFER_PER_SAMPLE_NOISE_SEED", None)

    model_dir = Path(args.model_dir)
    dit_dir = Path(args.dit_path) if args.dit_path else model_dir / "cola_dlm" / "cola_dit"
    vae_dir = Path(args.vae_path) if args.vae_path else model_dir / "cola_dlm" / "cola_vae"
    tok_path = Path(args.tokenizer_path) if args.tokenizer_path else model_dir / "tokenizer.json"

    # --- code + weights -----------------------------------------------------
    code_dir = resolve_code_dir(str(model_dir), args.code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    print(f"[load] cola_dlm code: {code_dir}", flush=True)

    check_weights(dit_dir, vae_dir)
    if not tok_path.exists():
        raise SystemExit(f"[cola-dlm] tokenizer.json not found: {tok_path}")

    from cola_dlm import ColaDiTModel, ColaTextVAEModel  # noqa: E402
    from tokenizers import Tokenizer  # noqa: E402

    preflight(code_dir)

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}.get(args.dtype) if args.dtype else None

    t0 = time.time()
    dit = ColaDiTModel.from_pretrained(str(dit_dir), local_files_only=True, torch_dtype=dtype)
    dit = dit.to(device).eval()
    print(f"[load] DiT  {dit_dir}  {sum(p.numel() for p in dit.parameters())/1e9:.3f}B "
          f"dtype={next(dit.parameters()).dtype} block_size={dit.block_size} ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    vae = ColaTextVAEModel.from_pretrained(str(vae_dir), local_files_only=True, torch_dtype=dtype)
    vae = vae.to(device).eval()
    print(f"[load] VAE  {vae_dir}  {sum(p.numel() for p in vae.parameters())/1e9:.3f}B "
          f"latent_dim={vae.latent_dim} patch_size={vae.patch_size} ({time.time()-t0:.0f}s)", flush=True)
    if vae.latent_dim != dit.config.txt_in_channels or vae.latent_dim != dit.config.txt_out_channels:
        raise SystemExit(f"[cola-dlm] latent mismatch: VAE latent_dim={vae.latent_dim} vs DiT "
                         f"txt_in={dit.config.txt_in_channels}/txt_out={dit.config.txt_out_channels}")
    check_model_surface(dit, vae)
    raw_tok = Tokenizer.from_file(str(tok_path))
    if not (hasattr(raw_tok, "encode") and hasattr(raw_tok, "decode_batch")):
        raise SystemExit("[cola-dlm] tokenizers.Tokenizer is missing encode/decode_batch")
    tok = TokenizerProxy(raw_tok)
    print(f"[load] tokenizer {tok_path} vocab={raw_tok.get_vocab_size()} device={device} "
          f"steps={args.steps} cfg={args.guidance_scale} bs={args.batch_size} "
          f"noise_seed={os.environ.get('COLA_INFER_PER_SAMPLE_NOISE_SEED')}", flush=True)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model_dir": str(model_dir), "code_dir": code_dir, "method": "diffusion_cola_dlm",
        "steps": args.steps, "guidance_scale": args.guidance_scale,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "batch_size": args.batch_size, "max_samples": args.max_samples,
        "prompt_protocol": "build_prompt (eval_generic_gen_match)", "tasks": {},
    }

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        path = Path(args.task_data_dir) / f"{task}.jsonl"
        if not path.exists():
            print(f"[{task}] SKIP (no {path})", flush=True)
            continue
        items: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
                if 0 < args.max_samples <= len(items):
                    break
        out_path = out_root / f"{task}.jsonl"
        t0, n = time.time(), 0
        with out_path.open("w", encoding="utf-8") as fo:
            for start in range(0, len(items), args.batch_size):
                chunk = items[start:start + args.batch_size]
                prompts, metas = [], []
                for j, item in enumerate(chunk):
                    idx = start + j
                    prompt = build_prompt(task, item)
                    prompts.append({"id": item.get("id", idx), "question": prompt})
                    metas.append((item, prompt, idx))

                try:
                    results = run_cola_sampler(dit, vae, tok, prompts, args, device)
                    gen_lengths = list(tok.gen_lengths)
                except Exception as e:  # noqa: BLE001
                    flush_kv(dit, vae)
                    print(f"[{task}] batch at {start} failed ({type(e).__name__}: {e}); "
                          f"falling back to batch_size=1", flush=True)
                    results, gen_lengths = [], []
                    for p_one in prompts:
                        try:
                            r_one = run_cola_sampler(dit, vae, tok, [p_one], args, device)
                            results.append(r_one[0])
                            gen_lengths.append(tok.gen_lengths[0] if tok.gen_lengths else 0)
                        except Exception as e1:  # noqa: BLE001
                            flush_kv(dit, vae)
                            print(f"[{task}] sample id={p_one['id']} failed: "
                                  f"{type(e1).__name__}: {e1}", flush=True)
                            results.append({"generate": ""})
                            gen_lengths.append(0)

                for k, (item, prompt, idx) in enumerate(metas):
                    raw_gen = str((results[k] if k < len(results) else {}).get("generate", ""))
                    gen = clean_generation(raw_gen)
                    ptok = len(raw_tok.encode(prompt).ids)
                    gtok = gen_lengths[k] if k < len(gen_lengths) else len(raw_tok.encode(gen).ids)
                    if args.smoke:
                        print(f"  [smoke {idx}] prompt_tokens={ptok} prompt_tail={prompt[-45:]!r} "
                              f"GT={get_ground_truth(item)[:60]!r} -> {gen[:110]!r}", flush=True)
                    rec = dict(item)
                    rec.update({
                        "id": item.get("id", idx),
                        "prompt": prompt,
                        "generate": gen,
                        "ground_truth": get_ground_truth(item),
                        "choices": item.get("choices", []),
                        "prompt_tokens": ptok,
                        "generated_tokens": gtok,
                        "method": "diffusion_cola_dlm",
                        "cfg_scale": args.guidance_scale,
                        "steps": args.steps,
                        "baseline_model": str(model_dir),
                    })
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                fo.flush()
                if args.smoke and n >= args.smoke:
                    break
                if n % 20 == 0 or n == len(items):
                    el = time.time() - t0
                    print(f"[{task}] {n}/{len(items)} samples ({el:.0f}s, {n/max(el,1e-9):.2f} it/s)", flush=True)
        summary["tasks"][task] = {"n": n, "elapsed_s": round(time.time() - t0, 1)}
        print(f"[{task}] done n={n} in {time.time()-t0:.0f}s -> {out_path}", flush=True)
        if args.smoke and n >= args.smoke:
            break

    (out_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

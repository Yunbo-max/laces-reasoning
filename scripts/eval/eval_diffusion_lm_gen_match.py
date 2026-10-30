#!/usr/bin/env python3
"""Generate-then-match harness for DISCRETE DIFFUSION language models.

Three implementations, sharing the prompts / record schema / scorer of
eval_generic_gen_match.py so diffusion-LM rows drop into the same table:

  --impl hf_diffusion   LLaDA / Dream style: HF custom code, model.generate()
  --impl mdlm           MDLM absorbing-state denoising (cosine schedule),
                        sampler re-implemented from Kuleshov et al. 2024
  --impl sedd           SEDD score-entropy denoising (same loop, different model)

A generation is produced for the *completion span only*: the prompt tokens are
fixed, the answer span starts fully masked and is denoised, which is the standard
masked-diffusion LM evaluation protocol.

--smoke N generates for the first N prompts only and prints prompt/completion,
so a job log is enough to verify that a sampler is wired correctly before the
full benchmark is trusted.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/baseline/Cola-DLM")
sys.path.insert(0, str(Path(__file__).resolve().parent))




def _patch_remote_models(model_dir: str):
    """LLaDA's remote code defines tie_weights(self) while this transformers
    release calls tie_weights(missing_keys=...); wrap the remote class so older
    signatures keep working. Also registers the Dream config for AutoModel."""
    import inspect
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except Exception:
        return
    for name in ("modeling_llada.LLaDAModelLM", "modeling_lladamoe.LLaDAMoEModel"):
        try:
            cls = get_class_from_dynamic_module(name, model_dir)
        except Exception:
            continue
        try:
            sig = inspect.signature(cls.tie_weights)
            if "missing_keys" not in sig.parameters:
                orig = cls.tie_weights

                import inspect as _insp
                _allowed = set(_insp.signature(orig).parameters)

                def _tie(self, *a, __orig=orig, __allowed=_allowed, **k):
                    k = {kk: vv for kk, vv in k.items() if kk in __allowed}
                    return __orig(self, *a, **k)
                cls.tie_weights = _tie
        except (ValueError, TypeError):
            pass


def _patch_rope_functions():
    """Remote Dream / LLaDA-MoE code indexes ROPE_INIT_FUNCTIONS['default'], a key that
    transformers 5.x no longer defines (it moved to the rope_parameters syntax). Add the
    classic default RoPE initialiser back under that name."""
    try:
        import torch
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            def _default_rope(config, device=None, seq_len=None, **kw):
                base = float(getattr(config, "rope_theta", 10000.0))
                dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
                return inv_freq, 1.0
            ROPE_INIT_FUNCTIONS["default"] = _default_rope
            print("[patch] ROPE_INIT_FUNCTIONS['default'] restored", flush=True)
    except Exception as e:
        print(f"[patch] rope patch skipped: {type(e).__name__}", flush=True)

def _patch_transformers():
    """LLaDA / Dream remote code predates the `all_tied_weights_keys` attribute
    that this transformers release expects during load; add it if missing."""
    import transformers.modeling_utils as mu
    if not hasattr(mu.PreTrainedModel, "all_tied_weights_keys"):
        mu.PreTrainedModel.all_tied_weights_keys = {}

_OPTION_MARKER_RE = re.compile(r"\(A\)")


def inline_choices(prompt: str, item: dict) -> str:
    """Inline the item's ``choices`` when the prompt does not already carry them.

    Same defect and same guard as eval_generic_gen_match.inline_choices: the six
    tasks arc_easy/arc_challenge/winogrande/sciq/commonsenseqa/piqa ship ``choices``
    but no option list in ``prompt``. Tasks that already carry a ``(A) ...`` block
    are returned byte-identical, so this is idempotent.
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
    if isinstance(item.get("prompt"), str) and item["prompt"]:
        prefix = item.get("few_shot_prefix")
        if isinstance(prefix, str) and prefix and not item["prompt"].startswith(prefix):
            return inline_choices(prefix + item["prompt"], item)
        return inline_choices(item["prompt"], item)
    return inline_choices("\n".join(p for p in [str(item.get("few_shot_prefix") or ""),
                                                str(item.get("context") or ""),
                                                str(item.get("question") or "")] if p), item)


def get_ground_truth(item: dict) -> str:
    return str(item.get("ground_truth", item.get("answer", item.get("target", ""))))


def cosine_alpha_bar(t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """MDLM's cosine noise schedule: alpha_bar(t) = cos((t+eps)/(1+eps) * pi/2)^2."""
    return torch.cos((t + eps) / (1.0 + eps) * (math.pi / 2.0)) ** 2


@torch.no_grad()
def mdlm_denoise(model, prompt_ids: torch.Tensor, gen_len: int, steps: int,
                 mask_id: int, device, temperature: float = 1.0) -> list[int]:
    """Absorbing-state ancestral sampler over the completion span only."""
    L = prompt_ids.shape[1]
    x = torch.cat([prompt_ids, torch.full((1, gen_len), mask_id, device=device, dtype=torch.long)], dim=1)
    for i in range(steps, 0, -1):
        t = torch.tensor([i / steps], device=device, dtype=torch.float32)
        s = torch.tensor([(i - 1) / steps], device=device, dtype=torch.float32)
        out = model(input_ids=x, timesteps=t)
        # mdlm-owt's config.json sets "return_dict": false, so MDLM.forward() returns a BARE
        # (1, L, vocab) tensor, not a MaskedLMOutput. The old `else out[0]` then indexed the
        # BATCH dim, yielding (L, vocab), and the next line's logits[:, L:, :] raised
        # IndexError. Take a bare tensor as-is; only unwrap real tuples.
        if hasattr(out, "logits"):
            logits = out.logits
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        if logits.dim() == 2:  # a model that squeezed the batch dim
            logits = logits.unsqueeze(0)
        span = logits[:, L:, :].float() / max(temperature, 1e-6)
        probs = torch.softmax(span, dim=-1)
        x0 = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(1, gen_len)
        ab_t = cosine_alpha_bar(t).to(device)
        ab_s = cosine_alpha_bar(s).to(device)
        if float(s.item()) > 0.0:
            # P(z_s = m | z_t = m) -- the probability a masked position is STILL masked at the
            # earlier time s. MDLM (arXiv 2406.07524) eq. (29) gives
            #   p_theta(z_s=m | z_t=m) = [a_s <x_theta,m> + (1-a_s)] / [a_t <x_theta,m> + (1-a_t)]
            # which, in the exact-reconstruction limit <x_theta,m> = 0 that this simplified
            # ancestral step uses, is (1-a_s)/(1-a_t).  [eq. (24)/(29), and the A.2.1 Case 2
            # derivation: q(z_s|z_t=m,x) = Cat([(1-a_s) m + (a_s-a_t) x] / (1-a_t)).]
            # The previous code used (a_s - a_t)/(1 - a_t) -- that is eq. (28), the probability
            # of the OPPOSITE event (becoming unmasked). It is the exact complement
            # (1-a_s)/(1-a_t) = 1 - (a_s-a_t)/(1-a_t), so the sampler was inverted: at t=1
            # (a_t=0) it unmasked the entire span in step 1 (p_mask = a_s ~ 6.7e-4) and then
            # progressively RE-masked as t->0, degenerating into a single one-shot prediction
            # from an all-masked span instead of iterative refinement. Measured consequence:
            # 94% of MMLU generations contained no answer choice and MMLU scored 1.5% (128
            # steps) / 3.0% (64 steps), i.e. far below the 25% 4-choice floor.
            p_mask = ((1.0 - ab_s) / (1.0 - ab_t).clamp(min=1e-8)).clamp(0.0, 1.0)
            still_masked = (torch.rand(1, gen_len, device=device) < p_mask).to(device)
            new_span = torch.where(still_masked, torch.full_like(x0, mask_id), x0)
        else:
            new_span = x0
        x = torch.cat([x[:, :L], new_span], dim=1)
    return x[0, L:].tolist()


@torch.no_grad()
def hf_diffusion_generate(model, tokenizer, prompt_ids: torch.Tensor, gen_len: int,
                          steps: int, device) -> list[int]:
    """LLaDA / Dream style masked-diffusion generate(): pass only the step keyword this
    checkpoint declares, and only in its own signature — transformers rejects unused
    kwargs with ValueError, and different releases name the argument differently."""
    import inspect
    try:
        accepted = set(inspect.signature(model.generate).parameters)
    except (TypeError, ValueError):
        accepted = set()
    step_kw = next((k for k in ("steps", "diffusion_steps", "num_steps", "num_denoise_steps")
                    if k in accepted), None)
    kw = dict(max_new_tokens=gen_len, do_sample=False)
    if step_kw:
        kw[step_kw] = steps
    for extra in ("block_length", "block_size"):
        if extra in accepted:
            pass
    out = model.generate(input_ids=prompt_ids, **kw)
    return out[0, prompt_ids.shape[1]:].tolist()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--tokenizer_dir", default=None, help="tokenizer dir override (MDLM/SEDD use GPT-2 BPE; tiktoken is unavailable)")
    p.add_argument("--impl", required=True, choices=["hf_diffusion", "mdlm", "sedd"])
    p.add_argument("--task_data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", default="mmlu,obqa,race,squad,story_cloze,hellaswag,lambada,siqa")
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--steps", type=int, default=64, help="denoising steps")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--smoke", type=int, default=0, help="generate for N prompts and print, then stop")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    _patch_transformers()
    _patch_rope_functions()
    _patch_remote_models(args.model_dir)

    device = torch.device(args.device)
    # MDLM's remote code calls `c = F.silu(self.sigma_map(sigma))` OUTSIDE its own
    # `with torch.cuda.amp.autocast(dtype=torch.bfloat16):` block (modeling_mdlm.py:393 vs
    # :396). `sigma` is built as float32 in the sampler, so a bf16-weight sigma_map hits
    # F.linear(float32, bfloat16) with no autocast to reconcile them:
    #   RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
    # This is device-independent -- it is not a CPU artefact. The published checkpoint is
    # float32 on disk, so load MDLM in its native dtype. Every other impl keeps bf16.
    load_dtype = torch.float32 if args.impl == "mdlm" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir or args.model_dir,
                                        trust_remote_code=True, local_files_only=True)
    def _explicit_classes():
        """Read config.json's auto_map and load the remote class directly."""
        import json as _json
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        out = []
        try:
            cfg = _json.load(open(Path(args.model_dir) / "config.json"))
        except Exception:
            return out
        am = cfg.get("auto_map", {}) or {}
        # kuleshov-group/mdlm-owt declares ONLY AutoConfig + AutoModelForMaskedLM in its
        # auto_map -- there is no AutoModel / AutoModelForCausalLM entry. With just those two
        # keys this function returned [] and the loader ended in
        # SystemExit("no loader could instantiate this checkpoint") for every task.
        # AutoModelForMaskedLM must come first because AutoModelForCausalLM rejects
        # MDLMConfig outright ("Unrecognized configuration class").
        for key in ("AutoModelForMaskedLM", "AutoModelForCausalLM", "AutoModel"):
            ref = am.get(key)
            if isinstance(ref, str):
                out.append(get_class_from_dynamic_module(ref, args.model_dir))
            elif isinstance(ref, (list, tuple)):
                for r in ref:
                    try:
                        out.append(get_class_from_dynamic_module(r, args.model_dir))
                    except Exception:
                        pass
        return out

    model = None
    _sedd_mask_id = None
    if args.impl == "sedd":
        # SEDD-medium is published as a RAW hydra training checkpoint: config.json has
        # graph/noise/model sections and no `model_type`, and pytorch_model.bin is a bare
        # 424M-param state_dict. No HF auto-class can build it, which is why every earlier
        # attempt ended in
        #   ValueError: Unrecognized model in .../SEDD-medium.
        #               Should have a `model_type` key in its config.json.
        # sedd_native rebuilds the published DDiT and strict-loads the state_dict, so skip
        # the AutoModel chain entirely.
        from sedd_native import load_sedd, mask_id_of, sedd_denoise
        model, _sedd_cfg = load_sedd(args.model_dir, device)
        _sedd_mask_id = mask_id_of(_sedd_cfg)
        loaders = []
    else:
        loaders = [AutoModel] if args.impl == "mdlm" else [AutoModelForCausalLM, AutoModel]
    for loader in loaders:
        try:
            model = loader.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code,
                                           torch_dtype=load_dtype, local_files_only=True)
            print(f"[load] via {loader.__name__}", flush=True)
            break
        except Exception as e:
            print(f"[load] {loader.__name__} failed: {type(e).__name__}: {str(e)[:110]}", flush=True)
    if model is None:
        # build the config from its own class first: Dream / LLaDA-MoE need a concrete
        # config object, otherwise the auto-config lookup raises KeyError: 'default'.
        cfg_obj = None
        try:
            import json as _json
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            raw = _json.load(open(Path(args.model_dir) / "config.json"))
            am = raw.get("auto_map", {}) or {}
            ref = am.get("AutoConfig")
            if isinstance(ref, (list, tuple)):
                ref = ref[0]
            if isinstance(ref, str):
                cls = get_class_from_dynamic_module(ref, args.model_dir)
                cfg_obj = cls.from_pretrained(args.model_dir, local_files_only=True)
                print(f"[load] explicit config {cls.__name__} ok", flush=True)
        except Exception as e:
            print(f"[load] explicit config failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
        base_kw = dict(trust_remote_code=args.trust_remote_code,
                       torch_dtype=load_dtype, local_files_only=True)
        if cfg_obj is not None:
            base_kw["config"] = cfg_obj
        for loader in [AutoModelForCausalLM, AutoModel]:
            try:
                model = loader.from_pretrained(args.model_dir, **base_kw)
                print(f"[load] {loader.__name__} with explicit config", flush=True)
                break
            except Exception as e:
                print(f"[load] {loader.__name__}+config failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
    if model is None:
        for cls in _explicit_classes():
            try:
                kw2 = dict(trust_remote_code=args.trust_remote_code,
                           torch_dtype=load_dtype, local_files_only=True)
                if cfg_obj is not None:
                    kw2["config"] = cfg_obj
                model = cls.from_pretrained(args.model_dir, **kw2)
                print(f"[load] via explicit {cls.__name__}", flush=True)
                break
            except Exception as e:
                print(f"[load] explicit {cls.__name__} failed: {type(e).__name__}: {str(e)[:110]}", flush=True)
    if model is None:
        raise SystemExit("no loader could instantiate this checkpoint")
    model = model.to(device).eval()
    # some remote configs (LLaDA) never define use_cache, which their generate() reads
    try:
        if not hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    except Exception:
        pass
    cfg = getattr(model, "config", None)
    if _sedd_mask_id is not None:
        # SEDD's absorbing index is `tokens` (50257), one past the last real GPT-2 id.
        mask_id = _sedd_mask_id
    else:
        mask_id = getattr(cfg, "mask_token_id", None)
        if mask_id is None:
            mask_id = tok.mask_token_id if tok.mask_token_id is not None else (cfg.vocab_size - 1)
    print(f"[load] impl={args.impl} mask_id={mask_id} "
          f"vocab={getattr(model, 'vocab_size', None) or getattr(cfg, 'vocab_size', None)}",
          flush=True)

    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        path = Path(args.task_data_dir) / f"{task}.jsonl"
        if not path.exists():
            print(f"[{task}] SKIP (no {path})", flush=True); continue
        out_path = out_root / f"{task}.jsonl"
        t0, n = time.time(), 0
        with path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as fo:
            for i, line in enumerate(f):
                if i >= args.max_samples: break
                line = line.strip()
                if not line: continue
                item: dict[str, Any] = json.loads(line)
                prompt = build_prompt(task, item)
                ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                if args.impl == "hf_diffusion":
                    gen_ids = hf_diffusion_generate(model, tok, ids, args.max_new_tokens, args.steps, device)
                elif args.impl == "sedd":
                    # SEDD's own loglinear schedule + score-entropy score; see sedd_native.
                    gen_ids = sedd_denoise(model, ids, args.max_new_tokens, args.steps,
                                           mask_id, device, args.temperature)
                else:
                    gen_ids = mdlm_denoise(model, ids, args.max_new_tokens, args.steps, mask_id,
                                           device, args.temperature)
                text = tok.decode(gen_ids, skip_special_tokens=True)
                for sp in getattr(tok, "all_special_tokens", ()):
                    text = text.replace(sp, " ")
                text = re.sub(r"[ \t]{2,}", " ", text).strip()
                if args.smoke:
                    print(f"  [smoke {i}] GT={get_ground_truth(item)[:40]!r} -> {text[:90]!r}", flush=True)
                rec = dict(item)
                rec.update({"id": item.get("id", i), "prompt": prompt, "generate": text,
                            "ground_truth": get_ground_truth(item), "choices": item.get("choices", []),
                            "prompt_tokens": int(ids.shape[1]), "generated_tokens": len(gen_ids),
                            "method": f"diffusion_{args.impl}", "cfg_scale": 0.0, "steps": args.steps,
                            "baseline_model": args.model_dir})
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if args.smoke and n >= args.smoke:
                    print(f"[{task}] smoke done ({n} samples)", flush=True); break
                if n % 20 == 0:
                    print(f"[{task}] {n} samples ({time.time()-t0:.0f}s)", flush=True)
        print(f"[{task}] done n={n} in {time.time()-t0:.0f}s -> {out_path}", flush=True)
        if args.smoke: break
    (out_root / "run_summary.json").write_text(json.dumps(
        {"model_dir": args.model_dir, "impl": args.impl, "steps": args.steps,
         "protocol": "generate-then-match (discrete diffusion)"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Efficiency and memory measurement for our models and HF baselines.

Measured per system, on one device, at a fixed input length:
  load_s            time to load the checkpoint (reported separately from compute)
  prefill_ms        one forward pass over a 512-token input
  decode_ms_tok     greedy decode of 64 tokens, batch 1 (autoregressive systems only)
  batch8_ms_tok     the same decode at batch 8 (the one-device parallel setting:
                    eight segments rendered together rather than in series)
  plan_ms           plan sampling + writing + injection (relay models only)
  peak_mem_gb       peak allocated device memory

Everything is measured after a warm-up pass and repeated, so the numbers are
steady-state and not JIT-compilation or co-tenancy artefacts.

NOT every system has a per-token decode step. Masked-diffusion (MDLM, SEDD) and
continuous-latent (CoLA-DLM, ELF) models denoise a whole span at once, so
`decode_ms_tok` is undefined for them: the comparable metric is the wall-clock to
produce a fixed-length completion, reported as `gen_total_ms` / `gen_len` /
`nfe` (network function evaluations). Those rows carry
`decode_ms_tok_bs1: null` and a `decode_undefined_reason` string rather than a
fabricated per-token figure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
DIFF = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
sys.path.insert(0, str(DIFF / "scripts" / "eval"))


MODELS = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/models")
GPT2 = MODELS / "gpt2-medium"   # MDLM/SEDD ship GPT-2 BPE; tiktoken is absent offline


def load_nonar(a, device):
    """Build a non-autoregressive checkpoint using the SAME code its accuracy rows use.

    Returns (model, tok, dtype). `model` is a plain module for mdlm/sedd and a
    small namespace for cola/elf, which are multi-network systems.
    """
    if a.impl == "sedd":
        # Raw hydra training checkpoint: no model_type/auto_map, so no HF auto-class can
        # build it. sedd_native rebuilds the published DDiT and strict-loads the weights.
        from sedd_native import load_sedd, mask_id_of
        from transformers import AutoTokenizer
        model, cfg = load_sedd(a.model_dir, device)
        model._mask_id = mask_id_of(cfg)
        tok = AutoTokenizer.from_pretrained(str(a.tokenizer_dir or GPT2),
                                            trust_remote_code=True, local_files_only=True)
        return model, tok, torch.float32   # fp32 weights; bf16 only inside autocast

    if a.impl == "mdlm":
        from transformers import AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        # MDLM calls sigma_map OUTSIDE its own autocast block, so bf16 weights meet an
        # fp32 sigma and die with "mat1 and mat2 must have the same dtype". The published
        # checkpoint is fp32 on disk; keep it there.
        tok = AutoTokenizer.from_pretrained(str(a.tokenizer_dir or GPT2),
                                            trust_remote_code=True, local_files_only=True)
        # AutoModel does NOT work here: mdlm-owt's auto_map declares only AutoConfig and
        # AutoModelForMaskedLM, so AutoModel.from_pretrained ends in
        #   ValueError: Unrecognized configuration class ...MDLMConfig for this kind of
        #               AutoModel
        # (this is what killed the first eff2 attempt). Resolve the class straight out of
        # auto_map, exactly as eval_diffusion_lm_gen_match.py does for the accuracy rows.
        cfg = json.load(open(Path(a.model_dir) / "config.json"))
        ref = (cfg.get("auto_map") or {}).get("AutoModelForMaskedLM")
        if isinstance(ref, (list, tuple)):
            ref = ref[0]
        if not isinstance(ref, str):
            raise SystemExit(f"mdlm: no AutoModelForMaskedLM in auto_map of {a.model_dir}")
        cls = get_class_from_dynamic_module(ref, a.model_dir)
        # MDLM's remote code never calls PreTrainedModel.post_init(), which is where
        # transformers 5.3 sets `all_tied_weights_keys`; _finalize_model_loading() then reads
        # it unconditionally and from_pretrained dies with
        #   AttributeError: 'MDLM' object has no attribute 'all_tied_weights_keys'
        # regardless of whether an explicit config object is passed. An empty mapping is the
        # CORRECT value here, not a workaround that skips real work: MDLM declares
        # `_tied_weights_keys = None`, and the checkpoint stores the input embedding
        # (backbone.vocab_embed.embedding) and the output head (backbone.output_layer.linear)
        # as two independent tensors, so this checkpoint ties nothing at all.
        if not hasattr(cls, "all_tied_weights_keys"):
            cls.all_tied_weights_keys = {}
        model = cls.from_pretrained(a.model_dir, trust_remote_code=True,
                                    torch_dtype=torch.float32,
                                    local_files_only=True).to(device).eval()
        print(f"[load] mdlm via {cls.__name__} (from auto_map)", flush=True)
        model._mask_id = getattr(model.config, "mask_token_id", None) or (
            tok.mask_token_id if tok.mask_token_id is not None else model.config.vocab_size - 1)
        return model, tok, torch.float32

    if a.impl == "cola":
        # DiT prior + Text VAE + its own 100278-vocab BPE. The weights dir holds no code,
        # so the importable `cola_dlm` package is resolved separately.
        import types
        sys.path.insert(0, str(REPO / "scripts" / "eval"))
        from eval_cola_dlm_gen_match import resolve_code_dir
        md = Path(a.model_dir)
        code_dir = resolve_code_dir(str(md), None)
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        print(f"[load] cola_dlm code: {code_dir}", flush=True)
        from cola_dlm import ColaDiTModel, ColaTextVAEModel
        from tokenizers import Tokenizer
        dit = ColaDiTModel.from_pretrained(str(md / "cola_dlm" / "cola_dit"),
                                           local_files_only=True).to(device).eval()
        vae = ColaTextVAEModel.from_pretrained(str(md / "cola_dlm" / "cola_vae"),
                                               local_files_only=True).to(device).eval()
        raw = Tokenizer.from_file(str(md / "tokenizer.json"))
        ns = types.SimpleNamespace(dit=dit, vae=vae, raw_tok=raw)
        print(f"[load] cola DiT {sum(p.numel() for p in dit.parameters())/1e9:.3f}B "
              f"VAE {sum(p.numel() for p in vae.parameters())/1e9:.3f}B "
              f"block_size={dit.block_size} latent_dim={vae.latent_dim}", flush=True)
        return ns, raw, torch.float32

    if a.impl == "elf":
        # Flow matching over frozen t5-small embeddings; the T5 ENCODER is a second
        # network on the device and its cost belongs in the total.
        import types
        from eval_elf_gen_match import ELFSampler, T5TextEncoder, build_elf_model
        model, meta = build_elf_model(Path(a.model_dir), use_ema=True, device=device)
        steps = a.diff_steps or 64
        sampler = ELFSampler(model, sampling_method="sde", num_sampling_steps=steps,
                             sde_gamma=1.0, self_cond_cfg_scale=3.0)
        txt = T5TextEncoder(model_name=str(a.tokenizer_dir or "t5-small"), device=device)
        print(f"[load] elf arch={meta['arch']} weights={meta['weight_source']} "
              f"ckpt={meta['ckpt']}", flush=True)
        ns = types.SimpleNamespace(model=model, sampler=sampler, txt=txt, meta=meta)
        return ns, txt.tok, torch.float32

    raise SystemExit(f"load_nonar: unsupported impl {a.impl}")


def _filler_ids(tok, n, impl):
    """A real length-n token sequence (random ids fall outside some vocabs)."""
    s = ("The recurrent state carries the evidence forward and the decoder renders from it. "
         * 200)
    if impl == "cola":
        ids = list(tok.encode(s).ids)
    else:
        ids = tok(s, add_special_tokens=False).input_ids
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


@torch.no_grad()
def measure_nonar(a, model, tok, device):
    """Wall-clock for a fixed-length completion, plus the exact forward-pass count.

    These systems have no per-token decode step, so `decode_ms_tok_bs1` is recorded as
    null with a reason string instead of a number that does not correspond to anything
    the architecture does.
    """
    out = {"gen_len": a.gen_len}

    def timed(fn, warmup=1, n=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / n * 1000.0

    if a.impl in ("mdlm", "sedd"):
        steps = a.diff_steps or 128     # both checkpoints' own published default
        ids = torch.tensor([_filler_ids(tok, a.len, a.impl)], device=device)
        mask_id = model._mask_id
        if a.impl == "sedd":
            from sedd_native import loglinear_sigma, sedd_denoise
            sig = torch.full((1,), float(loglinear_sigma(torch.tensor(1.0))), device=device)
            x = torch.cat([ids, torch.full((1, a.gen_len), mask_id,
                                           device=device, dtype=torch.long)], dim=1)

            def one():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(x, sig)

            def gen():
                sedd_denoise(model, ids, a.gen_len, steps, mask_id, device, 1.0)
        else:
            from eval_diffusion_lm_gen_match import mdlm_denoise
            t1 = torch.tensor([1.0], device=device, dtype=torch.float32)
            x = torch.cat([ids, torch.full((1, a.gen_len), mask_id,
                                           device=device, dtype=torch.long)], dim=1)

            def one():
                model(input_ids=x, timesteps=t1)

            def gen():
                mdlm_denoise(model, ids, a.gen_len, steps, mask_id, device, 1.0)

        out["denoise_steps"] = steps
        out["forward_ms_1step"] = round(timed(one), 1)
        out["gen_total_ms"] = round(timed(gen, warmup=1, n=2), 1)
        out["nfe"] = steps
        out["nfe_formula"] = "steps (one full non-causal forward per step, no KV cache)"
        out["decode_ms_tok_bs1"] = None
        out["decode_undefined_reason"] = (
            f"masked diffusion: {steps} full-sequence denoising passes over the whole "
            f"prompt+completion grid; generation length changes the tensor width, not the "
            f"number of passes, so there is no per-token decode step")
        out["gen_ms_per_tok_amortised"] = round(out["gen_total_ms"] / a.gen_len, 2)
        return out

    if a.impl == "cola":
        dit, vae, raw = model.dit, model.vae, model.raw_tok
        steps = a.diff_steps or 16
        from eval_cola_dlm_gen_match import TokenizerProxy, flush_kv
        from cola_dlm import generate_task_repaint_inference
        prompt = raw.decode(_filler_ids(raw, a.len, "cola"))
        proxy = TokenizerProxy(raw)

        def gen():
            flush_kv(dit, vae)
            generate_task_repaint_inference(
                dit=dit, vae=vae, tokenizer=proxy, prompts=[{"id": 0, "question": prompt}],
                task_name="raw", device=device, T=1000.0, timestep_num=steps,
                guidance_scale=7.0, max_new_tokens=a.gen_len, temperature=1.0,
                top_k=0, top_p=1.0, repetition_penalty=1.0,
                pad_token_id=100277, eos_token_id=100257, im_end_token_id=100265)

        blocks = -(-a.gen_len // (dit.block_size * dit.config.patch_size))
        out["denoise_steps"] = steps
        out["gen_total_ms"] = round(timed(gen, warmup=1, n=2), 1)
        out["nfe"] = 1 + 2 * steps * blocks + blocks
        out["nfe_formula"] = (f"DiT: 1 prefix + 2*steps*blocks (CFG runs a SEPARATE "
                              f"unconditional pass) + blocks commit, blocks={blocks}; "
                              f"plus {2 + blocks} VAE passes")
        out["block_size"] = int(dit.block_size)
        out["decode_ms_tok_bs1"] = None
        out["decode_undefined_reason"] = (
            f"continuous latent diffusion in {dit.block_size}-token blocks: cost is "
            f"{steps} flow steps x 2 (CFG) per block, not per token")
        out["gen_ms_per_tok_amortised"] = round(out["gen_total_ms"] / a.gen_len, 2)
        return out

    if a.impl == "elf":
        sampler, txt = model.sampler, model.txt
        gen_t = torch.Generator(device="cpu"); gen_t.manual_seed(0)
        prompt = txt.tok.decode(_filler_ids(txt.tok, a.len, "elf"),
                                skip_special_tokens=True)
        max_prompt = min(a.len, 1024 - a.gen_len)
        ids, msk, lat = txt.encode([prompt], max_prompt)
        L = int(msk[0].sum().item())
        canvas = min(1024, L + a.gen_len)
        cond = torch.zeros((1, canvas, sampler.model.text_encoder_dim),
                           dtype=torch.float32, device=device)
        cmask = torch.zeros((1, canvas), dtype=torch.float32, device=device)
        cond[0, :L] = lat[0, :L].to(device); cmask[0, :L] = 1.0

        def gen():
            z, tf = sampler.sample(cond, cmask, canvas, gen_t)
            sampler.decode_ids(z, tf)

        out["denoise_steps"] = sampler.n_steps
        out["canvas_len"] = canvas
        out["gen_total_ms"] = round(timed(gen, warmup=1, n=2), 1)
        out["nfe"] = sampler.n_steps + 1
        out["nfe_formula"] = (f"{sampler.n_steps} denoiser passes + 1 decode-head pass; "
                              f"guidance is a conditioning TOKEN, not a second pass, so "
                              f"CFG is free; independent of generation length")
        out["decode_ms_tok_bs1"] = None
        out["decode_undefined_reason"] = (
            f"continuous flow matching over a FIXED canvas of {canvas} token-aligned "
            f"512-d T5 embeddings: cost is {sampler.n_steps}+1 passes regardless of "
            f"generation length, which is realised as a slice of the decoded canvas")
        out["gen_ms_per_tok_amortised"] = round(out["gen_total_ms"] / a.gen_len, 2)
        return out

    raise SystemExit(f"measure_nonar: unsupported impl {a.impl}")


@torch.no_grad()
def time_forward(model, ids, mask, n=3):
    """One warmed-up prefill forward over `ids`, averaged over n runs.

    NOTE: this def line was lost when measure_nonar() was inserted above it -- the body
    survived as unreachable code after measure_nonar's final `raise`, which still compiled
    cleanly, so `python -m py_compile` passed while every AR run died at call time with
    `NameError: name 'time_forward' is not defined`. Four jobs were burned on that.
    """
    for _ in range(1):
        model(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        model(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


@torch.no_grad()
def time_decode(model, ids, mask, steps=64, warmup=1, n=3):
    """Greedy decode of `steps` tokens, warmed up and repeated.

    The original version did NEITHER: it timed a single un-warmed `generate` call, so the
    measurement absorbed one-off triton JIT compilation and any co-tenant GPU contention.
    That is not a cosmetic issue -- it produced a published artefact. On 2026-09-20 the four
    systems that share the SAME frozen bf16 2.9B RWKV-7 backbone (verified: both relay
    checkpoints carry zero `rwkv_model.*` keys, so the backbone weights are byte-identical)
    measured 381.26 / 378.08 / 374.87 / 373.52 ms/tok, while global-dit-50k measured 199.66
    -- a 1.87x "speedup" of a model against itself. Returns (warm_median, cold_first) so the
    cold number stays on record and that artefact remains explainable.
    """
    cold = None
    for _ in range(max(warmup, 0)):
        torch.cuda.synchronize(); t0 = time.time()
        model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=steps,
                       do_sample=False, pad_token_id=0)
        torch.cuda.synchronize()
        if cold is None:
            cold = (time.time() - t0) * 1000.0 / steps
    runs = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.time()
        model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=steps,
                       do_sample=False, pad_token_id=0)
        torch.cuda.synchronize()
        runs.append((time.time() - t0) * 1000.0 / steps)
    runs.sort()
    return runs[len(runs) // 2], (cold if cold is not None else runs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True,
                   choices=["relay", "hf", "mdlm", "sedd", "cola", "elf"])
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--tokenizer_dir", default=None,
                   help="tokenizer override (MDLM/SEDD ship GPT-2 BPE; tiktoken is absent)")
    ap.add_argument("--gen_len", type=int, default=64,
                   help="completion length for the fixed-length generation metric "
                        "(non-AR models have no per-token decode step)")
    ap.add_argument("--diff_steps", type=int, default=0,
                   help="denoising/sampling steps; 0 = each impl's published default")
    ap.add_argument("--vocab_size", type=int, default=65536)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--plan_steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    device = torch.device(a.device)
    torch.cuda.reset_peak_memory_stats()
    res = {"impl": a.impl, "input_len": a.len}
    t0 = time.time()
    if a.impl == "relay":
        from scripts.eval.relay_utils import load_relay_model
        model, _rwkv, tok, _ck, _cfg = load_relay_model(a.ckpt_dir, device)
        backbone = model.rwkv_model
        model.eval(); model._prefix_suffix_s2 = True
        dtype = next(model.latent_dit.parameters()).dtype
    elif a.impl in ("mdlm", "sedd", "cola", "elf"):
        # Non-autoregressive systems. They have NO per-token decode step, so the
        # comparable metric is wall-clock for a fixed-length completion; see
        # measure_nonar() below. Loading is delegated to the adapter that already
        # runs each of them for the accuracy tables, so the model built here is
        # bit-identical to the one the paper's other numbers come from.
        model, tok, dtype = load_nonar(a, device)
        backbone = model
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # trust_remote_code is forced on: RWKV-6/RWKV-7 ship custom modelling code in
        # auto_map and die with "contains custom code which must be executed" when the
        # caller's store_true flag defaults to False -- that is exactly how the 09-20
        # batch lost rwkv6-world and rwkv7-world3. The five models that already landed
        # (qwen x2, llama, phi-2, gemma-2, mamba) declare NO auto_map, so for them
        # trust_remote_code is a no-op and their published numbers stay comparable.
        tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True,
                                           local_files_only=True)
        kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16,
                  local_files_only=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(a.model_dir, **kw)
        except ValueError as e:
            if "does not recognize this architecture" not in str(e):
                raise
            # fla-hub snapshots (gla-1.3B-100B) declare model_type `gla` but ship no
            # remote-code file, so the class only resolves after `fla` registers it.
            # Import it ONLY here: fla.models also re-registers `mamba`, which would
            # hijack mamba-2.8b into fla's slow path and silently change its numbers.
            arch = (json.load(open(Path(a.model_dir) / "config.json")).get("architectures")
                    or [None])[0]
            import fla.models as _fm
            if arch and hasattr(_fm, arch):
                model = getattr(_fm, arch).from_pretrained(a.model_dir, **kw)
            else:
                model = AutoModelForCausalLM.from_pretrained(a.model_dir, **kw)
        model = model.to(device).eval()
        backbone = model
        dtype = torch.bfloat16
    res["load_s"] = round(time.time() - t0, 1)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    if a.impl in ("mdlm", "sedd", "cola", "elf"):
        res.update(measure_nonar(a, model, tok, device))
        res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2), flush=True)
        return

    # build a real token sequence of the requested length (random ids fall outside some vocabs
    # and trigger device-side asserts), then tile it for the batch
    filler = ("The recurrent state carries the evidence forward and the decoder renders from it. " * 200)
    enc = tok(filler, return_tensors="pt", add_special_tokens=False).input_ids
    while enc.shape[1] < a.len:
        enc = torch.cat([enc, enc], dim=1)
    enc = enc[:, :a.len]
    for bs in (1, 8):
        ids = enc.expand(bs, -1).contiguous().to(device)
        mask = torch.ones_like(ids)
        prefill = time_forward(backbone, ids, mask)
        dec, dec_cold = time_decode(backbone, ids, mask, a.steps)
        res[f"prefill_ms_bs{bs}"] = round(prefill, 1)
        res[f"decode_ms_tok_bs{bs}"] = round(dec, 2)
        # kept on record so the 09-20 numbers stay comparable/explainable: those were
        # single un-warmed calls, i.e. this cold figure, not the warm steady state
        res[f"decode_ms_tok_bs{bs}_cold"] = round(dec_cold, 2)
        del ids, mask
        torch.cuda.empty_cache()

    if a.impl == "relay":
        from scripts.eval.sample_prefix_suffix_cfg import encode_prefix, sample_ddim_cfg
        ids = enc[:, : a.len // 2].to(device)
        mask = torch.ones_like(ids)
        encode_prefix(model, ids, mask)
        torch.cuda.synchronize(); t0 = time.time()
        cond = encode_prefix(model, ids, mask)
        z = sample_ddim_cfg(model, cond, a.plan_steps, a.cfg_scale, device, dtype)
        states = model.predict_states(z)
        cache = model.inject_into_cache(
            model.rwkv_model(input_ids=ids, attention_mask=mask.bool(),
                             use_cache=True, return_dict=True).past_key_values, states)
        torch.cuda.synchronize()
        res["plan_ms"] = round((time.time() - t0) * 1000, 1)
        res["plan_steps"] = a.plan_steps
    res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()

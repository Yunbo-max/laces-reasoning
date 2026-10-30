#!/usr/bin/env python3
"""Language-modeling perplexity for both our models and HF baselines.

Two implementations, one protocol:
  --impl relay   our state-injection models. Each sequence is split into a prefix
                 and a suffix; the prefix is encoded, a plan is sampled, and the
                 writer's memory is injected before the suffix is scored. --mode
                 selects raw (no plan written), prefix_cfg (single global plan) or
                 traj_cfg (chunked trajectory), so the table shows raw vs written
                 memory on the same checkpoint.
  --impl hf      any HF causal LM; the whole sequence is scored.

Reported number is token perplexity on the scored span (the suffix for relay
models, the full sequence for HF models). Corpora are the preprocessed 512-token
shards already on shared storage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
DIFF = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
sys.path.insert(0, str(DIFF / "scripts" / "eval"))

CORPORA = {
    "wikitext2": "wikitext2_full", "wikitext103": "wikitext103", "pg19": "pg19",
    "lm1b": "lm1b", "ptb": "ptb", "wmt14": "wmt14_de_en", "xsum": "xsum",
}

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
    except Exception:
        pass
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


def npz_files(corpus: str, n: int):
    root = DIFF / "preprocessed_data" / CORPORA[corpus]
    files = sorted(root.rglob("*.npz"))
    return files[:n]


@torch.no_grad()
def score_hf(model, ids, device, text=None, tok=None):
    if text is not None and tok is not None:
        enc = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        if enc.shape[1] < 8:
            return 0.0, 0, 0
        ids = enc.to(device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), return_dict=True)
    logits = out.logits[:, :-1].float()
    tgt = ids[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
    nbytes = len(text.encode("utf-8")) if text is not None else int(tgt.numel())
    return float(nll), int(tgt.numel()), nbytes


def span_bytes(tok, id_seq):
    """UTF-8 byte length of the scored span, for a bits-per-byte that means what it says.

    The relay branches used to pass ``nbytes = ntok``, so every relay row's
    "bits_per_byte" was really bits-per-TOKEN and sat ~5x above the HF rows
    (RWKV world tokenizer averages ~5 bytes/token): chunked wikitext2 reported
    3.8888 == log2(14.81) exactly, against 0.74-0.81 for the HF baselines. Any
    table putting those two columns side by side was comparing different units.
    """
    if tok is None:
        return 0
    try:
        return len(tok.decode(list(int(t) for t in id_seq), skip_special_tokens=True).encode("utf-8"))
    except Exception:
        return 0


@torch.no_grad()
def score_relay_traj(model, ids, split, steps, cfg_scale, device, dtype, tok=None):
    """Chunk-wise trajectory injection, the mode chunked checkpoints are trained in:
    sample a trajectory from the prefix, then advance chunk by chunk, refreshing the cache
    after each chunk's state write, and score the suffix tokens teacher-forced."""
    from scripts.eval.sample_prefix_suffix_trajectory_cfg import (
        apply_generated_state, encode_prefix, sample_trajectory_cfg)
    from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (
        recompute_logits_from_injected_cache)
    prefix, suffix = ids[:, :split], ids[:, split:]
    z_prefix, cache, _ = encode_prefix(model, prefix, torch.ones_like(prefix))
    z_traj = sample_trajectory_cfg(model, z_prefix, steps, cfg_scale, device, dtype)
    blend = float(model.config.get("trajectory_state_blend", getattr(model, "trajectory_state_blend", 1.0)))
    chunk_size = int(model.trajectory_chunk_size)
    ctx = list(ids[0, :split].tolist())
    total_nll, n_tok, pos = 0.0, 0, 0
    for h in range(z_traj.shape[1]):
        if pos >= suffix.numel():
            break
        states_h = model.predict_states(z_traj[:, h])
        cache = apply_generated_state(model, cache, states_h, blend)
        c = torch.tensor([ctx], device=device, dtype=torch.long)
        cache, lb = recompute_logits_from_injected_cache(model, c, torch.ones_like(c), cache)
        logits = lb[0]
        for _ in range(chunk_size):
            if pos >= suffix.numel():
                break
            lp = torch.log_softmax(logits.float(), dim=-1)
            total_nll += float(-lp[suffix[0, pos]])
            n_tok += 1
            nxt = suffix[0, pos:pos + 1].view(1, 1)
            out = model.rwkv_model(input_ids=nxt, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            logits = out.logits[0, -1]
            ctx.append(int(suffix[0, pos]))
            pos += 1
    return total_nll, n_tok, span_bytes(tok, suffix[0, :pos].tolist())


@torch.no_grad()
def score_relay(model, ids, split, mode, steps, cfg_scale, device, dtype, tok=None):
    """Score the suffix under a (optionally plan-injected) prefix.

    Alignment: the suffix token at absolute position ``split`` must be predicted from
    the distribution the model emits *after* consuming position ``split-1``. The raw
    branch takes that from the full-sequence logits slice ``[split-1 : L-1]``. The
    injected branch must do the same, and the only way to get a post-injection logit
    for position ``split`` is to feed position ``split-1`` again through the injected
    cache -- hence the prefix cache is built from ``ids[:, :split-1]`` and the scored
    forward is fed ``ids[:, split-1:]`` (the extra leading token is context, not a
    target). That is exactly the convention training uses (see
    ``_singlez_coadapt_ce_from_z``: ``logits[:, :-1]`` against ``suffix[:, 1:]``) and
    that ``eval_prefix_suffix_cfg_ppl_matrix.score_cfg_suffix`` already used.
    """
    if mode == "raw":
        out = model.rwkv_model(input_ids=ids, attention_mask=torch.ones_like(ids).bool(),
                               return_dict=True)
        logits = out.logits[:, split - 1:ids.shape[1] - 1].float()
        tgt = ids[:, split:]
    else:
        from scripts.eval.sample_prefix_suffix_cfg import encode_prefix, sample_ddim_cfg
        # cfg_scale must be 1.0 here: a CFG-extrapolated epsilon defines no normalised
        # density, so a guided plan is not a conditional likelihood. The caller enforces
        # it; this is the plain conditional plan.
        prefix = ids[:, :split - 1]          # cache context: up to split-2
        scored = ids[:, split - 1:]          # first token is context, rest are targets
        cond = encode_prefix(model, prefix, torch.ones_like(prefix))
        z = sample_ddim_cfg(model, cond, steps, cfg_scale, device, dtype)
        states = model.predict_states(z)
        cache = model.inject_into_cache(
            model.rwkv_model(input_ids=prefix, attention_mask=torch.ones_like(prefix).bool(),
                             use_cache=True, return_dict=True).past_key_values, states)
        out = model.rwkv_model(input_ids=scored, attention_mask=torch.ones_like(scored).bool(),
                               past_key_values=cache, use_cache=False, return_dict=True)
        logits = out.logits[:, :-1].float()
        tgt = scored[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
    return float(nll), int(tgt.numel()), span_bytes(tok, tgt[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=["relay", "hf"])
    ap.add_argument("--ckpt_dir", default=None, help="relay checkpoint")
    ap.add_argument("--model_dir", default=None, help="HF model directory")
    ap.add_argument("--mode", default="prefix_cfg", choices=["raw", "prefix_cfg", "traj_cfg"])
    ap.add_argument("--corpora", default="wikitext2,wikitext103,pg19,lm1b,ptb,wmt14,xsum")
    ap.add_argument("--max_files", type=int, default=100)
    ap.add_argument("--split", type=int, default=256, help="prefix length for relay scoring")
    ap.add_argument("--split_frac", type=float, default=0.5,
                    help="fallback prefix fraction when a sequence is shorter than --split; "
                         "lm1b/ptb/wmt14 are 8-80 tokens, so a fixed 256 drops every sample")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=1.0,
                    help="guidance for the plan sampler. MUST stay 1.0 for likelihood: a "
                         "CFG-extrapolated epsilon is not a normalised density.")
    ap.add_argument("--allow_cfg_ppl", action="store_true",
                    help="escape hatch to report a non-likelihood guided number anyway")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--tokenizer_dir", default=None,
                    help="tokenizer dir override (MDLM/SEDD use GPT-2 BPE; tiktoken unavailable)")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    if a.impl == "relay" and a.mode != "raw" and a.cfg_scale != 1.0 and not a.allow_cfg_ppl:
        raise SystemExit(
            f"--cfg_scale={a.cfg_scale} cannot be used for perplexity. Classifier-free guidance "
            "extrapolates the denoiser output (eps_u + s*(eps_c - eps_u)); the resulting plan is "
            "not a sample from the conditional latent posterior, so the token distribution it "
            "induces is not the model's normalised conditional likelihood and exp(mean NLL) under "
            "it is not a perplexity. Use --cfg_scale 1.0, or pass --allow_cfg_ppl to record a "
            "guided number explicitly labelled as not-a-likelihood.")

    device = torch.device(a.device)
    if a.impl == "relay":
        from scripts.eval.relay_utils import load_relay_model
        model, _rwkv, relay_tok, _ck, _cfg = load_relay_model(a.ckpt_dir, device)
        model.eval(); model._prefix_suffix_s2 = True
        dtype = next(model.latent_dit.parameters()).dtype
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer_dir or a.model_dir, trust_remote_code=True,
                                           local_files_only=True)
        # DCLM (model_type "openlm", architectures ["OpenLMModel"]) ships no auto_map and no
        # modeling file, so from_pretrained raises ValueError("...does not recognize this
        # architecture") no matter what trust_remote_code says. mlfoundations' `open_lm`
        # self-registers the classes with the Auto classes on import. Registering before the
        # fla import below also keeps fla out of the way: it knows nothing about openlm and
        # re-registers `mamba`, a documented hazard here. Gated on model_type, so no other
        # checkpoint's path changes. NB `os` is NOT imported at module level in this file
        # (that is the root cause of the inert dtype probe in _load_hf_model above), hence the
        # local import -- without it this block would raise NameError exactly the same way.
        try:
            import os as _os2
            if json.load(open(_os2.path.join(str(a.model_dir),
                                             "config.json"))).get("model_type") == "openlm":
                import open_lm.hf  # noqa: F401  registers OpenLMConfig / OpenLMForCausalLM
                print("[load] model_type='openlm': imported open_lm.hf to register "
                      "OpenLMForCausalLM with the Auto classes", flush=True)
        except FileNotFoundError:
            pass
        try:
            # fla-hub checkpoints declare architectures (e.g. model_type `gla`) that
            # only resolve once fla registers them; snapshots ship no remote code.
            import fla.models  # noqa: F401
        except Exception:
            pass
        model = _load_hf_model(a.model_dir, device, trust_remote_code=a.trust_remote_code)
        dtype = torch.bfloat16
    # corpora are tokenised with the RWKV world tokenizer; for a model with a different
    # tokenizer we decode back to text and re-encode, and report bits-per-byte so the numbers
    # stay comparable across vocabularies.
    rwkv_tok = None
    if a.impl == "hf":
        from transformers import AutoTokenizer as _AT
        rwkv_tok = _AT.from_pretrained("/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/laces-reasoning/models/RWKV7-Goose-World3-2.9B-HF",
                                       trust_remote_code=True, local_files_only=True)

    results = {"impl": a.impl, "mode": a.mode, "ckpt": a.ckpt_dir or a.model_dir,
               "max_files": a.max_files, "perplexity": {}}
    if a.impl == "relay":
        results["cfg_scale"] = a.cfg_scale
        results["split"] = a.split
        results["split_frac"] = a.split_frac
        if a.cfg_scale != 1.0:
            results["WARNING"] = ("cfg_scale != 1.0: guided plans do not define a normalised "
                                  "conditional density; these numbers are NOT perplexities")
    for corpus in [c.strip() for c in a.corpora.split(",") if c.strip()]:
        files = npz_files(corpus, a.max_files)
        if not files:
            print(f"[{corpus}] no files", flush=True); continue
        tot_nll, tot_tok, tot_bytes, t0 = 0.0, 0, 0, time.time()
        n_used, n_skip = 0, 0
        for fp in files:
            d = np.load(fp)
            ids = torch.from_numpy(d["input_ids"].astype(np.int64)).unsqueeze(0).to(device)
            if a.impl == "relay":
                L = int(ids.shape[1])
                # A fixed --split of 256 silently discarded every lm1b/ptb/wmt14 sample
                # (8-80 tokens long), leaving tot_tok == 0 -> exp(0/max(1,0)) == PPL 1.0
                # and bpb 0.0, which is how those three corpora came out "perfect" for
                # both relay models while all 13 HF baselines had real numbers. Short
                # sequences now split proportionally instead of being dropped.
                split = a.split if L > a.split + 4 else max(2, int(L * a.split_frac))
                # need >=1 context token before the scored span and >=1 target after it
                if L < 4 or split < 2 or split >= L:
                    n_skip += 1
                    continue
                if a.mode == "traj_cfg":
                    nll, ntok, nbytes = score_relay_traj(model, ids, split, a.steps, a.cfg_scale,
                                                         device, dtype, tok=relay_tok)
                else:
                    nll, ntok, nbytes = score_relay(model, ids, split, a.mode, a.steps, a.cfg_scale,
                                                    device, dtype, tok=relay_tok)
                if ntok == 0:
                    n_skip += 1
                    continue
            else:
                text = rwkv_tok.decode(ids[0].tolist(), skip_special_tokens=True)
                nll, ntok, nbytes = score_hf(model, ids, device, text=text, tok=tok)
                if ntok == 0:
                    n_skip += 1
                    continue
            n_used += 1
            tot_nll += nll; tot_tok += ntok; tot_bytes += nbytes
        if tot_tok == 0:
            # never report exp(0/1) == 1.0 as if it were a perplexity
            print(f"[{corpus}] SKIPPED: 0 tokens scored ({n_skip} files unusable)", flush=True)
            results["perplexity"][corpus] = None
            results.setdefault("bits_per_byte", {})[corpus] = None
            results.setdefault("n_tokens", {})[corpus] = 0
            Path(a.output).parent.mkdir(parents=True, exist_ok=True)
            Path(a.output).write_text(json.dumps(results, indent=2))
            continue
        ppl = math.exp(tot_nll / tot_tok)
        results["perplexity"][corpus] = round(ppl, 2)
        results.setdefault("bits_per_byte", {})[corpus] = (
            round(tot_nll / tot_bytes / math.log(2), 4) if tot_bytes else None)
        results.setdefault("n_tokens", {})[corpus] = tot_tok
        results.setdefault("n_docs", {})[corpus] = n_used
        print(f"[{corpus}] PPL={ppl:.2f} over {tot_tok} tokens / {n_used} docs "
              f"({n_skip} skipped, {time.time()-t0:.0f}s)", flush=True)
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

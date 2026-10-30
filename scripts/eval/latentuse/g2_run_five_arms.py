#!/usr/bin/env python3
"""G2: matched latent-use controls -- five arms x 3 tasks x 200 examples.

Fills tab:latent-use-pending in paper_ICLR27/lalaland/appendix.tex.

The five arms (one checkpoint, one process, one tokenizer, one loaded model):

  A0 no_adapter      decode loop runs, adapter NEVER called, principal memory not written.
  A1 identity        adapter called (predict_states + inject_into_cache), then the ORIGINAL
                     principal memory (recurrent_state) restored bit-exactly via torch.equal
                     verification.  Whatever inject_into_cache did to auxiliary fields
                     (conv_state/ffn_state zeroing) and counters is LEFT INTACT.
  A2 mean_latent     write R(mean_train_latent) from the G1 tensor.
  A3 shuffled        write the sampled latent of a DIFFERENT prompt (derangement within task).
  A4 correct         write the correct prompt's sampled latent.

A0 is NOT lambda=0.  blend_into_cache(blend=0) keeps principal state but still zeroes
conv/ffn and calls _reset_cache_layer_seen_tokens + cache._seen_tokens=0;
inject_into_cache does NOT touch counters at all.  A0 skips the adapter entirely, A1 keeps
the adapter's other side effects.  Two distinct branches -- see CACHE_POLICY notes below.

PREFIX REPLAY is an explicit flag applied IDENTICALLY to all five arms, default OFF
(training-consistent: _singlez_coadapt_ce_from_z injects then forwards the SUFFIX only,
never the prefix again).

Invalid answers count as WRONG.  Never dropped, never shrink the denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.sample_prefix_suffix_cfg import (  # noqa: E402
    apply_repetition_penalty,
    apply_top_p,
    cosine_alpha_bar,
    encode_prefix,
)

TASKS = ("sciq", "commonsenseqa", "race")
ARMS = ("A0_no_adapter", "A1_identity", "A2_mean_latent", "A3_shuffled", "A4_correct")

# ---------------------------------------------------------------------------
# scorer: reuse the repo's single unified generate-then-match scorer verbatim.
# ---------------------------------------------------------------------------
_COLA = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/baseline/Cola-DLM"
if _COLA not in sys.path:
    sys.path.insert(0, _COLA)
_SGM = _REPO / "scripts" / "eval"
if str(_SGM) not in sys.path:
    sys.path.insert(0, str(_SGM))
import score_gen_match as SGM  # noqa: E402


def score_one(task: str, generate: str, ground_truth: str, choices: list[str]) -> tuple[bool, bool]:
    """Return (correct, invalid) for one record under the repo scorer's rule set.

    invalid == the scorer could not resolve the generation to any option at all.
    invalid rows are still scored (as wrong); they never leave the denominator.
    """
    gen = SGM.truncate(generate)
    if not choices:
        raise RuntimeError(f"task {task}: no choices -- all three G2 tasks must be choice tasks")
    # GUARD for a latent bug in the shared scorer, fixed HERE rather than by editing
    # score_gen_match.py (which would move other tables' numbers):
    #   letter_of("") returns "" because `len(t)<=2 and t.upper() in LETTERS` is True for
    #   the empty string ("" is a substring of "ABCDEFGH"), and LETTERS.index("") == 0.
    # So an EMPTY generation would be scored as option A instead of invalid.  Measured
    # frequency in the existing 1200 published rows for these three tasks: 0, so no
    # reported number depends on it -- but A0/A1 (no effective write) can plausibly emit
    # an immediate EOS, and silently crediting those rows with option A would inflate
    # exactly the arms we are trying to measure.
    if not gen.strip():
        return False, True
    pred = SGM.match_choice(gen, choices)
    if pred is None:
        pred = SGM.match_choice(SGM.strip_label(gen), choices)
    if pred is None:
        pred = SGM.match_choice(SGM.strip_answer_lead(SGM.strip_label(gen)), choices)
    if pred is None:
        lt = SGM.letter_of(gen)
        if lt is not None and SGM.LETTERS.index(lt) < len(choices):
            pred = SGM.LETTERS.index(lt)
    gold = None
    ng = SGM.normalize_text(ground_truth)
    for i, c in enumerate(choices):
        if SGM.normalize_text(c) == ng:
            gold = i
            break
    if gold is None:
        gold = SGM.match_choice(ground_truth, choices)
    if gold is None:
        raise RuntimeError(f"task {task}: gold {ground_truth!r} not resolvable against {choices!r}")
    invalid = pred is None
    correct = (not invalid) and pred == gold
    return bool(correct), bool(invalid)


# ---------------------------------------------------------------------------
# cache handling
# ---------------------------------------------------------------------------
def cache_layer_state(cache, layer_idx: int):
    """Mutable per-layer state dict, for FLA 0.3 (.states) or 0.4 (.layers) caches.

    Raises on an unknown cache shape rather than silently skipping.
    """
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx].state
    if hasattr(cache, "states"):
        return cache.states[layer_idx]
    raise RuntimeError(f"unknown cache type {type(cache)!r}: neither .layers nor .states")


def n_cache_layers(cache) -> int:
    if hasattr(cache, "layers"):
        return len(cache.layers)
    if hasattr(cache, "states"):
        return len(cache.states)
    raise RuntimeError(f"unknown cache type {type(cache)!r}")


def deep_copy_cache(cache):
    """Structural deep copy: every tensor cloned, no buffer shared with the source.

    Every arm decodes from its own copy so that no two arms can share a mutable cache.
    """
    import copy as _copy

    def clone(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().clone()
        if isinstance(obj, dict):
            return {k: clone(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clone(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(clone(v) for v in obj)
        if obj is None or isinstance(obj, (int, float, bool, str)):
            return obj
        raise RuntimeError(f"deep_copy_cache: unhandled state member {type(obj)!r}")

    if hasattr(cache, "layers"):
        new = _copy.copy(cache)
        new.layers = []
        for layer in cache.layers:
            nl = _copy.copy(layer)
            nl.state = clone(layer.state)
            nl._seen_tokens = int(getattr(layer, "_seen_tokens", 0))
            new.layers.append(nl)
        if hasattr(cache, "_seen_tokens"):
            new._seen_tokens = int(cache._seen_tokens)
        return new
    if hasattr(cache, "states"):
        new = _copy.copy(cache)
        new.states = [clone(s) for s in cache.states]
        if hasattr(cache, "_seen_tokens"):
            new._seen_tokens = int(cache._seen_tokens)
        return new
    raise RuntimeError(f"unknown cache type {type(cache)!r}")


def tensor_sig(t: Optional[torch.Tensor]) -> Optional[dict]:
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        raise RuntimeError(f"tensor_sig: expected Tensor or None, got {type(t)!r}")
    b = t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "md5": hashlib.md5(b).hexdigest(),
        "norm": float(t.detach().float().norm().item()),
    }


def cache_fingerprint(cache, layers=(0, 15, 31)) -> dict:
    """Per-field shape/dtype/checksum plus counters, for the recorded cache policy."""
    fp: dict[str, Any] = {"n_layers": n_cache_layers(cache)}
    fp["cache_seen_tokens"] = int(getattr(cache, "_seen_tokens", -1))
    for l in layers:
        if l >= fp["n_layers"]:
            continue
        st = cache_layer_state(cache, l)
        if st is None:
            raise RuntimeError(f"layer {l} state is None -- prefix forward did not populate the cache")
        entry: dict[str, Any] = {}
        for k in ("recurrent_state", "conv_state", "ffn_state", "attn_state"):
            v = st.get(k)
            if v is None:
                entry[k] = None
            elif isinstance(v, torch.Tensor):
                entry[k] = tensor_sig(v)
            elif isinstance(v, (tuple, list)):
                entry[k] = [tensor_sig(x) if isinstance(x, torch.Tensor) else str(type(x)) for x in v]
            else:
                raise RuntimeError(f"layer {l} field {k}: unhandled type {type(v)!r}")
        if hasattr(cache, "layers"):
            entry["layer_seen_tokens"] = int(getattr(cache.layers[l], "_seen_tokens", -1))
        fp[f"layer{l}"] = entry
    return fp


def cache_policy_hash(cache) -> str:
    return hashlib.sha256(
        json.dumps(cache_fingerprint(cache), sort_keys=True).encode()
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# latent sampling: byte-identical to the shipped evaluator's sampler
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_ddim_cfg_seeded(model, cond, steps, cfg_scale, device, dtype, seed: int):
    """sample_ddim_cfg with an explicit seed, so A3 and A4 provably share noise.

    Body is copied from scripts/eval/sample_prefix_suffix_cfg.sample_ddim_cfg; the only
    change is torch.manual_seed(seed) immediately before the randn, matching what the
    shipped evaluator does at its call site (torch.manual_seed(seed + 100000*task_i + i)).
    """
    torch.manual_seed(seed)
    z = torch.randn(cond.shape[0], model.latent_dim, device=device, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    for i in range(steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        t_batch = t_cur.expand(cond.shape[0])
        eps_cond = model.latent_dit(z, t_batch, cond=cond)
        if cfg_scale == 1.0:
            eps = eps_cond
        else:
            eps_uncond = model.latent_dit(z, t_batch, cond=uncond)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        z0_pred = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        z = ab_nxt.sqrt() * z0_pred + (1 - ab_nxt).sqrt() * eps
    return z


# ---------------------------------------------------------------------------
# the one decode loop, shared by all five arms
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode(model, tokenizer, input_ids, cache, logits, args) -> tuple[list[int], list[float]]:
    """Greedy/sampled decode from a pre-populated cache. Identical for every arm.

    Returns (generated_ids, first_token_logit_stats).  The first answer token is
    recorded, NOT dropped: the training CE helper drops the first suffix token, which
    is a property of teacher forcing, not a licence to drop it in evaluation.
    """
    generated_ids: list[int] = []
    all_ids = list(input_ids[0].tolist())
    eos_id = getattr(tokenizer, "eos_token_id", None)
    first_logits: list[float] = []
    for step in range(args.max_new_tokens):
        if step == 0:
            lg = logits.detach().float()
            top = torch.topk(lg, 5)
            first_logits = [
                float(lg.max().item()),
                float(lg.mean().item()),
                float(lg.std().item()),
                float(top.values[0].item()),
                float(top.indices[0].item()),
                float(lg.norm().item()),
            ]
        pen = apply_repetition_penalty(logits.float(), all_ids, args.repetition_penalty)
        if args.temperature <= 0:
            next_id = int(torch.argmax(pen).item())
        else:
            probs = torch.softmax(pen / args.temperature, dim=-1)
            if args.top_k > 0:
                tv, ti = torch.topk(probs, args.top_k)
                probs = torch.zeros_like(probs).scatter(-1, ti, tv)
                probs = probs / probs.sum().clamp(min=1e-12)
            probs = apply_top_p(probs, args.top_p)
            next_id = int(torch.multinomial(probs, 1).item())
        if eos_id is not None and next_id == eos_id:
            break
        generated_ids.append(next_id)
        all_ids.append(next_id)
        out = model.rwkv_model(
            input_ids=torch.tensor([[next_id]], device=input_ids.device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logits = out.logits[0, -1]
    return generated_ids, first_logits


@torch.no_grad()
def run_arm(model, tokenizer, arm: str, input_ids, attention_mask, prefix_cache, prefix_logits,
            z_write, args) -> dict:
    """Execute one arm on one example. prefix_cache is ALREADY a private deep copy."""
    t0 = time.time()
    cache = prefix_cache
    logits = prefix_logits
    latent_norm = None
    adapter_called = False
    restored_layers = 0

    if arm == "A0_no_adapter":
        # adapter NOT called, principal memory not written. Nothing touches the cache.
        pass
    else:
        if arm == "A1_identity":
            # The adapter is called with the CORRECT latent so the auxiliary-field and
            # counter side effects are exactly those of A4; only the principal memory is
            # then put back.  (Using a different latent here would confound A1 with A3.)
            z_eff = z_write
        else:
            z_eff = z_write
        if z_eff is None:
            raise RuntimeError(f"arm {arm}: z_write is None")
        latent_norm = float(z_eff.detach().float().norm().item())
        # snapshot the ORIGINAL principal memories before the adapter runs
        originals = None
        if arm == "A1_identity":
            originals = []
            for l in range(n_cache_layers(cache)):
                st = cache_layer_state(cache, l)
                rs = st.get("recurrent_state")
                if rs is None:
                    raise RuntimeError(f"A1: layer {l} recurrent_state is None before inject")
                originals.append(rs.detach().clone())
        states = model.predict_states(z_eff)
        adapter_called = True
        cache = model.inject_into_cache(cache, states)
        if arm == "A1_identity":
            for l in range(n_cache_layers(cache)):
                st = cache_layer_state(cache, l)
                st["recurrent_state"] = originals[l].detach().clone()
                if not torch.equal(st["recurrent_state"], originals[l]):
                    raise RuntimeError(f"A1: bit-exact restore FAILED at layer {l}")
                restored_layers += 1
            if restored_layers != n_cache_layers(cache):
                raise RuntimeError(
                    f"A1: restored {restored_layers} of {n_cache_layers(cache)} layers")

    policy_hash = cache_policy_hash(cache)

    # optional prefix replay -- applied IDENTICALLY to every arm, default OFF
    replay_tokens = 0
    if args.prefix_replay:
        out = model.rwkv_model(input_ids=input_ids, past_key_values=cache,
                              use_cache=True, return_dict=True)
        cache = out.past_key_values
        logits = out.logits[0, -1]
        replay_tokens = int(input_ids.shape[1])

    plan_s = time.time() - t0
    t1 = time.time()
    gen_ids, first_logits = decode(model, tokenizer, input_ids, cache, logits, args)
    dec_s = time.time() - t1
    return {
        "generated_ids": gen_ids,
        "prediction": tokenizer.decode(gen_ids, skip_special_tokens=True).strip(),
        "new_tokens": len(gen_ids),
        "latent_norm": latent_norm,
        "adapter_called": adapter_called,
        "a1_restored_layers": restored_layers,
        "cache_policy_hash": policy_hash,
        "first_token_logits": first_logits,
        "prefix_replay_tokens": replay_tokens,
        "planning_seconds": round(plan_s, 4),
        "decode_seconds": round(dec_s, 4),
    }


# ---------------------------------------------------------------------------
def derangement(n: int, seed: int) -> list[int]:
    """Shuffle then rotate by one: guarantees no fixed point for n >= 2."""
    if n < 2:
        raise RuntimeError(f"derangement needs n>=2, got {n}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    rot = perm[1:] + perm[:1]
    donor = [0] * n
    for recipient, d in zip(perm, rot):
        donor[recipient] = d
    for i, d in enumerate(donor):
        if i == d:
            raise RuntimeError(f"derangement has a fixed point at {i}")
    if sorted(donor) != list(range(n)):
        raise RuntimeError("derangement is not a permutation")
    return donor


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--source_dir", required=True,
                   help="already-produced result jsonl dir; supplies prompts + identical example IDs")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mean_latent", default="", help="G1 g1_mean_latent.pt; required for A2")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--expect_rows", type=int, default=200,
                   help="assert each source task file has exactly this many rows (RACE trap)")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prefix_replay", action="store_true",
                   help="OFF by default = training-consistent (training never re-feeds the prefix)")
    p.add_argument("--tag", default="main")
    p.add_argument("--sanity_only", action="store_true")
    p.add_argument("--sanity_n", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise RuntimeError(f"unknown arm {a!r}; known: {ARMS}")

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    ckpt_path = Path(args.ckpt_dir) / "model.pt"
    print(f"[provenance] hashing {ckpt_path}", flush=True)
    ck_sha = sha256_file(ckpt_path)
    git_commit = os.environ.get("G2_GIT_COMMIT", "unknown")
    print(f"[provenance] checkpoint_sha256={ck_sha}", flush=True)
    print(f"[provenance] git_commit={git_commit}", flush=True)
    print(f"[protocol] prefix_replay={args.prefix_replay} (default False = training-consistent)",
          flush=True)

    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model_any: Any = model
    cfg_any: Any = cfg
    model_any._prefix_suffix_s2 = True
    model_any._training_stage = 2
    model_any._cfg_drop_prob = float(cfg_any.training.get("cfg_drop_prob", 0.0))

    # ---- effective write strength: assert A2/A3/A4 share it, and print it -------------
    state_scale = float(model.state_scale.detach().float().item())
    use_learnable_blend = bool(getattr(model, "use_learnable_blend", False))
    blend_gate = getattr(model, "blend_gate_logit", None)
    writer = str(getattr(model, "s1_writer_type", "?"))
    denoiser = str(getattr(model, "global_denoiser_type", "?"))
    zero_rnn = os.environ.get("RELAY_INJECT_ZERO_RNN_CACHE", "0")
    print(f"[write-strength] state_scale={state_scale!r}  writer={writer}  denoiser={denoiser}",
          flush=True)
    print(f"[write-strength] use_learnable_blend={use_learnable_blend} "
          f"blend_gate_logit={'present' if blend_gate is not None else 'absent'}", flush=True)
    print(f"[write-strength] RELAY_INJECT_ZERO_RNN_CACHE={zero_rnn} "
          f"(is_rnn_denoiser={denoiser in ('rwkv','birwkv')})", flush=True)
    print("[write-strength] inject_into_cache takes NO lambda/blend argument: the ONLY "
          "write-strength knob on this path is the learned scalar state_scale, applied "
          "inside predict_states. A2/A3/A4 all go through predict_states -> "
          "inject_into_cache, so their effective write strength is identical by "
          "construction.", flush=True)
    effective_lambda = 1.0  # inject_into_cache is a full replacement of recurrent_state

    # ---- data: read prompts from already-produced result jsonl ------------------------
    src = Path(args.source_dir)
    examples: dict[str, list[dict]] = {}
    for t in tasks:
        f = src / f"{t}.jsonl"
        if not f.exists():
            raise RuntimeError(f"source jsonl missing: {f}")
        rows = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
        print(f"[data] {t}: {len(rows)} rows in {f}", flush=True)
        if len(rows) != args.expect_rows:
            raise RuntimeError(
                f"{t}: source has {len(rows)} rows, protocol requires exactly "
                f"{args.expect_rows}. RACE ships in both 100- and 200-row versions; "
                f"pin a 200-row source or the macro denominator is wrong.")
        rows = rows[: args.max_samples]
        for r in rows:
            if not r.get("choices"):
                raise RuntimeError(f"{t}: row id={r.get('id')} has no choices")
            if "prompt" not in r or not r["prompt"]:
                raise RuntimeError(f"{t}: row id={r.get('id')} has no prompt")
        examples[t] = rows

    if args.sanity_only:
        for t in tasks:
            examples[t] = examples[t][: args.sanity_n]

    # ---- A2's mean latent ------------------------------------------------------------
    mean_latent = None
    a2_blocked_reason = None
    if "A2_mean_latent" in arms:
        mp = Path(args.mean_latent) if args.mean_latent else None
        if mp is None or not mp.exists():
            a2_blocked_reason = f"G1 mean-latent tensor absent ({args.mean_latent or '<unset>'})"
            print(f"[A2] BLOCKED: {a2_blocked_reason}. Running the other arms; NOT "
                  f"substituting a zero vector (a zero vector is a different control).",
                  flush=True)
            arms = [a for a in arms if a != "A2_mean_latent"]
        else:
            obj = torch.load(mp, map_location="cpu", weights_only=False)
            if isinstance(obj, dict):
                for k in ("mean_latent", "mean", "z_mean", "tensor"):
                    if k in obj and isinstance(obj[k], torch.Tensor):
                        obj = obj[k]
                        break
            if not isinstance(obj, torch.Tensor):
                raise RuntimeError(f"G1 file {mp} does not hold a Tensor: {type(obj)!r}")
            v = obj.reshape(-1)
            if v.numel() != int(model.latent_dim):
                raise RuntimeError(
                    f"G1 mean latent has {v.numel()} elements, model.latent_dim="
                    f"{model.latent_dim}")
            if float(v.float().norm().item()) == 0.0:
                raise RuntimeError("G1 mean latent is all zeros -- refusing: a zero vector "
                                   "is a different control than the training mean")
            mean_latent = v.reshape(1, -1).to(args.device, dtype)
            print(f"[A2] loaded G1 mean latent: shape={list(mean_latent.shape)} "
                  f"norm={float(mean_latent.float().norm().item()):.6f} "
                  f"sha256={sha256_file(mp)[:16]}", flush=True)

    # =============================== pass 1: cache z ==================================
    # Sample every prompt's latent ONCE and persist, so A3 and A4 provably use the same
    # samples with the same noise seeds; only the assignment differs.
    z_path = out_dir / f"z_cache_seed{args.seed}.pt"
    z_cache: dict[str, dict[int, torch.Tensor]] = {}
    need_z = any(a in arms for a in ("A1_identity", "A3_shuffled", "A4_correct"))
    if need_z:
        if z_path.exists():
            blob = torch.load(z_path, map_location="cpu", weights_only=False)
            z_cache = {t: {int(k): v for k, v in d.items()} for t, d in blob["z"].items()}
            print(f"[z-cache] reloaded {z_path}", flush=True)
        else:
            for task_i, t in enumerate(tasks):
                z_cache[t] = {}
                for i, r in enumerate(examples[t]):
                    ids = tokenizer(r["prompt"], return_tensors="pt").input_ids.to(args.device)
                    am = torch.ones_like(ids)
                    cond = encode_prefix(model, ids, am)
                    seed_i = args.seed + 100000 * task_i + i
                    z = sample_ddim_cfg_seeded(model, cond, args.steps, args.cfg_scale,
                                               args.device, dtype, seed_i)
                    z_cache[t][int(r.get("id", i))] = z.detach().cpu()
                    if (i + 1) % 25 == 0:
                        print(f"[z-cache] {t} {i+1}/{len(examples[t])}", flush=True)
            torch.save({"z": {t: {k: v for k, v in d.items()} for t, d in z_cache.items()},
                        "seed": args.seed, "steps": args.steps, "cfg_scale": args.cfg_scale},
                       z_path)
            print(f"[z-cache] wrote {z_path}", flush=True)

    # ---- derangement per task, with tensor-level assertions --------------------------
    donors: dict[str, dict[int, int]] = {}
    if "A3_shuffled" in arms:
        for task_i, t in enumerate(tasks):
            ids_list = [int(r.get("id", i)) for i, r in enumerate(examples[t])]
            dmap = derangement(len(ids_list), args.seed + 7919 * task_i)
            donors[t] = {ids_list[i]: ids_list[dmap[i]] for i in range(len(ids_list))}
            n_same_tensor = 0
            for rid, did in donors[t].items():
                if rid == did:
                    raise RuntimeError(f"{t}: fixed point at id {rid}")
                if torch.equal(z_cache[t][rid], z_cache[t][did]):
                    n_same_tensor += 1
            print(f"[A3] {t}: derangement over {len(ids_list)} ids, fixed points=0, "
                  f"z_donor==z_recipient as TENSORS: {n_same_tensor}", flush=True)
            if n_same_tensor > 0:
                raise RuntimeError(
                    f"{t}: {n_same_tensor} donor latents are bit-identical to their "
                    f"recipient's -- A3 would be a no-op on those rows")

    # =============================== pass 2: decode ===================================
    tag = args.tag
    writers = {a: (out_dir / f"{a}__{tag}.jsonl").open("w", encoding="utf-8") for a in arms}
    tally: dict[tuple[str, str], dict[str, int]] = {
        (a, t): {"n": 0, "correct": 0, "invalid": 0} for a in arms for t in tasks}
    liveness: list[dict] = []

    for task_i, t in enumerate(tasks):
        for i, r in enumerate(examples[t]):
            ex_id = int(r.get("id", i))
            prompt = r["prompt"]
            gold = str(r.get("ground_truth", ""))
            choices = list(r.get("choices") or [])
            p_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
            am = torch.ones_like(ids)
            # ONE prefix forward, shared as the source for every arm's private deep copy
            out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                  use_cache=True, return_dict=True)
            base_cache = out.past_key_values
            base_logits = out.logits[0, -1].detach().clone()
            base_fp = cache_fingerprint(base_cache)

            per_arm_gen: dict[str, str] = {}
            for a in arms:
                if a == "A0_no_adapter":
                    z_write, donor_id = None, None
                elif a == "A1_identity":
                    z_write = z_cache[t][ex_id].to(args.device, dtype)
                    donor_id = None
                elif a == "A2_mean_latent":
                    z_write, donor_id = mean_latent, None
                elif a == "A3_shuffled":
                    donor_id = donors[t][ex_id]
                    z_write = z_cache[t][donor_id].to(args.device, dtype)
                elif a == "A4_correct":
                    z_write = z_cache[t][ex_id].to(args.device, dtype)
                    donor_id = None
                else:
                    raise RuntimeError(f"unhandled arm {a}")

                priv = deep_copy_cache(base_cache)
                # no two arms may share a mutable cache object
                if priv is base_cache:
                    raise RuntimeError("deep_copy_cache returned the same object")
                res = run_arm(model, tokenizer, a, ids, am, priv,
                              base_logits.clone(), z_write, args)
                correct, invalid = score_one(t, res["prediction"], gold, choices)
                rec = {
                    "arm": a, "task": t, "example_id": ex_id, "seed": args.seed,
                    "checkpoint_sha256": ck_sha, "git_commit": git_commit,
                    "donor_id": donor_id, "prompt_hash": p_hash,
                    "generated_ids": res["generated_ids"], "prediction": res["prediction"],
                    "gold": gold, "choices": choices,
                    "correct": correct, "invalid": invalid,
                    "new_tokens": res["new_tokens"], "latent_norm": res["latent_norm"],
                    "effective_lambda": effective_lambda, "state_scale": state_scale,
                    "adapter_name": f"predict_states[{writer}]+inject_into_cache"
                                    if res["adapter_called"] else "none",
                    "adapter_called": res["adapter_called"],
                    "a1_restored_layers": res["a1_restored_layers"],
                    "prefix_replay_tokens": res["prefix_replay_tokens"],
                    "prefix_encoding_rule": (
                        "forward(prompt) -> write -> forward(prompt) AGAIN -> decode"
                        if args.prefix_replay else
                        "forward(prompt) -> write -> decode (no re-feed; training-consistent)"),
                    "prefix_replay": bool(args.prefix_replay),
                    "sampler": f"ddim_cfg(steps={args.steps},cfg={args.cfg_scale},"
                               f"gen_type={getattr(model,'_gen_type','?')})",
                    "decoding": f"greedy(temp={args.temperature})" if args.temperature <= 0
                                else f"sample(temp={args.temperature},k={args.top_k},p={args.top_p})",
                    "repetition_penalty": args.repetition_penalty,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_tokens": int(ids.shape[1]),
                    "cache_policy_hash": res["cache_policy_hash"],
                    "first_token_logits": res["first_token_logits"],
                    "planning_seconds": res["planning_seconds"],
                    "decode_seconds": res["decode_seconds"],
                }
                writers[a].write(json.dumps(rec, ensure_ascii=False) + "\n")
                writers[a].flush()
                k = tally[(a, t)]
                k["n"] += 1
                k["correct"] += int(correct)
                k["invalid"] += int(invalid)
                per_arm_gen[a] = res["prediction"]

            if i < 3:
                uniq = len(set(per_arm_gen.values()))
                liveness.append({
                    "task": t, "example_id": ex_id, "n_arms": len(arms),
                    "n_distinct_generations": uniq,
                    "all_identical": uniq == 1,
                    "A4_vs_A0_differ": (per_arm_gen.get("A4_correct")
                                        != per_arm_gen.get("A0_no_adapter"))
                    if ("A4_correct" in per_arm_gen and "A0_no_adapter" in per_arm_gen) else None,
                    "generations": per_arm_gen,
                    "base_cache_seen_tokens": base_fp["cache_seen_tokens"],
                })
                print(f"[liveness] {t} id={ex_id}: {uniq}/{len(arms)} distinct generations "
                      f"across arms", flush=True)
            if (i + 1) % 20 == 0:
                print(f"[decode] {t} {i+1}/{len(examples[t])}", flush=True)

    for w in writers.values():
        w.close()

    # =============================== aggregate ========================================
    agg: dict[str, Any] = {
        "tag": tag, "seed": args.seed, "checkpoint_sha256": ck_sha,
        "ckpt_dir": args.ckpt_dir, "git_commit": git_commit,
        "source_dir": str(src), "tasks": tasks, "arms": arms,
        "prefix_replay": bool(args.prefix_replay),
        "prefix_encoding_rule": ("refeed" if args.prefix_replay else "no_refeed_training_consistent"),
        "state_scale": state_scale, "effective_lambda": effective_lambda,
        "use_learnable_blend": use_learnable_blend,
        "writer": writer, "denoiser": denoiser,
        "steps": args.steps, "cfg_scale": args.cfg_scale,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "scorer": "scripts/eval/score_gen_match.py match_choice rule set (LETTER_FIRST off)",
        "invalid_policy": "invalid counts as WRONG; never dropped; denominator never shrinks",
        "a2_blocked_reason": a2_blocked_reason,
        "liveness_samples": liveness,
        "rows": {},
    }
    print("\n=== G2 five-arm table (accuracy %, invalid in brackets) ===", flush=True)
    hdr = f"{'arm':18s}" + "".join(f"{t[:12]:>14s}" for t in tasks) + f"{'Macro3':>10s}"
    print(hdr, flush=True)
    for a in arms:
        accs = []
        row: dict[str, Any] = {}
        line = f"{a:18s}"
        for t in tasks:
            k = tally[(a, t)]
            if k["n"] == 0:
                raise RuntimeError(f"{a}/{t}: denominator is 0 -- nothing was evaluated")
            acc = 100.0 * k["correct"] / k["n"]
            accs.append(acc)
            row[t] = {"acc": round(acc, 2), "n": k["n"], "correct": k["correct"],
                      "invalid": k["invalid"]}
            line += f"{acc:9.2f}[{k['invalid']:>3d}]"
        macro = sum(accs) / len(accs)
        row["macro3"] = round(macro, 2)
        row["invalid_total"] = sum(tally[(a, t)]["invalid"] for t in tasks)
        agg["rows"][a] = row
        line += f"{macro:10.2f}"
        print(line, flush=True)
    print("(Macro3 = equal-weight mean of the three task accuracies. NOT Avg_11.)", flush=True)

    ap = out_dir / f"aggregate__{tag}.json"
    with ap.open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"\n[done] aggregate -> {ap}", flush=True)
    if a2_blocked_reason:
        print(f"[A2] STILL BLOCKED: {a2_blocked_reason}", flush=True)


if __name__ == "__main__":
    main()

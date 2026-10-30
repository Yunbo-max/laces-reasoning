#!/usr/bin/env python3
"""D3: five-arm matched latent-use controls on ONE checkpoint (GPU job).

Fills tab:latent-use-pending in paper_ICLR27/lalaland/appendix.tex.

    A0  same decode loop, adapter NOT called, principal memory not written
    A1  same adapter called, ORIGINAL principal memory restored bit-exactly
    A2  write R(mean_train_latent)
    A3  write the sampled latent of a DIFFERENT prompt (within-task derangement)
    A4  write the sampled latent of the correct prompt (full method)

This file is written and statically validated on CPU; it needs a GPU to run.

-----------------------------------------------------------------------------
WHY A0 IS NOT "lambda = 0"
-----------------------------------------------------------------------------
``blend_into_cache(cache, states, 0.0)`` (models/state_hijacking_dit.py, the
``layer_blend <= 0.0`` branch) does keep the principal ``recurrent_state``, but
in the SAME loop it then does, unconditionally and for every layer:

    for sub_key in ("conv_state", "ffn_state"):
        state[sub_key] = torch.zeros_like(state[sub_key])
    _reset_cache_layer_seen_tokens(cache, l)          # layer._seen_tokens = 0
  ...and after the loop:
    cache._seen_tokens = 0                            # whole-cache counter

So lambda=0 is a DIFFERENT intervention: principal memory preserved, auxiliary
conv/FFN buffers destroyed, and both the per-layer and whole-cache token
counters reset to zero.  A0 must be "the adapter is never called", i.e. the
cache object is handed to the decode loop untouched.

``inject_into_cache`` (the other adapter) overwrites ``recurrent_state``, and
because this checkpoint has NO ``global_denoiser_type`` key, ``is_rnn_denoiser``
is False and it ALSO zeroes conv_state/ffn_state -- but it does NOT touch either
counter.  Hence the counter table this runner records per arm:

    arm  recurrent_state      conv/ffn_state   layer._seen_tokens  cache._seen_tokens
    A0   untouched            untouched        untouched           untouched
    A1   written then RESTORED  zeroed (kept)  untouched           untouched
    A2-4 written (R(z))      zeroed           untouched           untouched
    (lambda=0 blend, NOT USED: kept / zeroed / RESET 0 / RESET 0)

A1 therefore differs from A0 in exactly the conv/ffn zeroing, which is the
auxiliary-cache effect the appendix wants isolated, and differs from A2/A3/A4 in
exactly the principal-memory content.

-----------------------------------------------------------------------------
PREFIX REPLAY
-----------------------------------------------------------------------------
The pre-existing pair was NOT comparable:
  generate_answer     (run_cola_dlm_tasks_prefix_suffix_cfg.py): forward(prompt)
                      -> inject -> forward(prompt) AGAIN -> decode
  generate_raw_answer (same file):                              forward(prompt) -> decode
so raw-vs-method conflated the latent with an extra ~512-token state evolution.
Training does NOT re-feed: ``_singlez_coadapt_ce_from_z`` prefills the prefix,
injects, then forwards the SUFFIX only.  ``--prefix_replay`` is therefore an
explicit flag applied IDENTICALLY to all five arms and recorded in every row.
Default ``none`` = training-consistent.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, cast

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from models.state_hijacking_dit import _cache_layer_state  # noqa: E402
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.sample_prefix_suffix_cfg import (  # noqa: E402
    apply_repetition_penalty, encode_prefix, sample_ddim_cfg,
)

ARMS_ALL = ("A0", "A1", "A2", "A3", "A4")
ARMS_NEED_Z_SAMPLE = ("A3", "A4")   # these two must share the SAME sampled z
ARMS_WRITE = ("A2", "A3", "A4")     # these three must share effective lambda/gamma
LETTERS = "ABCDEFGH"
# `x in LETTERS` is a SUBSTRING test: "" and "AB" both pass it, and
# LETTERS.index() returns 0 for both.  Membership must go through this set.
LETTER_SET = frozenset(LETTERS)


# ---------------------------------------------------------------- args --------

def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--examples", default=str(Path(__file__).with_name("examples_fixed.json")),
                   help="D2 output; pins ids + prompt hashes + gold + choices")
    p.add_argument("--provenance", default=str(Path(__file__).with_name("provenance.json")),
                   help="D1 output; supplies checkpoint_sha256 and git_commit for every row")
    p.add_argument("--out", required=True, help="NEW output dir; must not exist")
    p.add_argument("--arms", default=",".join(ARMS_ALL))
    p.add_argument("--tasks", default="sciq,commonsenseqa,race")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed_stream", choices=["additive", "multiplicative"], default="additive",
                   help="how the per-example DDIM seed is derived. "
                        "additive = seed + 100003*(task+1) + i, the ORIGINAL form: it is "
                        "collision-free WITHIN a seed (600/600 distinct) but seed s example i "
                        "and seed s+1 example i-1 get the same integer, so consecutive seeds "
                        "reuse 597/600 of the same noise draws at relabelled positions and "
                        "run-to-run spread is UNDERSTATED. multiplicative = seed*1000003 + "
                        "100003*(task+1) + i, which shares 0/600 integers between any two of "
                        "seeds 42/43/44. This is recorded per row; the two are NOT comparable, "
                        "so do not mix them within one reported table.")
    p.add_argument("--prefix_replay", choices=["none", "refeed"], default="none",
                   help="none = training-consistent (prefill prefix, write, decode). "
                        "refeed = the legacy generate_answer behaviour (re-forward the "
                        "whole prompt after writing). Applied identically to all arms.")
    p.add_argument("--mean_latent", default=None,
                   help="torch .pt holding the training-set mean z, REQUIRED for A2")
    p.add_argument("--z_cache", default=None,
                   help="torch .pt of sampled z per (task,id). Loaded if present, else "
                        "written. A3 and A4 both read it, so they provably share samples.")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="debug only: first N per task")
    p.add_argument("--letter_first", action="store_true",
                   help="score a bare option label as that option (SCORE_LETTER_FIRST=1 "
                        "equivalent). Applied identically to all arms or to none.")
    return p.parse_args(argv)


# ------------------------------------------------------- cache utilities ------

def cache_fingerprint(cache, n_layers: Optional[int] = None) -> dict[str, Any]:
    """Per-field shape / dtype / checksum / counter snapshot of a decoder cache.

    Raises on an unrecognised field type: an unknown shape is a reason to stop,
    not a reason to skip a field and report a shorter, cleaner-looking table.
    """
    if n_layers is None:
        n_layers = len(cache.layers) if hasattr(cache, "layers") else len(cache.states)
    fields: dict[str, Any] = {}
    for l in range(n_layers):
        st = _cache_layer_state(cache, l)
        for key, val in sorted(st.items()):
            name = f"L{l}.{key}"
            if val is None:
                fields[name] = {"kind": "none"}
            elif isinstance(val, torch.Tensor):
                v = val.detach().to(torch.float64).cpu()
                fields[name] = {
                    "kind": "tensor",
                    "shape": list(val.shape),
                    "dtype": str(val.dtype),
                    "sum": float(v.sum()),
                    "absmax": float(v.abs().max()) if v.numel() else 0.0,
                    "md5": hashlib.md5(val.detach().cpu().contiguous()
                                       .view(torch.uint8).numpy().tobytes()).hexdigest(),
                }
            else:
                raise TypeError(
                    f"cache field {name} has unhandled type {type(val)!r}; refusing to "
                    "fingerprint a cache whose shape I do not understand"
                )
        layer_obj = cache.layers[l] if hasattr(cache, "layers") else None
        fields[f"L{l}._seen_tokens"] = (
            getattr(layer_obj, "_seen_tokens", None) if layer_obj is not None else None
        )
    return {
        "n_layers": n_layers,
        "cache_seen_tokens": getattr(cache, "_seen_tokens", None),
        "fields": fields,
    }


def principal_memory(cache, n_layers: int) -> list[Optional[torch.Tensor]]:
    """Clone every layer's recurrent_state (the *principal* memory)."""
    out = []
    for l in range(n_layers):
        rs = _cache_layer_state(cache, l).get("recurrent_state")
        out.append(rs.detach().clone() if isinstance(rs, torch.Tensor) else None)
    return out


def restore_principal_memory(cache, saved: list[Optional[torch.Tensor]]) -> int:
    """Put the saved recurrent_state tensors back, BIT-EXACTLY.

    Everything the adapter did to auxiliary fields (conv_state / ffn_state) and
    to the counters is intentionally LEFT IN PLACE: that difference from A0 is
    what A1 is for.
    """
    n = 0
    for l, sv in enumerate(saved):
        st = _cache_layer_state(cache, l)
        if sv is None:
            continue
        st["recurrent_state"] = sv.clone()
        cur = st["recurrent_state"]
        if not torch.equal(cur, sv):
            raise AssertionError(f"A1 restore is not bit-exact at layer {l}")
        n += 1
    return n


def cache_policy_hash(prefix_replay: str, arm: str, adapter_name: str,
                      zero_rnn_env: str, deepcopy: bool) -> str:
    """Hash of the cache HANDLING POLICY (not of the cache contents).

    Recorded per row so two rows that claim to be the same arm can be shown to
    have been produced under the same policy.  Content is spelled out in the
    row's ``cache_policy`` field as well -- a hash alone is unreadable, and a
    reader must be able to tell WHICH policy without reversing a digest.
    """
    payload = json.dumps({
        "prefix_replay": prefix_replay,
        "arm": arm,
        "adapter": adapter_name,
        "RELAY_INJECT_ZERO_RNN_CACHE": zero_rnn_env,
        "deepcopy_before_mutation": deepcopy,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ------------------------------------------------------- answer parsing -------
# ONE parser, used by all five arms.  Lifted from scripts/eval/score_gen_match.py
# so the five new rows are scored by the same rule as the reported table, but
# reimplemented locally because that module imports acc_calc from the DiffRwkv
# repo, which is not a dependency of this runner.

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    t = str(text).lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def truncate_at_newline(gen: str) -> str:
    if not isinstance(gen, str):
        return ""
    i = gen.find("\n")
    return gen[:i] if i >= 0 else gen


def letter_of(text: str) -> Optional[str]:
    """Return a SINGLE option letter, or None.

    Every return path must be exactly one character.  The original version had
    two substring bugs, both rooted in ``x in LETTERS`` being a SUBSTRING test on
    the string "ABCDEFGH" rather than a membership test on a set of letters:

      * ``letter_of("")`` returned ``""`` because ``"" in "ABCDEFGH"`` is True and
        ``"ABCDEFGH".index("") == 0``.  An EMPTY generation was therefore scored
        as choice A with ``invalid=False``.  On the pinned 200s that is 49/200
        SciQ, 34/200 CQA, 37/200 RACE correct -- macro 20.0% for an arm that
        emitted nothing, and zero rows flagged invalid.  It also swallowed every
        generation that STARTS with a newline, since truncate_at_newline runs
        first: "\\nB) something" truncates to "" and was scored as A.
      * ``letter_of("AB")`` returned the two-character string ``"AB"``, and
        ``LETTERS.index("AB") == 0`` for the same reason, so "AB" also became A.

    The protocol line that breaks is "invalid answers count as WRONG, never
    dropped": a wrong answer was being converted into a 1-in-|choices| lottery
    ticket instead.
    """
    t = text.strip()
    if not t:
        return None
    m = re.match(r"^[\(\[]?([A-Ha-h])[\)\].:,]\s*", t)
    if m:
        return m.group(1).upper()
    if len(t) == 1 and t.upper() in LETTER_SET:
        return t.upper()
    if len(t) == 2 and t[0].upper() in LETTER_SET and not t[1].isalnum():
        return t[0].upper()
    m = re.match(r"^[\(\[]?([A-Ha-h])(?=[A-Z])", t)   # bare label glued: "BQuestion:"
    if m:
        return m.group(1).upper()
    return None


def letter_index(letter: Optional[str]) -> Optional[int]:
    """Index of a single option letter, or None.  Never uses str.index, which
    silently returns 0 for the empty string and for any substring like "AB"."""
    if letter is None or len(letter) != 1 or letter not in LETTER_SET:
        return None
    return LETTERS.index(letter)


def example_seed(seed: int, stream: str, task_index: int, i: int) -> int:
    """Per-example DDIM seed.

    ``additive`` is the original form and is kept as the default so already-running
    jobs stay reproducible, but it is NOT independent across seeds: the map
    ``s + 100003*(t+1) + i`` means (s, i) and (s+1, i-1) collide, so seeds 42/43/44
    share 597/600, 597/600 and 594/600 of their integers -- the same noise draws at
    relabelled positions.  Within one seed both forms are collision-free (600/600
    distinct).  ``multiplicative`` shares 0/600 between any pair of those seeds.
    """
    if stream == "additive":
        return seed + 100003 * (task_index + 1) + i
    if stream == "multiplicative":
        return seed * 1_000_003 + 100003 * (task_index + 1) + i
    raise ValueError(f"unknown seed_stream {stream!r}")


def strip_label(text: str) -> str:
    return re.sub(r"^\s*(?:answer\s*[:\-]\s*)?[\(\[]?[A-Ha-h][\)\].:,]\s*",
                  "", text, count=1, flags=re.IGNORECASE).strip()


def strip_answer_lead(text: str) -> str:
    return re.sub(r"^\s*(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)?\s*",
                  "", text, count=1, flags=re.IGNORECASE).strip()


def _f1(pred: str, gt: str) -> float:
    from collections import Counter
    p, g = normalize_text(pred).split(), normalize_text(gt).split()
    if not p or not g:
        return 0.0
    ns = sum((Counter(p) & Counter(g)).values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def match_choice(gen: str, choices: list[str]) -> Optional[int]:
    if not choices:
        return None
    ng = normalize_text(gen)
    if not ng:
        return None
    norms = [normalize_text(c) for c in choices]
    for i, c in enumerate(norms):
        if c and c == ng:
            return i
    best, best_len = None, 0
    for i, c in enumerate(norms):
        if c and (ng.startswith(c) or c.startswith(ng)):
            L = min(len(c), len(ng))
            if L > best_len:
                best, best_len = i, L
    if best is not None:
        return best
    best, best_len = None, 0                       # containment, longest wins
    for i, c in enumerate(norms):
        if c and c in ng and len(c) > best_len:
            best, best_len = i, len(c)
    if best is not None:
        return best
    scored = sorted(((_f1(gen, c), i) for i, c in enumerate(choices)), reverse=True)
    return scored[0][1] if scored and scored[0][0] >= 0.5 else None


def score_row(gen: str, gold: str, choices: list[str], letter_first: bool) -> tuple[bool, bool, Optional[int]]:
    """Return (correct, invalid, pred_index).

    INVALID ANSWERS COUNT AS WRONG.  ``invalid`` is a flag for reporting only;
    it never removes a row, so every arm's denominator is exactly the pinned 200.
    """
    g = truncate_at_newline(gen)
    pred: Optional[int] = None
    if letter_first:
        bi = letter_index(letter_of(g))
        if bi is not None and (len(strip_label(g)) == 0 or len(g.strip()) <= 2) \
                and bi < len(choices):
            pred = bi
    if pred is None:
        pred = match_choice(g, choices)
    if pred is None:
        pred = match_choice(strip_label(g), choices)
    if pred is None:
        pred = match_choice(strip_answer_lead(strip_label(g)), choices)
    if pred is None:
        li = letter_index(letter_of(g))
        if li is not None and li < len(choices):
            pred = li

    gold_idx: Optional[int] = None
    ngold = normalize_text(gold)
    for i, c in enumerate(choices):
        if normalize_text(c) == ngold:
            gold_idx = i
            break
    if gold_idx is None:
        gold_idx = match_choice(gold, choices)
    if gold_idx is None:
        raise ValueError(
            f"gold {gold!r} does not resolve to any choice {choices!r}; a row whose gold "
            "cannot be located would make the arm comparison meaningless"
        )
    invalid = pred is None
    return (pred == gold_idx and not invalid), invalid, pred


# ---------------------------------------------------------- derangement -------

def build_derangement(ids: list[Any], seed: int) -> dict[Any, Any]:
    """recipient -> donor, with NO fixed point.

    Shuffle, then rotate by one.  A rotation of a permutation of n >= 2 distinct
    elements has no fixed point by construction, and it is asserted anyway.
    """
    if len(ids) < 2:
        raise ValueError(f"cannot derange {len(ids)} example(s)")
    order = list(ids)
    random.Random(seed).shuffle(order)
    rotated = order[1:] + order[:1]
    mapping = dict(zip(order, rotated))
    fixed = [k for k, v in mapping.items() if k == v]
    if fixed:
        raise AssertionError(f"derangement has {len(fixed)} fixed point(s): {fixed[:5]}")
    if sorted(map(str, mapping.values())) != sorted(map(str, ids)):
        raise AssertionError("derangement is not a permutation of the id set")
    return mapping


# ------------------------------------------------------------- decoding -------

@torch.no_grad()
def prefill(model, input_ids, attention_mask):
    return model.rwkv_model(input_ids=input_ids, attention_mask=attention_mask.bool(),
                            use_cache=True, return_dict=True)


def prefill_span(input_ids: torch.Tensor, prefix_replay: str) -> torch.Tensor:
    """Which prompt tokens are prefilled BEFORE the write.

    ``none``   -> prompt[:-1].  The final prompt token is fed AFTER the write, so
                  the first-token logits come from the written cache while every
                  prompt token is still consumed exactly once.  Re-fed token
                  count = 0.  This is the training convention: the CE helper
                  prefills the prefix, injects, then forwards the SUFFIX, whose
                  position 0 already sees the written state.
    ``refeed`` -> the whole prompt, which is then fed a SECOND time after the
                  write (legacy ``generate_answer``).  Re-fed token count = |prompt|.
    """
    if prefix_replay == "none":
        if input_ids.shape[1] < 2:
            raise ValueError("prefix_replay=none needs a prompt of >= 2 tokens "
                             f"(got {input_ids.shape[1]})")
        return input_ids[:, :-1]
    if prefix_replay == "refeed":
        return input_ids
    raise ValueError(prefix_replay)


@torch.no_grad()
def decode_arm(model, tokenizer, input_ids, attention_mask, base_out, arm: str,
               states: Optional[list[torch.Tensor]], args) -> dict[str, Any]:
    """Run one arm on one example.

    ``base_out`` is the SHARED prefill result over ``prefill_span(...)``; its cache
    is DEEP-COPIED here so no two arms ever touch the same mutable cache object.
    The prefill span depends only on ``--prefix_replay``, never on the arm, so all
    five arms see byte-identical prefix handling.
    """
    t0 = time.time()
    cache = copy.deepcopy(base_out.past_key_values)
    n_layers = len(cache.layers) if hasattr(cache, "layers") else len(cache.states)
    fp_before = cache_fingerprint(cache, n_layers)

    adapter_name = "none"
    restored = 0
    if arm == "A0":
        # The adapter is NOT called.  Not blend_into_cache(..., 0.0): see the
        # module docstring's counter table.
        pass
    elif arm == "A1":
        if states is None:
            raise ValueError("A1 needs states: it calls the same adapter as A2-A4")
        saved = principal_memory(cache, n_layers)
        cache = model.inject_into_cache(cache, states)
        adapter_name = "inject_into_cache+restore_principal"
        restored = restore_principal_memory(cache, saved)
        if restored == 0:
            raise AssertionError(
                "A1 restored 0 layers: the check has no denominator, so it cannot fail"
            )
    elif arm in ARMS_WRITE:
        if states is None:
            raise ValueError(f"{arm} needs states")
        cache = model.inject_into_cache(cache, states)
        adapter_name = "inject_into_cache"
    else:
        raise ValueError(f"unknown arm {arm!r}")

    fp_after = cache_fingerprint(cache, n_layers)

    # Boundary input: the tokens fed through the WRITTEN cache to obtain the
    # first-token logits.  Identical across arms by construction.
    if args.prefix_replay == "none":
        boundary_ids = input_ids[:, -1:]
        replay_tokens = 0                      # no prompt token is fed twice
    else:
        boundary_ids = input_ids
        replay_tokens = int(input_ids.shape[1])
    out = model.rwkv_model(input_ids=boundary_ids, past_key_values=cache,
                           use_cache=True, return_dict=True)
    cache = out.past_key_values
    logits = out.logits[0, -1]
    boundary_list = boundary_ids[0].tolist()

    t_plan_done = time.time()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    gen_ids: list[int] = []
    all_ids = list(input_ids[0].tolist())
    first_token_logits_top: list[list[float]] = []
    for step in range(args.max_new_tokens):
        lg = apply_repetition_penalty(logits.float(), all_ids, args.repetition_penalty)
        if step == 0:
            # Record first-token logits (protocol item 7).  NOTE: the training CE
            # helper _singlez_coadapt_ce_from_z drops the first suffix token
            # (logits[:, :-1] vs suffix_tokens[:, 1:]) because it teacher-forces a
            # shifted target.  That is NOT a licence to drop the first ANSWER
            # token here: at eval the first generated token IS the answer's
            # opening token and dropping it would delete the answer.  These
            # logits come from the written cache in every arm.
            top = torch.topk(lg, k=min(10, lg.numel()))
            first_token_logits_top = [[int(i), float(v)]
                                      for v, i in zip(top.values.tolist(), top.indices.tolist())]
        nxt = int(torch.argmax(lg).item())          # greedy; identical for all arms
        if eos_id is not None and nxt == eos_id:
            break
        gen_ids.append(nxt)
        all_ids.append(nxt)
        out = model.rwkv_model(input_ids=torch.tensor([[nxt]], device=input_ids.device),
                               past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
        logits = out.logits[0, -1]
    t1 = time.time()

    return {
        "adapter_name": adapter_name,
        "a1_layers_restored": restored,
        "generated_ids": gen_ids,
        "new_tokens": len(gen_ids),
        "prediction": tokenizer.decode(gen_ids, skip_special_tokens=True).strip(),
        "first_token_logits_top10": first_token_logits_top,
        "prefix_replay_tokens": replay_tokens,
        "boundary_tokens": len(boundary_list),
        "boundary_ids_sha256": hashlib.sha256(
            json.dumps(boundary_list, separators=(",", ":")).encode()).hexdigest(),
        "prefill_tokens": int(base_out.logits.shape[1]),
        "cache_fp_before_sha256": hashlib.sha256(
            json.dumps(fp_before, sort_keys=True).encode()).hexdigest(),
        "cache_fp_after_sha256": hashlib.sha256(
            json.dumps(fp_after, sort_keys=True).encode()).hexdigest(),
        "cache_seen_tokens_before": fp_before["cache_seen_tokens"],
        "cache_seen_tokens_after": fp_after["cache_seen_tokens"],
        "planning_seconds": t_plan_done - t0,
        "decode_seconds": t1 - t_plan_done,
    }


# ----------------------------------------------------------------- main -------

def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS_ALL:
            raise SystemExit(f"unknown arm {a!r}; choose from {ARMS_ALL}")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"--out {out_dir} exists and is non-empty. This runner never writes into an "
            "existing results dir; pass a fresh path."
        )
    for guard in ("results", "outputs"):
        if guard in out_dir.parts and "latentuse" not in out_dir.parts:
            raise SystemExit(
                f"--out {out_dir} looks like a measurement dir ({guard}/...). Refusing."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed = json.loads(Path(args.examples).read_text())
    prov = json.loads(Path(args.provenance).read_text())
    ckpt_sha = prov["checkpoint"]["sha256"]
    git_commit = prov["git"]["head"]
    if prov["git"]["dirty"]:
        print(f"[WARN] laces-reasoning tree is dirty ({prov['git']['n_dirty_paths']} paths); "
              f"git_commit={git_commit} does not fully describe the code that ran.", flush=True)

    if "A2" in arms:
        if not args.mean_latent:
            raise SystemExit(
                "A2 requested but --mean_latent is missing. The training-set mean latent "
                "does not exist yet; see PLAN.md step 1. A zero vector is a DIFFERENT "
                "control, not a substitute, and this runner will not silently use one."
            )
        if not Path(args.mean_latent).is_file():
            raise SystemExit(f"--mean_latent {args.mean_latent} not found")

    mean_z: Optional[torch.Tensor] = None
    torch.manual_seed(args.seed)
    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model_any = cast(Any, model)
    cfg_any = cast(Any, cfg)
    model_any._prefix_suffix_s2 = True
    model_any._training_stage = 2
    model_any._cfg_drop_prob = float(cfg_any.training.get("cfg_drop_prob", 0.0))
    dtype = torch.bfloat16

    if args.mean_latent:
        blob = torch.load(args.mean_latent, map_location=args.device)
        mean_z = blob["mean_z"] if isinstance(blob, dict) else blob
        mean_z = mean_z.reshape(1, -1).to(args.device)
        if mean_z.shape[-1] != int(model.latent_dim):
            raise SystemExit(f"mean_latent dim {mean_z.shape[-1]} != latent_dim {model.latent_dim}")
        print(f"[A2] mean_z loaded from {args.mean_latent}, shape {tuple(mean_z.shape)}, "
              f"norm {float(mean_z.float().norm()):.4f}", flush=True)

    # Effective write strength.  inject_into_cache is a HARD write: there is no
    # lambda argument and no config field it reads, so lambda == 1.0 by
    # construction for A1..A4 and is undefined (recorded as null) for A0.
    # Guard against the failure mode the appendix flags: the independent
    # trajectory runner (scripts/eval/sample_prefix_suffix_trajectory_cfg.py,
    # apply_generated_state) calls hard injection whenever blend >= 1.0 even
    # when a saved config carries trajectory_state_blend=0.7, so a saved config
    # field does NOT identify the realized adapter.  This runner never consults
    # trajectory_state_blend; it asserts it is irrelevant here.
    if bool(cfg_any.model.get("trajectory_mode", False)):
        raise SystemExit("this checkpoint is trajectory_mode=True; the single-plan five-arm "
                         "protocol does not apply to it")
    effective_lambda = 1.0
    # state_scale is a TRAINABLE parameter, so load_relay_model casts it to
    # bfloat16 (scripts/eval/relay_utils.py:100-102), while provenance.py read it
    # straight off the fp32 checkpoint.  bf16 has 8 mantissa bits, so
    # 0.1848476082086563 lands on 0.1845703125 -- a 2.77e-04 gap that is NOT a
    # mismatch, it is the dtype of the run.  A zero-tolerance comparison here can
    # never pass; it blocked three GPU jobs before this was understood.  The
    # correct invariant is "the loaded value is the bf16 ROUNDING of the recorded
    # one", which still fires if the wrong checkpoint is loaded (any real change
    # is orders of magnitude larger than one bf16 ulp).
    state_scale = float(model.state_scale.detach().float().flatten()[0])
    prov_scale = float(prov["state_scale"])
    prov_scale_bf16 = float(torch.tensor(prov_scale, dtype=torch.float32).to(dtype))
    scale_ulp = abs(prov_scale_bf16 - prov_scale)
    print(f"[write-strength] state_scale loaded={state_scale!r} provenance={prov_scale!r} "
          f"provenance_rounded_to_{dtype}={prov_scale_bf16!r} (1 ulp = {scale_ulp:.3e})",
          flush=True)
    if state_scale != prov_scale_bf16:
        raise AssertionError(
            f"state_scale {state_scale!r} is not the {dtype} rounding of the provenance value "
            f"{prov_scale!r} (expected {prov_scale_bf16!r}). This is a DIFFERENT checkpoint, "
            f"not a rounding difference: one bf16 ulp here is {scale_ulp:.3e}."
        )
    state_scale_dtype = str(model.state_scale.dtype)
    zero_rnn_env = os.environ.get("RELAY_INJECT_ZERO_RNN_CACHE", "0")

    z_cache: dict[str, torch.Tensor] = {}
    if args.z_cache and Path(args.z_cache).is_file():
        z_cache = {k: v.to(args.device) for k, v in
                   torch.load(args.z_cache, map_location=args.device).items()}
        print(f"[z_cache] loaded {len(z_cache)} sampled latents from {args.z_cache}", flush=True)

    run_meta = {
        "argv": sys.argv,
        "args": vars(args),
        "arms": arms,
        "tasks": tasks,
        "checkpoint_sha256": ckpt_sha,
        "git_commit": git_commit,
        "git_dirty": prov["git"]["dirty"],
        "effective_lambda": effective_lambda,
        "state_scale": state_scale,
        "state_scale_dtype": state_scale_dtype,
        "state_scale_provenance_fp32": prov_scale,
        "state_scale_note": "the loaded value is the bf16 rounding of the fp32 checkpoint value; "
                            "relay_utils.load_relay_model casts trainable params to bf16.",
        "prefix_replay": args.prefix_replay,
        "seed_stream": args.seed_stream,
        "seed_stream_note": (
            "additive shares 597/600 per-example DDIM seed integers with the adjacent seed, "
            "because (s,i) and (s+1,i-1) map to the same integer, so run-to-run spread ACROSS "
            "seeds is understated; multiplicative shares 0/600. Within one seed both are "
            "collision-free (600/600 distinct)."),
        "sampler": f"ddim_cfg(steps={args.steps},cfg={args.cfg_scale})",
        "decode": f"greedy,max_new_tokens={args.max_new_tokens},rep_pen={args.repetition_penalty}",
        "RELAY_INJECT_ZERO_RNN_CACHE": zero_rnn_env,
        "counter_note": "inject_into_cache does NOT reset _seen_tokens; blend_into_cache DOES. "
                        "A0 is 'adapter not called', not lambda=0.",
    }

    totals: dict[tuple[str, str], list[int]] = {}
    for task in tasks:
        if task not in fixed["tasks"]:
            raise SystemExit(f"task {task!r} not pinned in {args.examples}")
        spec = fixed["tasks"][task]
        src = Path(spec["source_path"])
        rows = [json.loads(ln) for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(rows) != spec["n_examples"]:
            raise AssertionError(f"{task}: source now has {len(rows)} rows, pinned "
                                 f"{spec['n_examples']}")
        set_sha = hashlib.sha256(
            "".join(f"{r['id']}\x00{r['prompt']}" for r in rows).encode()).hexdigest()
        if set_sha != spec["prompt_set_sha256"]:
            raise AssertionError(f"{task}: prompt set changed\n  pinned   "
                                 f"{spec['prompt_set_sha256']}\n  computed {set_sha}")
        print(f"[{task}] DENOMINATOR n_examples={len(rows)}  prompt_set_sha256 OK", flush=True)

        by_id = {r["id"]: r for r in rows}
        ids = list(spec["ids"])
        if args.limit:
            ids = ids[:args.limit]
        pinned = {e["id"]: e for e in spec["examples"]}

        derange = build_derangement(ids, args.seed) if "A3" in arms else {}
        if derange:
            print(f"[{task}] derangement over {len(derange)} ids, 0 fixed points", flush=True)

        # ---- sample z once per example, shared by A3 and A4 ----
        need_z = [a for a in arms if a in ARMS_NEED_Z_SAMPLE]
        if need_z:
            for i, ex_id in enumerate(ids):
                key = f"{task}/{ex_id}"
                if key in z_cache:
                    continue
                prompt = by_id[ex_id]["prompt"]
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
                attn = torch.ones_like(input_ids)
                torch.manual_seed(example_seed(args.seed, args.seed_stream,
                                               tasks.index(task), i))
                cond = encode_prefix(model, input_ids, attn)
                z_cache[key] = sample_ddim_cfg(model, cond, args.steps, args.cfg_scale,
                                               args.device, dtype).detach()
                if (i + 1) % 25 == 0:
                    print(f"[{task}] sampled z {i + 1}/{len(ids)}", flush=True)
            if args.z_cache:
                torch.save({k: v.cpu() for k, v in z_cache.items()}, args.z_cache)
                print(f"[z_cache] saved {len(z_cache)} latents -> {args.z_cache}", flush=True)

        writers = {f: (out_dir / f"{task}__{f}.jsonl").open("w", encoding="utf-8") for f in arms}
        try:
            for i, ex_id in enumerate(ids):
                row = by_id[ex_id]
                pin = pinned[ex_id]
                prompt = row["prompt"]
                p_sha = hashlib.sha256(prompt.encode()).hexdigest()
                if p_sha != pin["prompt_sha256"]:
                    raise AssertionError(f"{task} id={ex_id}: prompt hash drift")
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
                attn = torch.ones_like(input_ids)
                span = prefill_span(input_ids, args.prefix_replay)
                base_out = prefill(model, span, torch.ones_like(span))

                for arm in arms:
                    donor_id = None
                    z_used: Optional[torch.Tensor] = None
                    if arm == "A1":
                        z_used = z_cache.get(f"{task}/{ex_id}")
                        if z_used is None:     # A1 only needs *some* states to call the
                            z_used = torch.zeros(1, int(model.latent_dim),
                                                 device=args.device, dtype=dtype)
                    elif arm == "A2":
                        z_used = cast(torch.Tensor, mean_z).to(dtype)
                    elif arm == "A3":
                        donor_id = derange[ex_id]
                        z_used = z_cache[f"{task}/{donor_id}"]
                        z_recip = z_cache[f"{task}/{ex_id}"]
                        # Tensor comparison, not index comparison: two different
                        # indices can carry identical tensors if sampling collapsed.
                        if torch.equal(z_used, z_recip):
                            raise AssertionError(
                                f"{task} id={ex_id}: donor z (id={donor_id}) is bit-identical "
                                "to the recipient's own z; the shuffle is a no-op for this row"
                            )
                    elif arm == "A4":
                        z_used = z_cache[f"{task}/{ex_id}"]

                    states = (model.predict_states(z_used.to(dtype))
                              if (arm != "A0" and z_used is not None) else None)
                    res = decode_arm(model, tokenizer, input_ids, attn, base_out, arm, states, args)
                    correct, invalid, pred_idx = score_row(
                        res["prediction"], pin["gold"], pin["choices"], args.letter_first)
                    key = (arm, task)
                    tot = totals.setdefault(key, [0, 0, 0])
                    tot[0] += 1
                    tot[1] += int(correct)
                    tot[2] += int(invalid)

                    rec = {
                        "arm": arm,
                        "task": task,
                        "example_id": ex_id,
                        "seed": args.seed,
                        "checkpoint_sha256": ckpt_sha,
                        "git_commit": git_commit,
                        "donor_id": donor_id,
                        "prompt_hash": p_sha,
                        "generated_ids": res["generated_ids"],
                        "prediction": res["prediction"],
                        "pred_choice_index": pred_idx,
                        "gold": pin["gold"],
                        "choices": pin["choices"],
                        "correct": bool(correct),
                        "invalid": bool(invalid),
                        "new_tokens": res["new_tokens"],
                        "latent_norm": (float(z_used.float().norm()) if z_used is not None else None),
                        "effective_lambda": (effective_lambda if arm != "A0" else None),
                        "state_scale": state_scale,
                        "adapter_name": res["adapter_name"],
                        "a1_layers_restored": res["a1_layers_restored"],
                        "prefix_replay": args.prefix_replay,
                        "seed_stream": args.seed_stream,
                        "prefix_replay_tokens": res["prefix_replay_tokens"],
                        "prefill_tokens": res["prefill_tokens"],
                        "boundary_tokens": res["boundary_tokens"],
                        "boundary_ids_sha256": res["boundary_ids_sha256"],
                        "prefix_encoding_rule":
                            "tokenizer(prompt) with no BOS injection and no truncation; "
                            "attention_mask=ones. prefix_replay=none prefills prompt[:-1], "
                            "writes, then feeds prompt[-1:] once (0 tokens re-fed); "
                            "prefix_replay=refeed prefills the whole prompt, writes, then "
                            "feeds the whole prompt again (|prompt| tokens re-fed). "
                            "Identical for all five arms.",
                        "sampler": run_meta["sampler"],
                        "cache_policy_hash": cache_policy_hash(
                            args.prefix_replay, arm, res["adapter_name"], zero_rnn_env, True),
                        "cache_policy": {
                            "deepcopy_before_mutation": True,
                            "adapter": res["adapter_name"],
                            "conv_ffn_zeroed_by_adapter": res["adapter_name"] != "none",
                            "seen_tokens_reset_by_adapter": False,
                            "cache_seen_tokens_before": res["cache_seen_tokens_before"],
                            "cache_seen_tokens_after": res["cache_seen_tokens_after"],
                            "cache_fp_before_sha256": res["cache_fp_before_sha256"],
                            "cache_fp_after_sha256": res["cache_fp_after_sha256"],
                        },
                        "first_token_logits_top10": res["first_token_logits_top10"],
                        "planning_seconds": res["planning_seconds"],
                        "decode_seconds": res["decode_seconds"],
                    }
                    writers[arm].write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (i + 1) % 20 == 0:
                    print(f"[{task}] {i + 1}/{len(ids)} examples x {len(arms)} arms", flush=True)
        finally:
            for w in writers.values():
                w.close()

    summary = {"meta": run_meta, "cells": {}}
    for (arm, task), (n, ncorr, ninv) in sorted(totals.items()):
        summary["cells"][f"{arm}/{task}"] = {
            "n": n, "correct": ncorr, "invalid": ninv, "acc_pct": 100.0 * ncorr / n,
        }
    for arm in arms:
        per = [summary["cells"][f"{arm}/{t}"] for t in tasks if f"{arm}/{t}" in summary["cells"]]
        if len(per) == len(tasks):
            summary["cells"][f"{arm}/MACRO3"] = {
                "n": sum(c["n"] for c in per),
                "acc_pct": sum(c["acc_pct"] for c in per) / len(per),
            }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cells"], indent=2))
    print(f"[done] -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

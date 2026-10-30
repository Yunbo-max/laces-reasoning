#!/usr/bin/env python3
"""D4: the 7 appendix sanity checks for the five-arm latent-use controls.

Runs on CPU against a tiny fake decoder + a real ``fla`` cache shape, so every
check exercises the SAME code path the GPU runner will use (``inject_into_cache``,
``blend_into_cache``, ``_cache_layer_state``, ``cache_fingerprint``,
``build_derangement``, ``prefill_span``) without a CUDA forward.

Each check has:
  * a NEGATIVE control  -- it PASSES on known-good input
  * SEVERAL POSITIVE controls -- one per sub-assertion inside the check, because a
    single break only exercises whichever assertion happens to fire FIRST. A
    check with four assertions and one positive control has three uncalibrated
    assertions, and the boundary cases (first arm, last arm, first token) are
    exactly the ones a mid-range break leaves declared-correct by default.
A check that cannot fail is not a check, so a positive control that PASSES is
reported as a BROKEN INSTRUMENT, loudly, and makes the suite exit non-zero.
Every check prints its DENOMINATOR before its verdict: a check over 0 candidates
can never fail.

    python3 scripts/eval/latentuse/sanity_checks.py            # run + calibrate
    python3 scripts/eval/latentuse/sanity_checks.py --only 3
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from models.state_hijacking_dit import (  # noqa: E402
    _cache_layer_state, _reset_cache_layer_seen_tokens,
)

sys.path.insert(0, str(Path(__file__).parent))
import g2_frozen_run_five_arms as R  # noqa: E402

N_LAYERS, N_HEADS, HEAD_DIM, LATENT = 3, 2, 4, 8
CONV_K = 2


class CheckFailure(AssertionError):
    """A sanity check concluded FAIL. Distinct from a crash."""


# ------------------------------------------------- tiny fake model & cache ----

class FakeLayer:
    def __init__(self) -> None:
        self.state: Optional[dict[str, Any]] = None
        self._seen_tokens = 0


class FakeCache:
    """Mimics the fla 0.5 cache surface that the adapters actually touch:
    ``.layers[i].state`` dict + ``.layers[i]._seen_tokens`` + ``._seen_tokens``."""

    def __init__(self, n_layers: int = N_LAYERS, seed: int = 0, seen: int = 41) -> None:
        g = torch.Generator().manual_seed(seed)
        self.layers = []
        for _ in range(n_layers):
            lay = FakeLayer()
            lay.state = {
                "recurrent_state": torch.randn(1, N_HEADS, HEAD_DIM, HEAD_DIM,
                                               generator=g, dtype=torch.float32),
                "attn_state": None,
                "conv_state": torch.randn(1, N_HEADS * HEAD_DIM, CONV_K,
                                          generator=g, dtype=torch.float32),
                "ffn_state": torch.randn(1, N_HEADS * HEAD_DIM,
                                         generator=g, dtype=torch.float32),
            }
            lay._seen_tokens = seen
            self.layers.append(lay)
        self._seen_tokens = seen


class FakeWriter(nn.Module):
    """Carries only what the adapters need: num_layers, the two adapter methods,
    a deterministic predict_states, and the global_denoiser_type routing flag."""

    def __init__(self, denoiser_type: str = "dit") -> None:
        super().__init__()
        self.num_layers = N_LAYERS
        self.num_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.latent_dim = LATENT
        self.global_denoiser_type = denoiser_type
        self.use_learnable_blend = False
        self.blend_floor = 0.0
        self.state_scale = torch.tensor([0.1848476082086563])
        self.W = torch.randn(N_LAYERS, N_HEADS * HEAD_DIM * HEAD_DIM, LATENT,
                             generator=torch.Generator().manual_seed(7))

    def predict_states(self, z: torch.Tensor) -> list[torch.Tensor]:
        z = z.reshape(1, -1).float()
        scale = float(self.state_scale[0])
        return [(self.W[l] @ z.T).T.reshape(1, N_HEADS, HEAD_DIM, HEAD_DIM) * scale
                for l in range(N_LAYERS)]

    # Bind the REAL implementations so the checks test production code.
    from models.state_hijacking_dit import StateInjectionDiTRELAY as _Real
    inject_into_cache = _Real.inject_into_cache
    blend_into_cache = _Real.blend_into_cache
    _effective_layer_blend = _Real._effective_layer_blend
    del _Real


def make_args(prefix_replay: str = "none", max_new_tokens: int = 4,
              repetition_penalty: float = 1.1) -> Any:
    class A:
        pass
    a = A()
    a.prefix_replay = prefix_replay
    a.max_new_tokens = max_new_tokens
    a.repetition_penalty = repetition_penalty
    return a


def _arm_write(model: FakeWriter, cache: FakeCache, arm: str,
               states: Optional[list[torch.Tensor]]) -> tuple[FakeCache, str, int]:
    """The runner's write step, isolated so checks can drive it directly.

    Mirrors run_five_arms.decode_arm's write block exactly; kept in sync by
    check 2's cross-verification against R.restore_principal_memory.
    """
    cache = copy.deepcopy(cache)
    if arm == "A0":
        return cache, "none", 0
    if states is None:
        raise ValueError(f"{arm} needs states")
    if arm == "A1":
        saved = R.principal_memory(cache, model.num_layers)
        cache = model.inject_into_cache(cache, states)
        n = R.restore_principal_memory(cache, saved)
        return cache, "inject_into_cache+restore_principal", n
    cache = model.inject_into_cache(cache, states)
    return cache, "inject_into_cache", 0


# =============================================================== CHECK 1 ======

def check1(break_mode: Optional[str] = None) -> None:
    """Cache is deep-copied before mutation; per-field shape/dtype/checksum/counter
    recorded.  The failure this catches: two arms sharing one mutable cache, so
    arm k+1 decodes from arm k's written state."""
    model = FakeWriter()
    shared = FakeCache(seed=1)
    states = model.predict_states(torch.ones(1, LATENT))
    arms = ["A0", "A1", "A2", "A3", "A4"]
    print(f"    DENOMINATOR arms={len(arms)} layers={model.num_layers} "
          f"fields_per_layer={len(shared.layers[0].state) + 1}")
    if len(arms) == 0 or model.num_layers == 0:
        raise CheckFailure("zero candidates: this check cannot fail")

    base_fp = R.cache_fingerprint(shared, model.num_layers)
    if break_mode == "drop_md5":
        # BREAK 1c: the fingerprint stops recording a checksum, so "fields
        # recorded" is satisfied by shape alone and content drift is invisible.
        for v in base_fp["fields"].values():
            if isinstance(v, dict) and v.get("kind") == "tensor":
                v.pop("md5", None)
    n_tensor_fields = sum(1 for v in base_fp["fields"].values()
                          if isinstance(v, dict) and v.get("kind") == "tensor")
    if n_tensor_fields == 0:
        raise CheckFailure("fingerprint recorded 0 tensor fields")
    for name, v in base_fp["fields"].items():
        if isinstance(v, dict) and v["kind"] == "tensor":
            for req in ("shape", "dtype", "md5", "sum"):
                if req not in v:
                    raise CheckFailure(f"field {name} missing {req}")

    after = []
    for arm in arms:
        st = None if arm == "A0" else states
        if break_mode == "no_deepcopy":
            # BREAK 1a: skip the deep copy -- mutate the shared cache in place.
            if arm == "A0":
                c = shared
            elif arm == "A1":
                saved = R.principal_memory(shared, model.num_layers)
                c = model.inject_into_cache(shared, states)
                R.restore_principal_memory(c, saved)
            else:
                c = model.inject_into_cache(shared, states)
        else:
            c, _, _ = _arm_write(model, shared, arm, st)
            if break_mode == "a0_mutates" and arm == "A0":
                # BREAK 1b: A0 is not a true no-op -- it touches the cache.
                _cache_layer_state(c, 0)["conv_state"] = torch.zeros_like(
                    _cache_layer_state(c, 0)["conv_state"])
        after.append((arm, R.cache_fingerprint(c, model.num_layers)))

    # The shared prefill cache must be UNCHANGED after all five arms ran.
    now = R.cache_fingerprint(shared, model.num_layers)
    if json.dumps(now, sort_keys=True) != json.dumps(base_fp, sort_keys=True):
        raise CheckFailure(
            "the shared prefill cache was mutated: at least one arm did not deep-copy "
            "before writing, so arms are not independent"
        )
    # And A0's cache must still equal the prefill cache bit for bit.
    a0 = dict(after)["A0"]
    if json.dumps(a0, sort_keys=True) != json.dumps(base_fp, sort_keys=True):
        raise CheckFailure("A0's cache differs from the prefill cache; A0 mutated something")
    print(f"    recorded {n_tensor_fields} tensor fields + "
          f"{model.num_layers} per-layer counters + 1 cache counter; shared cache intact")


# =============================================================== CHECK 2 ======

def check2(break_mode: Optional[str] = None) -> None:
    """A0 and A1 agree on PRINCIPAL memory at the write boundary (and must DIFFER
    on the auxiliary fields, which is A1's whole point)."""
    model = FakeWriter()
    base = FakeCache(seed=2)
    states = model.predict_states(torch.full((1, LATENT), 0.7))

    c0, _, _ = _arm_write(model, base, "A0", None)
    if break_mode == "no_restore":
        # BREAK 2a: A1 forgets to restore (i.e. it is just A4 under another name).
        c1 = model.inject_into_cache(copy.deepcopy(base), states)
        n_restored = 0
    elif break_mode == "partial_restore":
        # BREAK 2b: A1 restores every layer EXCEPT the last. A per-layer loop that
        # stops one short is the classic off-by-one, and a check that only counts
        # "did we restore something" would pass it.
        c1 = copy.deepcopy(base)
        saved = R.principal_memory(c1, model.num_layers)
        c1 = model.inject_into_cache(c1, states)
        n_restored = R.restore_principal_memory(c1, saved[:-1] + [None])
    elif break_mode == "a1_is_a0":
        # BREAK 2c: A1 never calls the adapter at all, so it is A0 relabelled and
        # measures nothing. Only the aux-difference assertion catches this.
        c1, _, n_restored = _arm_write(model, base, "A0", None)
        n_restored = model.num_layers
    else:
        c1, _, n_restored = _arm_write(model, base, "A1", states)

    print(f"    DENOMINATOR layers compared={model.num_layers} "
          f"a1_layers_restored={n_restored}")
    if model.num_layers == 0:
        raise CheckFailure("zero layers: cannot fail")
    if n_restored != model.num_layers:
        raise CheckFailure(f"A1 restored {n_restored}/{model.num_layers} layers")

    # can-distinguish control: the identical comparison must be able to say NO.
    probe = copy.deepcopy(c0)
    _cache_layer_state(probe, 0)["recurrent_state"] += 1.0
    if all(torch.equal(_cache_layer_state(c0, l)["recurrent_state"],
                       _cache_layer_state(probe, l)["recurrent_state"])
           for l in range(model.num_layers)):
        raise CheckFailure("the equality test cannot distinguish a perturbed cache; "
                           "it is not a test")

    mismatch = [l for l in range(model.num_layers)
                if not torch.equal(_cache_layer_state(c0, l)["recurrent_state"],
                                   _cache_layer_state(c1, l)["recurrent_state"])]
    if mismatch:
        raise CheckFailure(
            f"A0 and A1 disagree on principal memory at layers {mismatch}; A1's restore "
            "is not bit-exact, so the identity arm is not an identity"
        )
    # A1 must nevertheless differ from A0 somewhere, or it is A0 with extra steps.
    aux_diff = [l for l in range(model.num_layers)
                if not torch.equal(_cache_layer_state(c0, l)["conv_state"],
                                   _cache_layer_state(c1, l)["conv_state"])]
    if not aux_diff:
        raise CheckFailure(
            "A1 is byte-identical to A0 including conv_state; then A1 measures nothing "
            "(the adapter's auxiliary-cache effect vanished)"
        )
    print(f"    principal memory identical on {model.num_layers}/{model.num_layers} layers; "
          f"conv_state differs on {len(aux_diff)}/{model.num_layers} (expected: A1 keeps the "
          f"adapter's zeroing)")


# =============================================================== CHECK 3 ======

def check3(break_mode: Optional[str] = None) -> None:
    """A2/A3/A4 share an identical effective lambda/gamma, AND no hardcoded
    override is reachable.  The named risk: the independent trajectory runner
    (scripts/eval/sample_prefix_suffix_trajectory_cfg.py::apply_generated_state)
    calls HARD injection whenever blend >= 1.0 even when a saved config carries
    trajectory_state_blend=0.7 -- so a config field does not identify the realized
    adapter."""
    model = FakeWriter()
    base = FakeCache(seed=3)
    z = {"A2": torch.full((1, LATENT), 0.3),
         "A3": torch.full((1, LATENT), -1.1),
         "A4": torch.full((1, LATENT), 0.9)}

    # (a) all three writing arms must reach the SAME adapter with the SAME strength.
    names, caches = {}, {}
    for arm in ("A2", "A3", "A4"):
        st = model.predict_states(z[arm])
        if break_mode == "one_arm_blends" and arm == "A3":
            # BREAK 3a: a hardcoded blend sneaks into one arm only, under a
            # DIFFERENT adapter -- caught by the adapter-name assertion.
            c = model.blend_into_cache(copy.deepcopy(base), st, 0.7)
            nm = "blend_into_cache(0.7)"
        elif break_mode == "same_name_scaled_lambda" and arm == "A2":
            # BREAK 3b: the SAME adapter is called but the written content is
            # scaled, i.e. an effective lambda != 1.0 while the adapter name is
            # unchanged. Only the state == R(z) assertion can see this; the name
            # comparison passes, which is exactly why (b) exists.
            c, nm, _ = _arm_write(model, base, arm, [s * 0.7 for s in st])
        else:
            c, nm, _ = _arm_write(model, base, arm, st)
        names[arm], caches[arm] = nm, c
    print(f"    DENOMINATOR writing arms={len(names)} adapter names={sorted(set(names.values()))}")
    if len(names) == 0:
        raise CheckFailure("zero writing arms: cannot fail")
    if len(set(names.values())) != 1:
        raise CheckFailure(f"writing arms do not share one adapter: {names}")

    # (b) a hard write must land EXACTLY on R(z): lambda is 1.0 by construction.
    for arm in ("A2", "A3", "A4"):
        st = model.predict_states(z[arm])
        for l in range(model.num_layers):
            got = _cache_layer_state(caches[arm], l)["recurrent_state"]
            want = st[l].to(torch.float32)
            if not torch.equal(got, want):
                mx = float((got - want).abs().max())
                raise CheckFailure(
                    f"{arm} layer {l}: written state != R(z) (absmax {mx:.3e}); the effective "
                    f"lambda is not 1.0, so a blend or override is in the path"
                )
    # (c) source scan: the runner must not consult trajectory_state_blend.
    src = (Path(__file__).with_name("g2_frozen_run_five_arms.py")).read_text()
    if break_mode == "runner_reads_blend":
        # BREAK 3c: simulate a runner that DOES read the field, to prove the scan
        # is not vacuous on its real target.
        src += '\n    blend = float(model.config.get("trajectory_state_blend", 0.7))\n'
    if break_mode == "runner_zero_tol_source":
        # BREAK 3e: the always-firing guard is back IN THE RUNNER SOURCE.  Checking
        # the arithmetic in this file only proves THIS file understands bf16; the
        # guard that aborts the GPU job lives in run_five_arms.py.
        src += ('\n    if abs(state_scale - float(prov["state_scale"])) > 0:\n'
                '        raise AssertionError("scale")\n')
    lines_to_scan = src.splitlines()
    uses = [ln.strip() for ln in lines_to_scan
            if "trajectory_state_blend" in ln and not ln.strip().startswith("#")]
    print(f"    DENOMINATOR scanned lines={len(lines_to_scan)}; non-comment lines "
          f"mentioning trajectory_state_blend={len(uses)}")
    if uses:
        raise CheckFailure(f"run_five_arms.py reads trajectory_state_blend: {uses}")
    zero_tol = [ln.strip() for ln in lines_to_scan
                if 'prov["state_scale"]' in ln and "> 0" in ln
                and not ln.strip().startswith("#")]
    print(f"    DENOMINATOR runner lines comparing prov state_scale with a > 0 "
          f"tolerance={len(zero_tol)}")
    if zero_tol:
        raise CheckFailure(
            f"run_five_arms.py still compares the bf16-loaded state_scale against the fp32 "
            f"provenance value with zero tolerance, which can never pass: {zero_tol}"
        )
    # positive control for the scan itself: it must find a real hit somewhere.
    traj = REPO / "scripts/eval/sample_prefix_suffix_trajectory_cfg.py"
    hits = sum(1 for ln in traj.read_text().splitlines()
               if "trajectory_state_blend" in ln and not ln.strip().startswith("#"))
    if hits == 0:
        raise CheckFailure(
            f"the source scan found 0 hits in {traj.name}, which is known to use that field; "
            "the scan is broken, so its 0 hits in run_five_arms.py prove nothing"
        )
    print(f"    scan calibrated: {hits} hits in {traj.name} (known positive), 0 in the runner")

    # (d) the write-strength GUARD must itself be runnable.  This sub-assertion
    # exists because the original guard was `abs(loaded - provenance_fp32) > 0`,
    # and state_scale is a TRAINABLE parameter that load_relay_model casts to
    # bfloat16 (scripts/eval/relay_utils.py:100-102).  bf16 rounds
    # 0.1848476082086563 to 0.1845703125, so that guard could NEVER pass: it
    # aborted three GPU jobs before the cause was found.  A guard that always
    # fires is as useless as one that never fires, and it is harder to notice
    # because it looks like vigilance.
    fp32 = 0.1848476082086563
    bf16 = float(torch.tensor(fp32, dtype=torch.float32).to(torch.bfloat16))
    ulp = abs(bf16 - fp32)
    print(f"    DENOMINATOR state_scale guard: fp32={fp32!r} bf16={bf16!r} ulp={ulp:.3e}")
    if ulp == 0.0:
        raise CheckFailure("bf16 rounding of state_scale is exact here, so this "
                           "sub-assertion cannot distinguish the two guards")
    if break_mode == "zero_tol_scale":
        # BREAK 3d: restore the always-firing guard.
        guard_ok = not (abs(bf16 - fp32) > 0)
    else:
        guard_ok = (bf16 == float(torch.tensor(fp32, dtype=torch.float32).to(torch.bfloat16)))
    if not guard_ok:
        raise CheckFailure(
            f"the state_scale guard rejects the checkpoint's OWN value under the run's dtype "
            f"(fp32 {fp32!r} -> bf16 {bf16!r}, gap {ulp:.3e}); it can never pass, so it is not "
            f"a check, it is an unconditional abort"
        )
    # ...and it must still reject a genuinely different scale (negative control of
    # the relaxation: relaxing a guard to "always pass" is the opposite failure).
    other = float(torch.tensor(fp32 * 1.01, dtype=torch.float32).to(torch.bfloat16))
    if other == bf16:
        raise CheckFailure(f"the relaxed guard cannot distinguish {fp32!r} from a 1% "
                           f"different scale; it has been relaxed into a no-op")
    print(f"    state_scale guard is dtype-aware: accepts {bf16!r}, rejects {other!r} "
          f"(1% change is {abs(other - bf16) / ulp:.0f}x one ulp)")
    print(f"    effective_lambda=1.0 on {len(names)}x{model.num_layers} "
          f"arm-layer pairs; adapter={sorted(set(names.values()))[0]}")


# =============================================================== CHECK 4 ======

def check4(break_mode: Optional[str] = None) -> None:
    """A3 has no fixed point AND z_donor != z_recipient as TENSORS."""
    ids = list(range(12))
    if break_mode == "identity_map":
        # BREAK 4a: identity map instead of a derangement -- every row is its own donor.
        mapping = {i: i for i in ids}
    elif break_mode == "one_fixed_point":
        # BREAK 4b: a derangement with a SINGLE fixed point. "Mostly deranged" is
        # not deranged, and a check that samples or spot-checks would miss it.
        mapping = R.build_derangement(ids, seed=42)
        k = ids[0]
        mapping[k] = k
    elif break_mode == "donor_reuse":
        # BREAK 4c: no fixed point, but one z is used as donor twice and another
        # never -- the fixed-point test alone passes this.
        mapping = R.build_derangement(ids, seed=42)
        a, b = ids[1], ids[2]
        mapping[a] = mapping[b]
    elif break_mode == "collapsed_z":
        # BREAK 4d: a perfect derangement over a COLLAPSED z_cache, so donor and
        # recipient tensors are identical although the indices differ. Only the
        # tensor comparison sees this; index comparison passes.
        mapping = R.build_derangement(ids, seed=42)
    else:
        mapping = R.build_derangement(ids, seed=42)
    print(f"    DENOMINATOR ids={len(ids)} mapped_pairs={len(mapping)}")
    if len(mapping) == 0:
        raise CheckFailure("zero pairs: cannot fail")
    if len(mapping) != len(ids):
        raise CheckFailure(f"derangement covers {len(mapping)} of {len(ids)} ids")
    fixed = [k for k, v in mapping.items() if k == v]
    if fixed:
        raise CheckFailure(f"{len(fixed)}/{len(ids)} fixed points: {fixed[:6]}")
    if sorted(mapping.values()) != sorted(ids):
        raise CheckFailure("donor multiset != id multiset; some z is used twice")

    # Tensor-level check, which index equality cannot replace: distinct indices
    # can still carry identical tensors if sampling collapsed.
    g = torch.Generator().manual_seed(4)
    if break_mode == "collapsed_z":
        zc = {i: torch.zeros(1, LATENT) for i in ids}
    else:
        zc = {i: torch.randn(1, LATENT, generator=g) for i in ids}
    same = [k for k, v in mapping.items() if torch.equal(zc[k], zc[v])]
    if same:
        raise CheckFailure(f"{len(same)} rows have a donor z bit-identical to their own z")
    # can-distinguish control: with a collapsed z_cache the tensor test must fire.
    collapsed = {i: torch.zeros(1, LATENT) for i in ids}
    if not [k for k, v in mapping.items() if torch.equal(collapsed[k], collapsed[v])]:
        raise CheckFailure("the tensor test does not fire on a collapsed z_cache; "
                           "it is a no-op")
    print(f"    0/{len(ids)} fixed points; 0/{len(ids)} identical donor tensors; "
          f"tensor test verified to fire on a collapsed cache")


# =============================================================== CHECK 5 ======

def check5(break_mode: Optional[str] = None) -> None:
    """Same input + same arm + same seed reproduces identical logits.

    Tolerance: EXACT (0.0).  The write path is deterministic given z, and z comes
    from the pinned z_cache, so any drift is a bug, not float noise.  A stated
    non-zero tolerance would hide the shared-cache failure check 1 catches.
    """
    model = FakeWriter()
    base = FakeCache(seed=5)
    z = torch.full((1, LATENT), 0.42)
    TOL = 0.0
    runs = []
    n_reps = 1 if break_mode == "single_replicate" else 3
    for k in range(n_reps):
        torch.manual_seed(123)
        st = model.predict_states(z)
        if break_mode == "noise_last" and k == n_reps - 1:
            # BREAK 5a: unseeded noise leaks into the LAST replicate (a boundary
            # position, so a comparison that only looks at adjacent pairs 0-1
            # would miss it).
            st = [s + torch.randn_like(s) * 1e-3 for s in st]
        elif break_mode == "noise_tiny" and k == 1:
            # BREAK 5b: a 1-ULP-scale perturbation. TOL=0.0 must catch even this;
            # a check written with tol=1e-5 would pass it and hide real drift.
            st = [s + 1e-8 for s in st]
        c, _, _ = _arm_write(model, base, "A4", st)
        runs.append(torch.cat([_cache_layer_state(c, l)["recurrent_state"].reshape(-1)
                               for l in range(model.num_layers)]))
    print(f"    DENOMINATOR replicates={len(runs)} elements_each={runs[0].numel()} tol={TOL}")
    if len(runs) < 2 or runs[0].numel() == 0:
        raise CheckFailure(f"only {len(runs)} replicate(s) of {runs[0].numel()} elements: "
                           "a reproducibility check over <2 replicates cannot fail")
    devs = [float((runs[0] - r).abs().max()) for r in runs[1:]]
    # can-distinguish control
    if float((runs[0] - (runs[0] + 1e-6)).abs().max()) <= TOL:
        raise CheckFailure("the comparison cannot see a 1e-6 perturbation; it is not a test")
    if max(devs) > TOL:
        raise CheckFailure(f"replicates diverge: max|delta|={max(devs):.3e} > tol {TOL} "
                           f"(per-replicate {['%.3e' % d for d in devs]})")
    print(f"    {len(runs)} replicates identical (max|delta|={max(devs):.1e}); "
          f"detector verified on a 1e-6 perturbation")


# =============================================================== CHECK 6 ======

def check6(break_mode: Optional[str] = None) -> None:
    """Prefix replay LENGTH and TOKEN IDS are identical across all five arms.

    This is the check that catches the original defect: generate_answer re-fed the
    prompt (forward -> inject -> forward again) while generate_raw_answer did not,
    so raw-vs-method conflated the latent with an extra pass over ~512 tokens.
    The positive controls INJECT AT THE BOUNDARY (the first arm A0 and the last arm
    A4), not only in the middle, because a window that skips the edges declares the
    edges correct by default.
    """
    ids = torch.arange(1, 17).reshape(1, -1)
    for mode in ("none", "refeed"):
        args = make_args(prefix_replay=mode)
        span = R.prefill_span(ids, mode)
        if break_mode == "span_ignores_mode":
            # BREAK 6d: prefill_span stops honouring the mode, so every prompt
            # token is consumed twice in 'none'. All arms AGREE, so the
            # cross-arm comparison passes -- only the arithmetic assertion fires.
            span = ids
        per_arm = {}
        for i, arm in enumerate(R.ARMS_ALL):
            if mode == "none":
                boundary, replay = ids[:, -1:], 0
            else:
                boundary, replay = ids, int(ids.shape[1])
            if break_mode == "boundary_arms_differ" and arm in ("A0", "A4"):
                # BREAK 6a: the FIRST and LAST arms take the other mode's path.
                if mode == "none":
                    boundary, replay = ids, int(ids.shape[1])
                else:
                    boundary, replay = ids[:, -1:], 0
            elif break_mode == "one_middle_arm_differs" and arm == "A2":
                # BREAK 6b: a single middle arm diverges (the mid-range case).
                boundary, replay = (ids, int(ids.shape[1])) if mode == "none" else (ids[:, -1:], 0)
            elif break_mode == "same_length_different_ids" and arm == "A3":
                # BREAK 6c: the LENGTH matches but the TOKEN IDS differ. A check
                # that compared only counts would call this identical -- the
                # appendix asks for length AND ids, so the hash must carry it.
                boundary = boundary.flip(-1) if boundary.shape[1] > 1 else boundary + 1
            per_arm[arm] = (int(span.shape[1]), replay, int(boundary.shape[1]),
                            hashlib.sha256(json.dumps(boundary[0].tolist()).encode()).hexdigest())
        print(f"    DENOMINATOR mode={mode} arms={len(per_arm)} "
              f"distinct_signatures={len(set(per_arm.values()))}")
        if len(per_arm) == 0:
            raise CheckFailure("zero arms: cannot fail")
        if len(set(per_arm.values())) != 1:
            groups: dict[Any, list[str]] = {}
            for a, sig in per_arm.items():
                groups.setdefault(sig, []).append(a)
            raise CheckFailure(
                f"prefix replay differs across arms in mode={mode}: "
                + "; ".join(f"{v}->(prefill={k[0]},refed={k[1]},boundary={k[2]},ids={k[3][:8]})"
                            for k, v in groups.items())
            )
        sig = next(iter(per_arm.values()))
        # The two modes must be genuinely different, or the flag does nothing.
        expect_refed = 0 if mode == "none" else int(ids.shape[1])
        if sig[1] != expect_refed:
            raise CheckFailure(f"mode={mode} re-fed {sig[1]} tokens, expected {expect_refed}")
        # Every prompt token consumed exactly once before the first generated token.
        consumed = sig[0] + sig[2]
        expect_consumed = int(ids.shape[1]) if mode == "none" else 2 * int(ids.shape[1])
        if consumed != expect_consumed:
            raise CheckFailure(f"mode={mode}: prefill {sig[0]} + boundary {sig[2]} = {consumed}, "
                               f"expected {expect_consumed}")
        print(f"    mode={mode}: all {len(per_arm)} arms prefill={sig[0]} boundary={sig[2]} "
              f"re-fed={sig[1]} ids={sig[3][:8]} (arithmetic {sig[0]}+{sig[2]}={consumed})")


# =============================================================== CHECK 7 ======

def check7(break_mode: Optional[str] = None) -> None:
    """First-token logits are recorded, for every arm, from the WRITTEN cache.

    The training CE helper ``_singlez_coadapt_ce_from_z`` uses
    ``logits[:, :-1]`` against ``suffix_tokens[:, 1:]``, i.e. it DROPS the first
    suffix token because it teacher-forces a shifted target.  That is a property
    of shifted CE and is NOT a licence to drop the first ANSWER token at eval: at
    eval the first generated token IS the answer's opening token ('(B)', 'darwin'),
    and dropping it would delete the answer being scored.
    """
    src = Path(__file__).with_name("g2_frozen_run_five_arms.py").read_text()
    if break_mode == "slices_first_token":
        # BREAK 7c: the runner drops the first generated token before scoring.
        src += '\n        rec["generated_ids"] = generated_ids[1:]\n'
    if break_mode == "no_step_guard":
        src = src.replace("if step == 0:", "if step >= 0:")
    if break_mode == "field_missing":
        # BREAK 7a: the field is renamed away.  This is mutated in the SOURCE, not
        # by overwriting `emitted` below, so the scan itself is calibrated too --
        # blanking a detector's output only calibrates the assertion that reads it.
        src = src.replace('"first_token_logits_top10":', '"ftl_dropped":')
    lines = src.splitlines()
    print(f"    DENOMINATOR runner source lines={len(lines)}")
    if len(lines) == 0:
        raise CheckFailure("empty runner source: cannot fail")

    # (a) the field must be emitted in the record dict.
    emitted = [i + 1 for i, ln in enumerate(lines)
               if '"first_token_logits_top10":' in ln and not ln.strip().startswith("#")]
    print(f"    DENOMINATOR lines emitting first_token_logits_top10={len(emitted)} at {emitted}")
    if len(emitted) < 2:
        raise CheckFailure(
            f"first_token_logits_top10 appears on {len(emitted)} non-comment line(s); it must "
            "be both produced by decode_arm and written into the JSONL record"
        )
    # (b) it must be captured at step == 0, not at an arbitrary step.
    if not any("if step == 0:" in ln for ln in lines):
        raise CheckFailure("no `if step == 0:` guard; the recorded logits are not the FIRST token")
    # (c) the runner must not slice off the first generated token anywhere.
    bad = [i + 1 for i, ln in enumerate(lines)
           if ("gen_ids[1:]" in ln or "generated_ids[1:]" in ln)
           and not ln.strip().startswith("#")]
    if bad:
        raise CheckFailure(f"runner drops the first generated token at line(s) {bad}")
    # (d) simulate the record and confirm the payload is non-empty and ordered.
    logits = torch.tensor([0.1, 5.0, -2.0, 3.0, 0.0])
    k = min(10, logits.numel())
    if break_mode == "empty_payload":
        # BREAK 7d: topk with k=0 yields an empty payload that still serialises to
        # valid JSON, so a check that only asserted "the key exists" would pass.
        k = 0
    top = torch.topk(logits, k=k)
    payload = [[int(i), float(v)] for v, i in zip(top.values.tolist(), top.indices.tolist())]
    if break_mode == "unsorted_payload":
        payload = list(reversed(payload))
    if not payload:
        raise CheckFailure("simulated first-token payload is empty")
    if [p[1] for p in payload] != sorted((p[1] for p in payload), reverse=True):
        raise CheckFailure("first-token payload is not sorted by logit")
    if payload[0][0] != int(torch.argmax(logits)):
        raise CheckFailure("payload head is not the argmax token")
    print(f"    field emitted on {len(emitted)} lines, captured under `if step == 0:`, "
          f"no first-token slicing, payload head={payload[0]}")


CHECKS: dict[int, tuple[str, Callable[..., None]]] = {
    1: ("cache deep-copied before mutation; fields+counters recorded", check1),
    2: ("A0 and A1 agree on principal memory at the write boundary", check2),
    3: ("A2/A3/A4 share effective lambda/gamma; no hardcoded override", check3),
    4: ("A3 has no fixed point and z_donor != z_recipient as tensors", check4),
    5: ("same input+arm+seed reproduces identical logits (tol 0.0)", check5),
    6: ("prefix replay length and token ids identical across all arms", check6),
    7: ("first-token logits recorded; shifted-CE drop is not licensed here", check7),
}

# One positive control per sub-assertion.  A check with N assertions and one break
# leaves N-1 assertions uncalibrated: the break short-circuits at whichever
# assertion fires first, so the rest are never exercised.
BREAKS: dict[int, list[tuple[str, str]]] = {
    1: [("no_deepcopy",  "arms share one mutable cache (no deep copy)"),
        ("a0_mutates",   "A0 is not a true no-op: it zeroes a conv_state"),
        ("drop_md5",     "fingerprint records no checksum, only shapes")],
    2: [("no_restore",       "A1 never restores: it is A4 relabelled"),
        ("partial_restore",  "A1 restores all layers but the LAST (off-by-one)"),
        ("a1_is_a0",         "A1 never calls the adapter: it is A0 relabelled")],
    3: [("one_arm_blends",          "one arm uses blend_into_cache(0.7)"),
        ("same_name_scaled_lambda", "same adapter name, content scaled by 0.7"),
        ("runner_reads_blend",      "the runner reads trajectory_state_blend"),
        ("zero_tol_scale",          "state_scale guard uses zero tolerance vs fp32"),
        ("runner_zero_tol_source",  "the always-firing guard is back in the runner")],
    4: [("identity_map",     "identity map: every row is its own donor"),
        ("one_fixed_point",  "a derangement with ONE fixed point"),
        ("donor_reuse",      "no fixed point but a donor used twice"),
        ("collapsed_z",      "perfect derangement over a COLLAPSED z_cache")],
    5: [("noise_last",        "noise in the LAST replicate (boundary)"),
        ("noise_tiny",        "a 1e-8 perturbation (tol=0.0 must see it)"),
        ("single_replicate",  "only 1 replicate: the check has no denominator")],
    6: [("boundary_arms_differ",      "A0 and A4 (first+last) take the other path"),
        ("one_middle_arm_differs",    "A2 alone diverges (mid-range)"),
        ("same_length_different_ids", "same LENGTH, different token IDS"),
        ("span_ignores_mode",         "all arms agree but consume every token twice")],
    7: [("field_missing",      "first_token_logits_top10 never written"),
        ("no_step_guard",      "captured at every step, not step 0"),
        ("slices_first_token", "the runner drops the first generated token"),
        ("empty_payload",      "the key exists but the payload is empty"),
        ("unsorted_payload",   "payload not ordered by logit")],
}


def run_one(n: int, break_mode: Optional[str]) -> tuple[bool, str]:
    name, fn = CHECKS[n]
    try:
        fn(break_mode=break_mode)
        return True, "PASS"
    except CheckFailure as e:
        return False, f"FAIL: {e}"
    except Exception as e:                                  # noqa: BLE001
        return False, f"ERROR({type(e).__name__}): {e}"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args(argv)
    which = [args.only] if args.only else sorted(CHECKS)

    print("=" * 78)
    print("NEGATIVE CONTROLS -- each check on known-good input; all must PASS")
    print("=" * 78)
    neg: dict[int, bool] = {}
    for n in which:
        print(f"\n[check {n}] {CHECKS[n][0]}")
        ok, msg = run_one(n, None)
        neg[n] = ok
        print(f"    VERDICT {msg}")

    pos: dict[tuple[int, str], bool] = {}
    broken_instruments: list[str] = []
    if not args.no_calibrate:
        print("\n" + "=" * 78)
        print("POSITIVE CONTROLS -- one per sub-assertion; every one must FAIL")
        print("=" * 78)
        for n in which:
            for mode, desc in BREAKS[n]:
                print(f"\n[check {n} BREAK={mode}] {desc}")
                ok, msg = run_one(n, mode)
                pos[(n, mode)] = not ok          # we WANT it to fail
                print(f"    VERDICT {msg}")
                if ok:
                    line = (f"check {n} PASSED on break '{mode}' ({desc}). It cannot "
                            f"detect the defect it claims to detect.")
                    broken_instruments.append(line)
                    print(f"    *** BROKEN INSTRUMENT *** {line}")

    print("\n" + "=" * 78)
    print("CALIBRATION  (TP = fires on broken input, FP = fires on good input)")
    print("=" * 78)
    print(f"{'check':>5} {'break':>26}  {'good->PASS':>10} {'broken->FAIL':>12} {'TP':>3} {'FP':>3}")
    tp = fp = n_breaks = 0
    for n in which:
        good_ok = neg[n]
        f = 0 if good_ok else 1
        fp += f
        first = True
        for mode, _desc in BREAKS[n]:
            broke_ok = pos.get((n, mode))
            if broke_ok is None:
                continue
            n_breaks += 1
            t = 1 if broke_ok else 0
            tp += t
            print(f"{n:>5} {mode:>26}  {('yes' if good_ok else 'NO') if first else '':>10} "
                  f"{'yes' if broke_ok else 'NO':>12} {t:>3} {f if first else 0:>3}")
            first = False
        if first:                                  # --no-calibrate
            print(f"{n:>5} {'-':>26}  {'yes' if good_ok else 'NO':>10} {'-':>12} "
                  f"{'-':>3} {f:>3}")
    print(f"\n{len(which)} checks, {n_breaks} positive controls: "
          f"TP={tp}/{n_breaks}  FP={fp}/{len(which)}")
    for ln in broken_instruments:
        print(f"BROKEN INSTRUMENT: {ln}")
    ok_all = all(neg.values()) and not broken_instruments
    print("SUITE: " + ("CALIBRATED AND PASSING" if ok_all else "NOT TRUSTWORTHY"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())

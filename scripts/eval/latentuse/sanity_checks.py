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
import run_five_arms as R  # noqa: E402

# Scan the file the imported module ACTUALLY came from, not a sibling with the
# right name.  `Path(__file__).with_name("run_five_arms.py")` audits whatever sits
# next to THIS script, which for a copied or frozen suite is a different file than
# the one under test -- the "grep a file is not grep its import target" failure.
# R.__file__ cannot diverge from the module whose functions these checks call.
RUNNER_SRC = Path(R.__file__).resolve()


def runner_source() -> str:
    src = RUNNER_SRC.read_text()
    if not src.strip():
        raise CheckFailure(f"runner source {RUNNER_SRC} is empty; a scan over it cannot fail")
    return src

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
    #     The scan target must be the file the tested module was imported FROM.
    if break_mode == "scan_wrong_file":
        # BREAK 3f: the scan reads a sibling with the right NAME instead of the
        # imported module's own file, which is what a copied/frozen suite does.
        # Simulated by pointing at a file that is provably not R.__file__.
        scanned = Path(__file__).resolve()
    else:
        scanned = RUNNER_SRC
    if scanned != Path(R.__file__).resolve():
        raise CheckFailure(
            f"the source scan reads {scanned} but the checks import run_five_arms from "
            f"{Path(R.__file__).resolve()}; grepping one file proves nothing about the other"
        )
    src = runner_source()
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
    src = runner_source()
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


# =============================================================== CHECK 8 ======

def check8(break_mode: Optional[str] = None) -> None:
    """An empty or unparseable generation scores WRONG and is flagged INVALID.

    This check exists because the runner's first scorer failed it.  ``letter_of``
    tested ``t.upper() in LETTERS`` where ``LETTERS == "ABCDEFGH"`` is a STRING, so
    ``"" in LETTERS`` was True and ``LETTERS.index("") == 0``: an empty generation
    was scored as choice A with ``invalid=False``.  Measured on the pinned 200s that
    gave an arm that emits NOTHING 49/200 SciQ, 34/200 CQA, 37/200 RACE -- macro
    20.0% and zero invalids.  ``"AB"`` had the same hole (``LETTERS.index("AB")==0``),
    and because ``truncate_at_newline`` runs first, every generation that merely
    STARTS with a newline was truncated to "" and scored as A too.

    The distinguishing property is direction: the defect could only ever INFLATE an
    arm, and it inflates the arms that degenerate most, which are exactly the
    controls whose collapse the table is trying to detect.
    """
    fixed_p = Path(__file__).with_name("examples_fixed.json")
    fixed = json.loads(fixed_p.read_text())
    tasks = sorted(fixed["tasks"])
    print(f"    DENOMINATOR tasks={len(tasks)} rows="
          f"{sum(fixed['tasks'][t]['n_examples'] for t in tasks)}")
    if not tasks:
        raise CheckFailure("no pinned tasks: cannot fail")

    def letter_of(text: str) -> Optional[str]:
        if break_mode == "substring_letters":
            # BREAK 8a: the ORIGINAL buggy implementation, verbatim.
            t = text.strip()
            if len(t) <= 2 and t.upper() in R.LETTERS:
                return t.upper()
            return None
        return R.letter_of(text)

    def index_of(lt: Optional[str]) -> Optional[int]:
        if break_mode in ("substring_letters", "str_index"):
            # BREAK 8b: membership fixed but str.index kept, which still maps
            # "" and "AB" to 0.
            return None if lt is None else R.LETTERS.index(lt)
        return R.letter_index(lt)

    # (a) THE INVARIANT: every return is None or EXACTLY ONE character.  The
    #     original code violated it twice -- letter_of("") == "" (length 0) and
    #     letter_of("AB") == "AB" (length 2) -- and LETTERS.index() mapped both to
    #     0.  Note "AB" -> "A" IS legitimate: it is the documented glued-bare-label
    #     rule that also turns "BQuestion:" into "B" (score_gen_match.py:82-84,
    #     calibrated at 74-78% correct against a 24-37% chance baseline).  What was
    #     never legitimate is a 2-character "letter" reaching str.index.
    probe = ["", "   ", "\n", "\n\n", "\t", "A", "AB", "CD", "(B) x", "BQuestion:", "zz", "hello"]
    shapes = {s: letter_of(s) for s in probe}
    bad_shape = {s: v for s, v in shapes.items() if v is not None and len(v) != 1}
    print(f"    DENOMINATOR strings probed={len(probe)}; returns that are neither None nor a "
          f"single character={len(bad_shape)} {bad_shape}")
    if bad_shape:
        raise CheckFailure(
            f"letter_of returned a non-single-character value for {bad_shape}; `x in LETTERS` is "
            f"a SUBSTRING test on \"ABCDEFGH\", so \"\" and \"AB\" both pass it and "
            f"LETTERS.index maps both to 0 (= choice A)"
        )

    # (b) THE INDEXER, probed DIRECTLY on malformed input.  Feeding it only what
    #     the fixed letter_of emits makes this sub-assertion vacuous -- the break
    #     'str_index' passed for exactly that reason and was reported as a broken
    #     instrument.  letter_index is a separate function with its own contract:
    #     it must reject anything that is not one of the eight letters, because a
    #     future caller can hand it "" or "AB" again.
    malformed: list[Optional[str]] = [None, "", "  ", "AB", "ABC", "a", "I", "1", "\n"]
    # 'a' and 'I' are the two interesting near-misses: lowercase (letter_of always
    # upper-cases, so a lowercase arrival means a caller bypassed it) and the first
    # letter OUTSIDE A-H (index() would raise, not return 0 -- also not None).
    must_reject = [v for v in malformed if v not in ("a",)]
    offenders = {}
    for v in must_reject:
        try:
            got = index_of(v)
        except ValueError:
            got = "raised ValueError"
        if got is not None and got != "raised ValueError":
            offenders[repr(v)] = got
    print(f"    DENOMINATOR malformed letters probed={len(must_reject)}; "
          f"mapped to an option index={len(offenders)} {offenders}")
    if offenders:
        raise CheckFailure(
            f"letter_index mapped malformed input to a real option index: {offenders}. "
            f"str.index on \"ABCDEFGH\" returns 0 for \"\" and for any substring like \"AB\", "
            f"so a bad shape becomes choice A instead of an invalid answer."
        )
    if index_of("A") != 0 or index_of("H") != 7:
        raise CheckFailure(
            f"letter_index no longer maps real letters (A->{index_of('A')}, H->{index_of('H')}); "
            f"rejecting everything would satisfy the test above for the wrong reason"
        )

    # (c) empty and whitespace-only generations must not resolve to ANY option.
    degenerate = ["", "   ", "\n", "\n\n", "\t"]
    resolved = {s: index_of(letter_of(s)) for s in degenerate}
    bad = {s: v for s, v in resolved.items() if v is not None}
    print(f"    DENOMINATOR empty/whitespace strings tested={len(degenerate)}; "
          f"resolved to an option index={len(bad)} {bad}")
    if bad:
        raise CheckFailure(
            f"{len(bad)}/{len(degenerate)} empty or whitespace-only generation(s) resolved to an "
            f"option index instead of None: {bad}. An empty prediction is being scored as a real "
            f"answer (index 0 = choice A), so it wins whenever gold sits at A."
        )
    # the glued rule must still work, or (a)+(b) could be satisfied by a parser
    # that rejects everything.
    if letter_of("BQuestion:") != "B" or letter_of("(C) x") != "C":
        raise CheckFailure(
            f"the glued/parenthesised label rules stopped working "
            f"(BQuestion:->{letter_of('BQuestion:')!r}, (C) x->{letter_of('(C) x')!r}); "
            f"rejecting everything would pass the empty-string test for the wrong reason"
        )
    # (b) end to end, over the REAL pinned rows: an empty arm must score 0 and be
    #     invalid on every row.  This is the assertion with the 600-row denominator.
    n_correct = n_invalid = n_rows = 0
    per_task = {}
    for t in tasks:
        c = inv = 0
        for e in fixed["tasks"][t]["examples"]:
            if break_mode == "score_empty_as_a":
                # BREAK 8c: the scorer is fine but the RUNNER maps a None pred to 0
                # before recording, which is the same defect one layer up.
                ok, invalid = (0 == _gold_index(e), False)
            else:
                ok, invalid, _pi = R.score_row("", e["gold"], e["choices"], False)
            c += bool(ok)
            inv += bool(invalid)
        per_task[t] = (len(fixed["tasks"][t]["examples"]), c, inv)
        n_rows += len(fixed["tasks"][t]["examples"])
        n_correct += c
        n_invalid += inv
    print(f"    DENOMINATOR empty-generation rows scored={n_rows}; "
          f"per task {{" + ", ".join(f"{t}:{v[1]}/{v[0]}" for t, v in per_task.items()) + "}")
    if n_rows == 0:
        raise CheckFailure("zero rows scored: cannot fail")
    if n_correct != 0:
        raise CheckFailure(
            f"an arm that generates NOTHING scores {n_correct}/{n_rows} correct "
            f"({100 * n_correct / n_rows:.1f}%): " +
            ", ".join(f"{t} {v[1]}/{v[0]}" for t, v in per_task.items()) +
            ". Empty predictions are being credited."
        )
    if n_invalid != n_rows:
        raise CheckFailure(
            f"only {n_invalid}/{n_rows} empty generations were flagged invalid; the rest were "
            f"recorded as genuine answers, so the invalid count understates the failure"
        )
    # (c) the scorer must still WORK: a correct answer must be scored correct, or
    #     "everything is invalid" would pass (a). Negative control of the fix.
    n_ok = 0
    for t in tasks:
        for e in fixed["tasks"][t]["examples"][:20]:
            ok, invalid, _ = R.score_row(e["gold"], e["gold"], e["choices"], False)
            n_ok += bool(ok) and not invalid
    print(f"    DENOMINATOR gold-as-prediction rows={20 * len(tasks)} scored correct={n_ok}")
    if n_ok != 20 * len(tasks):
        raise CheckFailure(
            f"feeding the GOLD answer back scores only {n_ok}/{20 * len(tasks)}; the scorer has "
            f"been relaxed into rejecting everything, which would pass the empty-string test "
            f"for the wrong reason"
        )
    print(f"    empty generation -> 0/{n_rows} correct, {n_invalid}/{n_rows} invalid; "
          f"gold -> {n_ok}/{20 * len(tasks)} correct")


def _gold_index(e: dict[str, Any]) -> int:
    ng = R.normalize_text(e["gold"])
    for i, c in enumerate(e["choices"]):
        if R.normalize_text(c) == ng:
            return i
    mi = R.match_choice(e["gold"], e["choices"])
    if mi is None:
        raise CheckFailure(f"gold {e['gold']!r} not locatable in {e['choices']!r}")
    return mi


# =============================================================== CHECK 9 ======

def check9(break_mode: Optional[str] = None) -> None:
    """The per-example DDIM seed stream is collision-free WITHIN a seed, and the
    recorded ``seed_stream`` label matches the arithmetic actually used.

    The original stream ``seed + 100003*(task+1) + i`` is collision-free within one
    seed but NOT across seeds: (s, i) and (s+1, i-1) collide, so seeds 42/43/44
    reuse 597/600, 597/600 and 594/600 of the same integers at relabelled
    positions, and run-to-run spread across seeds is understated.  That is a real
    property, so this check does not forbid it -- it requires that the row records
    WHICH stream was used, and that the multiplicative option really decouples.
    """
    n_tasks, n_ex, seeds = 3, 200, (42, 43, 44)
    print(f"    DENOMINATOR seeds={len(seeds)} tasks={n_tasks} examples={n_ex} "
          f"draws_per_seed={n_tasks * n_ex}")

    def stream(seed: int, kind: str) -> list[int]:
        if break_mode == "stream_ignores_label":
            # BREAK 9a: both labels compute the SAME thing, so the flag is inert and
            # "multiplicative" rows are silently additive.
            kind = "additive"
        return [R.example_seed(seed, kind, t, i) for t in range(n_tasks) for i in range(n_ex)]

    for kind in ("additive", "multiplicative"):
        for s in seeds:
            v = stream(s, kind)
            if break_mode == "within_seed_collision" and kind == "additive":
                # BREAK 9b: a stream that collides WITHIN one seed, which would make
                # two different examples share one z. Must fire for either label.
                v = [R.example_seed(s, kind, t, 0) for t in range(n_tasks) for _i in range(n_ex)]
            if len(set(v)) != len(v):
                raise CheckFailure(
                    f"{kind} seed {s}: only {len(set(v))} distinct integers over {len(v)} "
                    f"examples; two examples would share one DDIM noise draw"
                )
    add = {s: set(stream(s, "additive")) for s in seeds}
    mul = {s: set(stream(s, "multiplicative")) for s in seeds}
    pairs = [(42, 43), (43, 44), (42, 44)]
    a_ov = {p: len(add[p[0]] & add[p[1]]) for p in pairs}
    m_ov = {p: len(mul[p[0]] & mul[p[1]]) for p in pairs}
    print(f"    DENOMINATOR cross-seed pairs={len(pairs)}; additive overlap={a_ov}; "
          f"multiplicative overlap={m_ov}")
    if max(m_ov.values()) != 0:
        raise CheckFailure(
            f"the 'multiplicative' stream still shares {max(m_ov.values())}/{n_tasks * n_ex} "
            f"integers between seeds, so it does not decouple the runs: {m_ov}"
        )
    if max(a_ov.values()) == 0:
        raise CheckFailure(
            "the 'additive' stream shows 0 cross-seed overlap, contradicting the measured "
            "597/600; the two labels are computing the same thing, so seed_stream is inert"
        )
    # the runner must RECORD which one it used, in run_meta and in every row.
    src = runner_source()
    if break_mode == "not_recorded":
        src = src.replace('"seed_stream": args.seed_stream,', '')
    rec = [i + 1 for i, ln in enumerate(src.splitlines())
           if '"seed_stream": args.seed_stream' in ln and not ln.strip().startswith("#")]
    print(f"    DENOMINATOR runner lines recording seed_stream={len(rec)} at {rec}")
    if len(rec) < 2:
        raise CheckFailure(
            f"seed_stream is recorded on {len(rec)} line(s); it must appear in run_meta AND in "
            f"the per-row record, or a reader cannot tell which stream produced a number"
        )
    print(f"    additive overlap {a_ov[(43, 44)]}/{n_tasks * n_ex} across adjacent seeds, "
          f"multiplicative 0; label recorded on {len(rec)} lines")


CHECKS: dict[int, tuple[str, Callable[..., None]]] = {
    1: ("cache deep-copied before mutation; fields+counters recorded", check1),
    2: ("A0 and A1 agree on principal memory at the write boundary", check2),
    3: ("A2/A3/A4 share effective lambda/gamma; no hardcoded override", check3),
    4: ("A3 has no fixed point and z_donor != z_recipient as tensors", check4),
    5: ("same input+arm+seed reproduces identical logits (tol 0.0)", check5),
    6: ("prefix replay length and token ids identical across all arms", check6),
    7: ("first-token logits recorded; shifted-CE drop is not licensed here", check7),
    8: ("an empty/unparseable generation scores WRONG and INVALID", check8),
    9: ("per-example seed stream is collision-free and its label is recorded", check9),
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
        ("runner_zero_tol_source",  "the always-firing guard is back in the runner"),
        ("scan_wrong_file",         "the scan reads a sibling, not R.__file__")],
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
    8: [("substring_letters",  "the original `t in LETTERS` substring bug"),
        ("str_index",          "membership fixed but LETTERS.index('') == 0 kept"),
        ("score_empty_as_a",   "runner maps a None prediction to index 0")],
    9: [("stream_ignores_label",    "both labels compute the additive stream"),
        ("within_seed_collision",   "two examples share one DDIM seed"),
        ("not_recorded",            "seed_stream absent from the row record")],
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

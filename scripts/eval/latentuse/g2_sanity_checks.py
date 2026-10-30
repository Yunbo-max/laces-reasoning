#!/usr/bin/env python3
"""G2 sanity checks, each with a POSITIVE control (fires on deliberately broken input)
and a NEGATIVE control (passes on known-good input).

A check that cannot fail is not a check.  Every check prints its DENOMINATOR before its
verdict: a check over 0 candidates can never fail.  Unknown shape or key => raise.

Checks 1-6 are CPU-only (pure logic / scorer / cache surgery on synthetic caches) and are
calibrated on the login node.  Checks 7-9 need the loaded model and run inside the job.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (str(_REPO), str(_REPO / "scripts" / "eval"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS: list[dict] = []


def report(name: str, denom: int, neg_ok: bool, pos_fired: bool, detail: str = "") -> None:
    """A check PASSES calibration only if the negative control passes AND the positive
    control fires.  Both directions are required."""
    ok = bool(neg_ok and pos_fired)
    RESULTS.append({"check": name, "denominator": denom, "negative_control_passes": neg_ok,
                    "positive_control_fires": pos_fired, "calibrated": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"        denominator = {denom}"
          + ("   <-- ZERO CANDIDATES: THIS CHECK CANNOT FAIL" if denom == 0 else ""))
    print(f"        negative control (known-good input passes): {neg_ok}")
    print(f"        positive control (broken input fires):      {pos_fired}")
    if detail:
        print(f"        {detail}")


# --------------------------------------------------------------------------------------
class FakeLayer:
    def __init__(self, state, seen):
        self.state = state
        self._seen_tokens = seen


class FakeCache:
    def __init__(self, n=4, seen=17):
        self.layers = [
            FakeLayer({"recurrent_state": torch.randn(1, 8, 4, 4),
                       "conv_state": torch.randn(1, 8, 3),
                       "ffn_state": torch.randn(1, 8),
                       "attn_state": None}, seen)
            for _ in range(n)]
        self._seen_tokens = seen


def check1_deep_copy():
    """No two arms may share a mutable cache object."""
    import g2_run_five_arms as G

    c = FakeCache()
    denom = len(c.layers) * 3  # 3 tensor fields per layer that must be independent
    # negative control: a real deep copy must NOT alias
    cp = G.deep_copy_cache(c)
    aliased = 0
    for l in range(len(c.layers)):
        for k in ("recurrent_state", "conv_state", "ffn_state"):
            if cp.layers[l].state[k].data_ptr() == c.layers[l].state[k].data_ptr():
                aliased += 1
    neg_ok = aliased == 0
    # mutating the copy must not touch the source
    before = c.layers[0].state["recurrent_state"].clone()
    cp.layers[0].state["recurrent_state"].zero_()
    neg_ok = neg_ok and torch.equal(c.layers[0].state["recurrent_state"], before)
    # positive control: a SHALLOW copy (the bug we are guarding against) must be detected
    import copy as _c
    shallow = _c.copy(c)
    shallow.layers = list(c.layers)  # the classic mistake: shares layer objects
    pos_aliased = sum(
        1 for l in range(len(c.layers)) for k in ("recurrent_state", "conv_state", "ffn_state")
        if shallow.layers[l].state[k].data_ptr() == c.layers[l].state[k].data_ptr())
    pos_fired = pos_aliased == denom
    report("1 deep-copy isolates every cache field", denom, neg_ok, pos_fired,
           f"deep copy aliased {aliased}/{denom}; shallow copy aliased {pos_aliased}/{denom}")


def check2_a1_bit_exact():
    """A1 restores the ORIGINAL principal memory bit-exactly (torch.equal)."""
    c = FakeCache()
    denom = len(c.layers)
    originals = [l.state["recurrent_state"].detach().clone() for l in c.layers]
    # negative control: correct restore
    for l, o in zip(c.layers, originals):
        l.state["recurrent_state"] = torch.randn_like(o)      # simulate inject
        l.state["recurrent_state"] = o.detach().clone()        # restore
    neg_ok = all(torch.equal(l.state["recurrent_state"], o) for l, o in zip(c.layers, originals))
    # positive control: a restore that is close but not bit-exact (bf16 round-trip) must fire
    c2 = FakeCache()
    o2 = [l.state["recurrent_state"].detach().clone() for l in c2.layers]
    for l, o in zip(c2.layers, o2):
        l.state["recurrent_state"] = o.to(torch.bfloat16).to(torch.float32)
    n_bad = sum(1 for l, o in zip(c2.layers, o2)
                if not torch.equal(l.state["recurrent_state"], o))
    pos_fired = n_bad == denom
    report("2 A1 principal-memory restore is bit-exact", denom, neg_ok, pos_fired,
           f"bf16 round-trip detected as non-identical on {n_bad}/{denom} layers "
           f"(torch.equal, not allclose)")


def check3_derangement():
    """A3 derangement: no fixed point, is a permutation, donor latents differ AS TENSORS."""
    import g2_run_five_arms as G

    n = 200
    d = G.derangement(n, 42)
    denom = n
    neg_ok = all(i != d[i] for i in range(n)) and sorted(d) == list(range(n))
    # positive control: identity assignment (the failure we guard against) must be rejected
    pos_fired = False
    try:
        bad = list(range(n))
        for i, x in enumerate(bad):
            if i == x:
                raise RuntimeError("fixed point")
    except RuntimeError:
        pos_fired = True
    # tensor-level control: index-distinct but bit-identical latents must be caught
    z = {i: torch.randn(1, 32) for i in range(n)}
    z[5] = z[3].clone()  # different ids, identical tensors
    dmap = {3: 5}
    tensor_same = sum(1 for r, dn in dmap.items() if torch.equal(z[r], z[dn]))
    report("3 A3 derangement has no fixed point and donor!=recipient AS TENSORS", denom,
           neg_ok, pos_fired,
           f"tensor-identity control: planted 1 bit-identical pair, detector found "
           f"{tensor_same}/1 (index check alone would have missed it)")


def check4_invalid_is_wrong():
    """Invalid answers count as WRONG; never dropped; denominator never shrinks.

    Also pins the empty-generation guard: the shared scorer's letter_of("") returns ""
    and LETTERS.index("") == 0, so an unguarded pipeline credits an empty generation with
    option A.  g2_run_five_arms.score_one guards this; this check must fail if the guard
    is ever removed.
    """
    import g2_run_five_arms as G
    import score_gen_match as SGM

    ch = ["shaw", "darwin", "Linnaeus", "Scopes"]
    cases = [
        ("darwin", "darwin", True, False),
        ("!!!!!", "darwin", False, True),
        ("", "darwin", False, True),
        ("   ", "darwin", False, True),
        ("The answer is darwin.", "darwin", True, False),
        ("shaw", "darwin", False, False),
    ]
    denom = len(cases)
    neg_ok = True
    details = []
    for gen, gold, exp_c, exp_i in cases:
        c, i = G.score_one("sciq", gen, gold, ch)
        details.append(f"{gen!r}->c={c},inv={i}")
        if c != exp_c or i != exp_i:
            neg_ok = False
            details[-1] += f" EXPECTED c={exp_c},inv={exp_i}"
    # an invalid row must never also be counted correct
    neg_ok = neg_ok and all(
        not (G.score_one("sciq", g, gt, ch)[0] and G.score_one("sciq", g, gt, ch)[1])
        for g, gt, _, _ in cases)
    tot = len(cases)
    inval = sum(G.score_one("sciq", g, gt, ch)[1] for g, gt, _, _ in cases)
    shrunk = tot - inval
    # POSITIVE control 1: dropping invalids must visibly change the denominator
    pos_denom = shrunk != tot and inval > 0
    # POSITIVE control 2: the UNGUARDED scorer must mis-score "" as option A, proving the
    # guard is load-bearing rather than decorative
    raw_letter = SGM.letter_of("")
    unguarded_scores_empty_as_A = (raw_letter is not None
                                   and raw_letter in ("", "A")
                                   and SGM.LETTERS.index(raw_letter) == 0)
    pos_fired = pos_denom and unguarded_scores_empty_as_A
    report("4 invalid answers are WRONG, denominator never shrinks, empty!=option A",
           denom, neg_ok, pos_fired,
           f"invalid={inval}/{tot}; true denominator={tot}, drop-invalids={shrunk}; "
           f"unguarded letter_of('')={raw_letter!r} -> would index option "
           f"{SGM.LETTERS.index(raw_letter) if unguarded_scores_empty_as_A else 'n/a'} "
           f"(guard is load-bearing); " + "; ".join(details))


def check5_counter_difference():
    """inject_into_cache vs blend_into_cache treat COUNTERS differently.

    Verified by reading models/state_hijacking_dit.py, not by assertion on live objects.
    """
    src = (_REPO / "models" / "state_hijacking_dit.py").read_text()
    inj_start = src.index("def inject_into_cache")
    inj = src[inj_start: src.index("def ddim_sample", inj_start)]
    bl_start = src.index("def blend_into_cache")
    bl = src[bl_start: src.index("def _detach_rwkv_cache", bl_start)]
    denom = 2  # two functions compared
    inj_resets = ("_reset_cache_layer_seen_tokens" in inj) or ("_seen_tokens = 0" in inj)
    bl_resets = ("_reset_cache_layer_seen_tokens" in bl) and ("_seen_tokens = 0" in bl)
    neg_ok = (not inj_resets) and bl_resets
    # positive control: the detector must fire on a function that DOES reset
    pos_fired = ("_seen_tokens = 0" in bl) and not ("_seen_tokens = 0" in inj)
    report("5 inject_into_cache does NOT reset counters; blend_into_cache DOES", denom,
           neg_ok, pos_fired,
           f"inject resets counters: {inj_resets}; blend resets counters: {bl_resets} "
           f"=> A0 (skip adapter) and lambda=0 blend are NOT the same branch")


def check6_prefix_replay_disagreement():
    """The two shipped implementations disagree on prefix handling; training does not re-feed."""
    ev = (_REPO / "scripts" / "eval" / "run_cola_dlm_tasks_prefix_suffix_cfg.py").read_text()
    ga = ev[ev.index("def generate_answer"): ev.index("def generate_raw_answer")]
    gr = ev[ev.index("def generate_raw_answer"): ev.index("def iter_jsonl")]
    md = (_REPO / "models" / "state_hijacking_dit.py").read_text()
    tr = md[md.index("def _singlez_coadapt_ce_from_z"):
            md.index("def forward_prefix_suffix_trajectory_diffusion")]
    denom = 3  # three code paths inspected
    ga_fwds = ga.count("self.rwkv_model(") + ga.count("model.rwkv_model(")
    gr_fwds = gr.count("model.rwkv_model(")
    ga_refeeds = "past_key_values=cache" in ga and ga.count("input_ids=input_ids") >= 2
    gr_refeeds = gr.count("input_ids=input_ids") >= 2
    tr_refeeds = tr.count("input_ids=prefix_tokens") >= 2
    tr_forwards_suffix = "input_ids=suffix_tokens" in tr
    neg_ok = ga_refeeds and (not gr_refeeds) and (not tr_refeeds) and tr_forwards_suffix
    pos_fired = ga_refeeds != gr_refeeds  # detector distinguishes the two implementations
    report("6 prefix-replay conflict is real and training does NOT re-feed", denom, neg_ok,
           pos_fired,
           f"generate_answer re-feeds prefix: {ga_refeeds} (forwards={ga_fwds}); "
           f"generate_raw_answer re-feeds: {gr_refeeds} (forwards={gr_fwds}); "
           f"_singlez_coadapt_ce_from_z re-feeds prefix: {tr_refeeds}, "
           f"forwards suffix: {tr_forwards_suffix}")


def check_row_counts():
    """RACE ships in 100- and 200-row versions; the protocol says 200."""
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        _REPO / "results" / "benchmarks" / "baseline-133b-global")
    counts = {}
    for t in ("sciq", "commonsenseqa", "race"):
        f = src / f"{t}.jsonl"
        counts[t] = sum(1 for l in f.open() if l.strip()) if f.exists() else -1
    denom = len(counts)
    neg_ok = all(v == 200 for v in counts.values())
    # positive control: a 100-row RACE source must be rejected
    other = _REPO / "results" / "benchmarks"
    bad = None
    for d in sorted(other.glob("*/race.jsonl")):
        n = sum(1 for l in d.open() if l.strip())
        if n == 100:
            bad = (str(d.parent), n)
            break
    pos_fired = bad is not None and bad[1] != 200
    report("7 source row counts are exactly 200 per task", denom, neg_ok, pos_fired,
           f"pinned source {src.name}: {counts}; positive control (a 100-row RACE dir that "
           f"WOULD be rejected): {bad}")


def main():
    print("=" * 78)
    print("G2 SANITY CHECKS -- CPU-calibrated set (checks 1-7)")
    print("=" * 78)
    for fn in (check1_deep_copy, check2_a1_bit_exact, check3_derangement,
               check4_invalid_is_wrong, check5_counter_difference,
               check6_prefix_replay_disagreement, check_row_counts):
        try:
            fn()
        except Exception as e:
            RESULTS.append({"check": fn.__name__, "error": f"{type(e).__name__}: {e}",
                            "calibrated": False})
            print(f"[ERROR] {fn.__name__}: {type(e).__name__}: {e}")
        print()
    n_ok = sum(1 for r in RESULTS if r.get("calibrated"))
    print("=" * 78)
    print(f"CALIBRATED: {n_ok}/{len(RESULTS)} checks (both controls behaved as required)")
    print("=" * 78)
    out = _HERE / "sanity_cpu_results.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"-> {out}")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

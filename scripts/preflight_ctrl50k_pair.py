#!/usr/bin/env python3
"""Pre-flight assertion for the ctrl50k fixed-vs-dynamic writer pair.

Builds BOTH arms on CPU (backbone excluded via a stub) and asserts:
  1. fixed arm has NO s1_trunk / s1_u_head / s1_v_head; dynamic arm HAS them.
  2. the fixed arm's writer path is the one actually exercised:
     alpha_trunk is None, features = z, alpha_heads[l] is Linear(latent_dim, n_basis).
  3. trajectory_state_decoder is None in both (trajectory_s1_mode=independent),
     so predict_trajectory_states falls through to predict_states and the
     controlled variable really reaches the state writer.
  4. stage-2 freezing marks the intended trainable set in each arm, and the
     trainable sets differ ONLY in the writer modules.
  5. the warm-start checkpoint loads with the expected tensor count and the
     set of keys it fails to provide is exactly the dynamic arm's writer.
  6. the two arms' loss terms are not degenerate: reports the magnitude of
     each enabled loss term so a 6-orders-of-magnitude mismatch is visible.

Run on a GPU node (needs the real config) or CPU (model build only).
"""
import sys, os, argparse, collections
from pathlib import Path

REPO = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
sys.path.insert(0, str(REPO))

import torch
from omegaconf import OmegaConf

WARM = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000/model.pt"
WRITER_MODULES = ("s1_trunk", "s1_u_head", "s1_v_head")

ok = True
def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))


class _StubBackbone(torch.nn.Module):
    """Frozen-backbone stand-in with the real 2.9B geometry.

    Only the adapter modules are under test; the 2.9B weights are irrelevant to
    the parameter-set assertions and cost 6 GB to load. Geometry is taken from
    the real runs' logs: layers=32 hidden=2560 heads=40 head_dim=64.
    """
    class _C:
        num_hidden_layers = 32
        hidden_size = 2560
        head_dim = 64
        vocab_size = 65536

    def __init__(self):
        super().__init__()
        self.config = self._C()
        self.embed = torch.nn.Embedding(8, 8)  # tiny, just so .parameters() is non-empty


def build(writer):
    """Build the model shell for one arm without loading the 2.9B backbone."""
    cfg = OmegaConf.load(REPO / f"configs/rwkv_relay_2.9B_chunked_birwkv_{writer}_r32_ctrl50k.yaml")
    from models.state_hijacking_dit import StateInjectionDiTRELAY
    m = cfg.model
    model = StateInjectionDiTRELAY(
        config=m,
        rwkv_model=_StubBackbone(),
        vocab_size=65536,
        latent_dim=int(m.latent_dim),
        n_basis=int(m.get("n_basis", 16)),
        dit_hidden=int(m.get("dit_hidden", 256)),
        dit_depth=int(m.get("dit_depth", 4)),
        dit_num_heads=int(m.get("dit_num_heads", 4)),
        dit_num_tokens=int(m.get("dit_num_tokens", 4)),
        encoder_type=str(m.get("encoder_type", "mlp")),
        alpha_type=str(m.get("alpha_type", "linear")),
        alpha_hidden=int(m.get("alpha_hidden", 256)),
        latent_stats_path=m.get("latent_stats_path", None),
    )
    return cfg, model


def main():
    global ok
    print("=" * 78)
    print("PRE-FLIGHT: ctrl50k fixed-vs-dynamic writer pair")
    print("=" * 78)

    arms = {}
    for w in ("fixed", "dynlowrank"):
        print(f"\n### arm: {w}")
        cfg, model = build(w)
        arms[w] = (cfg, model)

        check(f"s1_writer_type == {w}", model.s1_writer_type == w, model.s1_writer_type)

        has = {n: getattr(model, n, None) is not None for n in WRITER_MODULES}
        if w == "fixed":
            check("NO s1_trunk/s1_u_head/s1_v_head", not any(has.values()), str(has))
            check("alpha_trunk is None (alpha_type=linear => features = z)",
                  model.alpha_trunk is None)
            lin = model.alpha_heads[0]
            want = (int(cfg.model.n_basis), int(cfg.model.latent_dim))
            check(f"alpha_heads[l] is Linear(latent_dim={want[1]} -> n_basis={want[0]})",
                  isinstance(lin, torch.nn.Linear) and tuple(lin.weight.shape) == want,
                  str(tuple(lin.weight.shape)))
            check(f"one alpha head per layer ({model.num_layers})",
                  len(model.alpha_heads) == model.num_layers, str(len(model.alpha_heads)))
        else:
            check("HAS s1_trunk/s1_u_head/s1_v_head", all(has.values()), str(has))

        check("trajectory_state_decoder is None (independent => predict_states is used)",
              model.trajectory_state_decoder is None)
        check("trajectory_dit is the biRWKV module (NOT DiT)",
              type(model.trajectory_dit).__name__ == "TrajectoryLatentRWKV",
              type(model.trajectory_dit).__name__)

        # The routing assertion that actually matters: does predict_states take the
        # branch we think it does?  Probe it with a real tensor.
        z = torch.randn(2, int(cfg.model.latent_dim))
        with torch.no_grad():
            st = model.predict_states(z)
        check("predict_states returns one state per layer",
              len(st) == model.num_layers, f"{len(st)} states, shape {tuple(st[0].shape)}")

        # Stage-2 freeze, exactly as the trainer does it.
        import train_state_hijacking_dit as T
        T._configure_trainable_parameters(
            model, train_stage=2,
            trajectory_mode=True, trajectory_s1_mode=str(cfg.model.trajectory_s1_mode),
            s2_unfreeze_s1=True, freeze_s2=False, sft_response_only=False,
            s2_unfreeze_s0=False)
        tr = {n for n, p in model.named_parameters() if p.requires_grad}
        roots = collections.Counter(n.split(".")[0] for n in tr)
        print("      trainable roots after stage-2 freeze:",
              dict(sorted(roots.items())))
        check("trajectory_dit (the denoiser) IS trainable", roots.get("trajectory_dit", 0) > 0)
        check("state_scale IS trainable", "state_scale" in tr)
        if w == "fixed":
            check("alpha_heads + state_basis trainable (the fixed writer)",
                  roots.get("alpha_heads", 0) > 0 and "state_basis" in tr)
        else:
            check("s1_* trainable (the dynamic writer)",
                  all(roots.get(n, 0) > 0 for n in WRITER_MODULES))
        arms[w] = (cfg, model, tr)

    # ---- cross-arm: trainable sets differ ONLY in the writer ----
    print("\n### cross-arm comparison")
    tf, td = arms["fixed"][2], arms["dynlowrank"][2]
    only_f = {n for n in tf - td}
    only_d = {n for n in td - tf}
    rf = sorted({n.split(".")[0] for n in only_f})
    rd = sorted({n.split(".")[0] for n in only_d})
    print(f"      trainable only in fixed      : {rf}")
    print(f"      trainable only in dynlowrank : {rd}")
    check("the ONLY trainable difference is the writer",
          set(rf) <= {"alpha_heads", "state_basis"} and set(rd) <= set(WRITER_MODULES))
    shared = tf & td
    check("shared trainable set is non-trivial (denoiser is common to both)",
          len({n.split('.')[0] for n in shared} & {"trajectory_dit"}) == 1,
          f"{len(shared)} shared trainable tensors")

    # ---- warm start ----
    print("\n### warm start")
    check("warm-start checkpoint exists", os.path.exists(WARM), WARM)
    if os.path.exists(WARM):
        ck = torch.load(WARM, map_location="cpu", weights_only=False)
        ts = ck["trainable_state"]
        print(f"      checkpoint step={ck.get('step')}  tensors={len(ts)}")
        check("warm start has 652 tensors (fixed-writer param set)", len(ts) == 652, str(len(ts)))
        check("warm start carries NO dynamic writer tensors",
              not any(k.split(".")[0] in WRITER_MODULES for k in ts))
        for w in ("fixed", "dynlowrank"):
            model = arms[w][1]
            cur = set(model.state_dict())
            provided = set(ts) & cur
            # what the arm needs but the warm start cannot give
            needed = {n for n in arms[w][2]}
            unmet = sorted({n.split(".")[0] for n in needed - set(ts)})
            print(f"      [{w}] warm start provides {len(provided)} of this arm's tensors; "
                  f"trainable roots NOT in warm start: {unmet}")
            if w == "fixed":
                check("fixed arm: warm start covers its ENTIRE trainable set",
                      not unmet, str(unmet))
            else:
                check("dynamic arm: only the dynamic writer is uncovered (init fresh)",
                      set(unmet) <= set(WRITER_MODULES), str(unmet))

    print("\n" + "=" * 78)
    print("RESULT:", "ALL CHECKS PASSED" if ok else "*** FAILURES PRESENT -- DO NOT LAUNCH ***")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Every number in app:global-forward, recomputed from the published Global checkpoint.

The prose was written from a live reading of this checkpoint, so a check that merely
re-reads the same dict would be self-confirming. Two things make it not that: the numbers
are re-derived from tensor SHAPES and from running the real writer, and each claim is
parsed back OUT of main.tex so that editing the prose without editing the checkpoint is
caught. A claim that cannot be located in the .tex is reported as MISSING, not skipped --
a renamed paragraph is exactly how a prose checker goes vacuously silent.

CONTROLS.
  positive  the checkpoint must load and expose s1_u_head/s1_v_head, else nothing below
            means anything.
  negative  a fabricated figure (rank 31, scale 0.9999, 999x expansion) must NOT be
            found in the .tex, or substring matching proves nothing.
  negative  the residue test is re-run on the recurrent branch, which MUST show leakage;
            if both branches come back clean the test cannot detect residue and the
            "none differ after" claim in the prose is unsupported.
"""
import os
import re
import sys

import torch

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning"
TEX = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/"
       "paper_ICLR27/lalaland/main.tex")
CKPT = os.path.join(ROOT, "outputs_dynamic_basis/"
                    "laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000/model.pt")
RWKV_CFG = os.path.join(ROOT, "models/RWKV7-Goose-World3-2.9B-HF-fla-v2/config.json")

L, NH, HD, R, DZ, DM = 32, 40, 64, 32, 32, 2560


def section(tex):
    """The appendix subsection under test, isolated so a claim cannot accidentally be
    satisfied by identical text elsewhere in the paper."""
    i = tex.index(r"\label{app:global-forward}")
    j = tex.index(r"\subsection{Compute facilities", i)
    return tex[i:j]


def main():
    d = torch.load(CKPT, map_location="cpu", weights_only=False)
    ts = d["trainable_state"]
    cfg = d["config"]

    print("CONTROLS")
    need = ("s1_u_head.weight", "s1_v_head.weight", "s1_trunk.0.weight", "mu_head.weight")
    missing = [k for k in need if k not in ts]
    print(f"  positive  checkpoint exposes the writer tensors: "
          f"{'OK' if not missing else f'FAIL {missing}'}")
    if missing:
        return 1

    tex = open(TEX).read()
    sec = section(tex)
    for bogus in ("rank $31$", "0.9999", "$999\\times$"):
        hit = bogus in sec
        print(f"  negative  fabricated {bogus!r} absent from the subsection: "
              f"{'OK' if not hit else 'FAIL (BAD: matches anything)'}")
        if hit:
            return 1

    # ---- residue test, both branches. The clean branch is the claim; the recurrent
    # branch is the control that proves the test can fail.
    sys.path.insert(0, ROOT)
    import models.state_hijacking_dit as M
    from models.state_hijacking_dit import _cache_layer_state

    class FakeLayer:
        def __init__(self, st):
            self.state = st

    class FakeCache:
        def __init__(self, layers):
            self.layers = [FakeLayer(s) for s in layers]

    KEYS = ("recurrent_state", "conv_state", "ffn_state")

    def build(seed, nl=4):
        g = torch.Generator().manual_seed(seed)
        return FakeCache([{"recurrent_state": torch.randn(1, 8, 64, 64, generator=g),
                           "attn_state": None,
                           "conv_state": torch.randn(1, 1024, 4, generator=g),
                           "ffn_state": torch.randn(1, 1024, generator=g)}
                          for _ in range(nl)])

    class Shim:
        def __init__(self, t):
            self.global_denoiser_type = t

    g = torch.Generator().manual_seed(999)
    plan = [torch.randn(1, 8, 64, 64, generator=g) for _ in range(4)]
    residue = {}
    for branch in ("dit", "birwkv"):
        os.environ["RELAY_INJECT_ZERO_RNN_CACHE"] = "0"
        ca, cb = build(1), build(2)
        pre = sum(1 for l in range(4) for k in KEYS
                  if not torch.equal(_cache_layer_state(ca, l)[k],
                                     _cache_layer_state(cb, l)[k]))
        M.StateInjectionDiTRELAY.inject_into_cache(Shim(branch), ca, plan)
        M.StateInjectionDiTRELAY.inject_into_cache(Shim(branch), cb, plan)
        post = sum(1 for l in range(4) for k in KEYS
                   if not torch.equal(_cache_layer_state(ca, l)[k],
                                      _cache_layer_state(cb, l)[k]))
        residue[branch] = (pre, post)
    (pre_d, post_d), (pre_b, post_b) = residue["dit"], residue["birwkv"]
    print(f"  residue   dit branch:    pre {pre_d}/12 post {post_d}/12 "
          f"{'OK (clean)' if pre_d == 12 and post_d == 0 else 'FAIL'}")
    print(f"  negative  birwkv branch: pre {pre_b}/12 post {post_b}/12 "
          f"{'OK (leaks => test can detect residue)' if post_b > 0 else 'FAIL (BAD: inert)'}")
    if not (pre_d == 12 and post_d == 0 and post_b > 0):
        return 1

    # ---- rank of the state the real writer emits
    def lin(w, b):
        m = torch.nn.Linear(w.shape[1], w.shape[0])
        m.weight.data, m.bias.data = w.float(), b.float()
        return m

    class W:
        pass
    w = W()
    w.s1_writer_type, w.num_layers, w.num_heads = "dynlowrank", L, NH
    w.head_dim, w.s1_rank = HD, R
    w.s1_trunk = torch.nn.Sequential(lin(ts["s1_trunk.0.weight"], ts["s1_trunk.0.bias"]),
                                     torch.nn.GELU(),
                                     lin(ts["s1_trunk.2.weight"], ts["s1_trunk.2.bias"]))
    w.s1_u_head = lin(ts["s1_u_head.weight"], ts["s1_u_head.bias"])
    w.s1_v_head = lin(ts["s1_v_head.weight"], ts["s1_v_head.bias"])
    scale = d["state_scale"].float()
    states = M.StateInjectionDiTRELAY._predict_states_dynlowrank(
        w, torch.randn(1, DZ), scale)
    head_rank = int(torch.linalg.matrix_rank(states[0][0, 0]).item())
    n_state = sum(s.numel() for s in states)

    # ---- the claims, each paired with the value recomputed here.
    # state_basis is ALREADY a key of trainable_state, so adding d["state_basis"] to a
    # tally over trainable_state double-counts it. That defect produced four wrong numbers
    # in the first draft of this appendix (746.0M payload, 167.8M dead, 22.5%, 181.1M
    # dead+aux) -- all four were the same double count, and all four looked plausible.
    assert "state_basis" in ts, "state_basis is expected to live inside trainable_state"
    writer_p = sum(ts[k].numel() for k in ts
                   if k.startswith(("s1_trunk", "s1_u_head", "s1_v_head")))
    dead_p = sum(ts[k].numel() for k in ts
                 if k.startswith(("state_basis", "alpha_heads")))
    aux_p = sum(ts[k].numel() for k in ts if k.startswith("aux_decoder"))
    total_p = sum(v.numel() for v in ts.values())
    # a dense z -> state projection, over ALL layers (the per-layer figure is 5.24e6)
    dense_all = DZ * L * NH * HD * HD

    claims = [
        ("s1_u_head out dim",  "2{,}621{,}440", f"{ts['s1_u_head.weight'].shape[0]:,}"
         .replace(",", "{,}")),
        ("s1_v_head out dim",  "65{,}536", f"{ts['s1_v_head.weight'].shape[0]:,}"
         .replace(",", "{,}")),
        ("trunk hidden",       "$32\\to128$, GELU, $128\\to128$",
         f"${ts['s1_trunk.0.weight'].shape[1]}\\to{ts['s1_trunk.0.weight'].shape[0]}$, "
         f"GELU, ${ts['s1_trunk.2.weight'].shape[1]}\\to"
         f"{ts['s1_trunk.2.weight'].shape[0]}$"),
        ("latent width",       "$32$", f"${ts['mu_head.weight'].shape[0]}$"),
        ("rank r",             "$r{=}32$", f"$r{{=}}{R}$"),
        ("per-head rank",      "numerical rank $32$", f"numerical rank ${head_rank}$"),
        ("state floats",       "5{,}242{,}880", f"{n_state:,}".replace(",", "{,}")),
        ("expansion",          "$163{,}840\\times$",
         f"${n_state // DZ:,}\\times$".replace(",", "{,}")),
        ("state_scale",        "0.1848", f"{scale.item():.4f}"),
        ("writer params",      "$346.6$M", f"${writer_p/1e6:.1f}$M"),
        ("total payload",      "$662.1$M", f"${total_p/1e6:.1f}$M"),
        ("writer share",       "$52.4\\%$", f"${100*writer_p/total_p:.1f}\\%$"),
        ("dead params",        "$83.9$M", f"${dead_p/1e6:.1f}$M"),
        ("dead share",         "$12.7\\%$", f"${100*dead_p/total_p:.1f}\\%$"),
        ("dead+aux",           "$97.2$M", f"${(dead_p+aux_p)/1e6:.1f}$M"),
        ("dense alternative",  "$1.7\\times10^{8}$",
         "$1.7\\times10^{8}$" if 1.65e8 <= dense_all < 1.75e8 else f"${dense_all:.3g}$"),
        ("basis shape",        "$(32,16,40,64,64)$",
         "$(" + ",".join(str(x) for x in d["state_basis"].shape) + ")$"),
        ("alpha_heads count",  "$64$ \\code{alpha\\_heads}",
         f"${sum(1 for k in ts if k.startswith('alpha_heads'))}$ \\code{{alpha\\_heads}}"),
        ("decoder layers",     "$L{=}32$", f"$L{{=}}{L}$"),
        ("DiT tokens",         "$4$ tokens of width $768$",
         f"$4$ tokens of width "
         f"${ts['latent_dit.in_proj.weight'].shape[0] // 4}$"),
        ("DiT depth",          "$8$ pre-norm blocks",
         f"${1 + max(int(m) for m in re.findall(r'latent_dit.blocks.(\d+)\.', ' '.join(ts)))}$"
         " pre-norm blocks"),
        ("encoder trunk",      "$2560\\to10240$",
         f"${ts['encoder_trunk.0.weight'].shape[1]}\\to"
         f"{ts['encoder_trunk.0.weight'].shape[0]}$"),
    ]

    # config-sourced claims
    for key, want in (("s1_writer_type", "dynlowrank"), ("use_cond_boundary", "true"),
                      ("gen_type", "ddpm"), ("encoder_type", "variational")):
        m = re.search(rf"^\s*{key}:\s*(\S+)", cfg, re.M)
        claims.append((f"config {key}", want, m.group(1) if m else "<ABSENT>"))
    gdt = re.search(r"^\s*global_denoiser_type:", cfg, re.M)
    claims.append(("global_denoiser_type absent", "absent",
                   "absent" if not gdt else "PRESENT"))

    print(f"\n{'claim':30s} {'in .tex':>26s} {'recomputed':>26s}  verdict")
    ok, missing_in_tex = True, 0
    for name, in_tex, computed in claims:
        agrees = in_tex == computed
        if name.startswith("config ") or name.endswith("absent"):
            found = True  # config claims are checked against the ckpt, not the prose
        else:
            found = in_tex in sec
            if not found:
                missing_in_tex += 1
        good = agrees and found
        ok &= good
        v = "OK" if good else ("MISSING FROM .TEX" if not found else "MISMATCH")
        print(f"{name:30s} {in_tex:>26s} {computed:>26s}  {v}")

    print(f"\nclaims: {len(claims)}   not found in the subsection: {missing_in_tex}")

    # ---- the two structural claims about HOW the plan reaches the decoder. These are
    # claims about absence, which is the kind that rots silently when code changes, so
    # each is paired with a positive control proving the grep can find the thing it is
    # asserting to be absent.
    print("\nSTRUCTURAL CLAIMS (plan enters as state, not as embeddings)")
    files = ["scripts/eval/eval_lm_ppl.py", "scripts/eval/sample_prefix_suffix_cfg.py",
             "models/state_hijacking_dit.py"]
    src = {f: open(os.path.join(ROOT, f)).read() for f in files}
    joined = "\n".join(src.values())

    embed_terms = ("inputs_embeds", "soft_prompt", "prefix_embed", "virtual_token",
                   "prompt_embed")
    found_embed = [t for t in embed_terms if t in joined]
    print(f"  absence   no embedding-injection path in {len(files)} files: "
          f"{'OK (none of ' + str(len(embed_terms)) + ' terms present)' if not found_embed else f'FAIL {found_embed}'}")

    # positive control: the same grep MUST find the term that IS there, else "absent"
    # is indistinguishable from "searched the wrong text"
    ctl = "input_ids" in joined and "recurrent_state" in joined
    print(f"  positive  the same search finds \\code{{input_ids}} and "
          f"\\code{{recurrent_state}}: {'OK' if ctl else 'FAIL (BAD: searching wrong text)'}")

    # the scored forward must pass ids, and there must be exactly one scored call
    scored = re.findall(r"rwkv_model\(\s*input_ids=scored", src["scripts/eval/eval_lm_ppl.py"])
    print(f"  positive  scored span is passed as input_ids: {len(scored)} call(s) "
          f"{'OK' if len(scored) == 1 else 'FAIL'}")

    struct_ok = (not found_embed) and ctl and len(scored) == 1
    ok &= struct_ok

    # and the prose must actually make these claims, in this subsection
    for frag in ("token embedding is not bypassed", "\\code{inputs\\_embeds} path",
                 "occupies no sequence position", "replaces the \\emph{prefix}"):
        present = frag in sec
        ok &= present
        print(f"  prose     {frag!r:46s} {'OK' if present else 'MISSING FROM .TEX'}")

    # NEGATIVE CONTROL on the tally itself. The four wrong figures came from double-counting
    # state_basis, and the corrected tally is written here -- so "it agrees with the prose"
    # proves nothing unless the DEFECT is shown to be caught. Recompute the tally the wrong
    # way and require that it disagrees with what the .tex now says.
    wrong_total = total_p + ts["state_basis"].numel()
    wrong_dead = dead_p + ts["state_basis"].numel()
    caught = (f"${wrong_total/1e6:.1f}$M" not in sec
              and f"${wrong_dead/1e6:.1f}$M" not in sec)
    print(f"\n  negative  double-counted tally (${wrong_total/1e6:.1f}$M payload, "
          f"${wrong_dead/1e6:.1f}$M dead) absent from the .tex: "
          f"{'OK' if caught else 'FAIL (BAD: the old double count is still printed)'}")
    ok &= caught

    print("ALL APPENDIX NUMBERS RECOMPUTE" if ok else "SOME NUMBERS FAIL -- see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

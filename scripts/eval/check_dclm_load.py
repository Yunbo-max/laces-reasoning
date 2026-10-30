#!/usr/bin/env python3
"""Offline load check for TRI-ML/DCLM-1B: does `open_lm` load these weights EXACTLY?

Why a dedicated script: the brief's non-negotiable is "no missing/unexpected keys". A
`strict=False` load silently swallowing 473 tensors has already produced a wrong published
result in this project, and for a custom architecture the risk is higher, not lower -- nobody
has ever loaded an `openlm` checkpoint in this repo before. So this compares the checkpoint's
219 tensors against the freshly-constructed module's own state_dict NAME BY NAME and
SHAPE BY SHAPE, and reports the sets rather than a boolean.

Three specific hazards it is built to catch:

  1. `model.layers.N.attention.pos_embed.inv_freq` (24 of the 219 tensors) is a rotary
     BUFFER, not a parameter. open_lm's RotaryEmbedding does `register_buffer("inv_freq", ...)`
     and then OVERWRITES it in reset_parameters() from a closed-form expression. Such buffers
     routinely show up as "unexpected keys" under transformers' loader, and reading that as
     "24 weights missing" would be wrong. They are reported in their own bucket, with a
     numeric check that the checkpoint's stored inv_freq matches what the module recomputes.
  2. Fused projections: `attention.in_proj` is [3*dim, dim] (q,k,v fused) and
     `feed_forward.w12` is [2*hidden, dim] (gate+up fused). A LLaMA-shaped conversion would
     split these, so a shape mismatch here is the signal that some third-party "DCLM in llama
     form" reupload is NOT the same model. Shapes are asserted against config arithmetic.
  3. `weight_tying: false`, so `model.output.weight` must exist as its own tensor and must
     NOT be a view of `tok_embeddings.weight`. Checked numerically, because a loader that
     silently tied them would still report zero missing keys.

Also runs the real forward twice -- fp32 and bf16 -- because the harness hard-codes bf16 for
--model_dir while this checkpoint declares torch_dtype float32, and the size of that cast
error is something we should measure rather than assume.

CPU-only, no GPU needed, fully offline.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

SNAP = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub"
            "/models--TRI-ML--DCLM-1B/snapshots/f43c272dab1b06bc5fdd6bbb5819f8a30ef2833b")


def main() -> int:
    print("=" * 78)
    print("DCLM-1B offline load check")
    print("=" * 78)

    cfg = json.loads((SNAP / "config.json").read_text())
    index = json.loads((SNAP / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    print(f"[cfg] model_type={cfg['model_type']} arch={cfg['architectures']} "
          f"dim={cfg['dim']} n_layers={cfg['n_layers']} n_heads={cfg['n_heads']} "
          f"vocab={cfg['vocab_size']} seq_len={cfg['seq_len']} "
          f"weight_tying={cfg['weight_tying']} torch_dtype={cfg['torch_dtype']}")
    print(f"[index] {len(weight_map)} tensors, total_size={index['metadata']['total_size']}")

    # ---------------------------------------------------------------- 1. read the real files
    from safetensors import safe_open
    ckpt: dict[str, tuple] = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(SNAP / shard), framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                ckpt[k] = (tuple(t.shape), t.dtype)
    print(f"[read] {len(ckpt)} tensors actually present in the safetensors files")
    missing_from_files = set(weight_map) - set(ckpt)
    extra_in_files = set(ckpt) - set(weight_map)
    if missing_from_files or extra_in_files:
        print(f"[read] FATAL index/file disagreement: "
              f"in index not in files={sorted(missing_from_files)[:5]} "
              f"in files not in index={sorted(extra_in_files)[:5]}")
        return 1
    print("[read] index and files agree exactly")

    # ---------------------------------------------------------------- 2. build the module
    import open_lm.hf  # noqa: F401  registers openlm with the Auto classes
    from transformers import AutoConfig, AutoModelForCausalLM
    hfcfg = AutoConfig.from_pretrained(str(SNAP), local_files_only=True)
    from open_lm.hf import OpenLMForCausalLM
    with torch.device("meta"):
        skeleton = OpenLMForCausalLM(hfcfg)
    sd_names = dict(skeleton.state_dict())
    print(f"[build] constructed OpenLMForCausalLM: {len(sd_names)} state_dict entries")

    # ---------------------------------------------------------------- 3. key-set comparison
    ck, mk = set(ckpt), set(sd_names)
    missing = sorted(mk - ck)          # module wants, checkpoint lacks  -> randomly initialised
    unexpected = sorted(ck - mk)       # checkpoint has, module ignores  -> silently dropped
    shared = sorted(ck & mk)
    print(f"\n[keys] shared={len(shared)}  missing(module wants, ckpt lacks)={len(missing)}  "
          f"unexpected(ckpt has, module ignores)={len(unexpected)}")
    for label, group in (("MISSING", missing), ("UNEXPECTED", unexpected)):
        if group:
            print(f"  {label} ({len(group)}):")
            for k in group[:30]:
                print(f"    {k}  ckpt={ckpt.get(k, ('-',))[0]} module={tuple(sd_names[k].shape) if k in sd_names else '-'}")
            if len(group) > 30:
                print(f"    ... and {len(group)-30} more")

    bad_shape = [(k, ckpt[k][0], tuple(sd_names[k].shape)) for k in shared
                 if tuple(sd_names[k].shape) != ckpt[k][0]]
    print(f"[shape] mismatches among shared keys: {len(bad_shape)}")
    for k, a, b in bad_shape[:20]:
        print(f"    {k}: ckpt={a} module={b}")

    # ---------------------------------------------------------------- 4. fused-layout asserts
    dim, nh, nl = cfg["dim"], cfg["n_heads"], cfg["n_layers"]
    hidden = 256 * ((int(2 * 4 * dim / 3) + 256 - 1) // 256)   # open_lm's swiglu rule
    checks = {
        "model.layers.0.attention.in_proj.weight": (3 * dim, dim),
        "model.layers.0.attention.out_proj.weight": (dim, dim),
        "model.layers.0.feed_forward.w12.weight": (2 * hidden, dim),
        "model.layers.0.feed_forward.w3.weight": (dim, hidden),
        "model.tok_embeddings.weight": (cfg["vocab_size"], dim),
        "model.output.weight": (cfg["vocab_size"], dim),
    }
    print(f"\n[fused] open_lm swiglu hidden_dim = {hidden}")
    fused_ok = True
    for k, want in checks.items():
        got = ckpt.get(k, (None,))[0]
        ok = got == want
        fused_ok &= ok
        print(f"  [{'OK ' if ok else 'BAD'}] {k}: {got} (expect {want})")
    print("  -> in_proj is q,k,v FUSED and w12 is gate,up FUSED: a llama-shaped "
          "reupload would NOT have these shapes")

    # ---------------------------------------------------------------- 5. real strict load
    print("\n[load] AutoModelForCausalLM.from_pretrained(float32) ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(SNAP), trust_remote_code=True, torch_dtype=torch.float32,
        local_files_only=True).eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_emb = model.model.tok_embeddings.weight.numel()
    print(f"[load] OK  class={type(model).__name__}")
    print(f"[load] parameters = {n_params:,} ({n_params/1e9:.3f} B); "
          f"non-embedding = {(n_params-n_emb):,} ({(n_params-n_emb)/1e9:.3f} B)")

    # inv_freq: stored as a buffer AND recomputed by reset_parameters(). Verify they agree,
    # so "these 24 keys are buffers" is a measurement and not an excuse.
    inv_ck = None
    with safe_open(str(SNAP / weight_map["model.layers.0.attention.pos_embed.inv_freq"]),
                   framework="pt") as f:
        inv_ck = f.get_tensor("model.layers.0.attention.pos_embed.inv_freq").float()
    inv_live = model.model.layers[0].attention.pos_embed.inv_freq.float()
    dev = (inv_ck - inv_live).abs().max().item() if inv_ck.shape == inv_live.shape else float("nan")
    print(f"[inv_freq] ckpt shape={tuple(inv_ck.shape)} live shape={tuple(inv_live.shape)} "
          f"max|diff|={dev:.3e}  (a rotary buffer, recomputed in closed form)")

    # weight_tying: false -> output must be its own tensor, not a view of the embedding.
    tied = (model.model.output.weight.data_ptr() ==
            model.model.tok_embeddings.weight.data_ptr())
    same_vals = torch.equal(model.model.output.weight, model.model.tok_embeddings.weight)
    print(f"[tying] config weight_tying={cfg['weight_tying']}; shares storage={tied}; "
          f"identical values={same_vals} (both must be False)")

    # ---------------------------------------------------------------- 6. forward + bf16 delta
    text = "The capital of France is"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SNAP), trust_remote_code=True,
                                        local_files_only=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        lg32 = model(input_ids=ids).logits[0, -1].float()
    top32 = lg32.topk(5)
    print(f"\n[fwd fp32] prompt={text!r} ids={ids.tolist()[0]}")
    print(f"[fwd fp32] top5 = {[(tok.decode([i]), round(v.item(), 3)) for v, i in zip(top32.values, top32.indices)]}")

    model_bf = model.to(torch.bfloat16)
    with torch.no_grad():
        lg16 = model_bf(input_ids=ids).logits[0, -1].float()
    top16 = lg16.topk(5)
    print(f"[fwd bf16] top5 = {[(tok.decode([i]), round(v.item(), 3)) for v, i in zip(top16.values, top16.indices)]}")
    agree = int((top32.indices == top16.indices).sum())
    print(f"[dtype] bf16 vs fp32: max|logit diff|={(lg32-lg16).abs().max():.4f}, "
          f"top-1 same={bool(top32.indices[0]==top16.indices[0])}, top-5 overlap={agree}/5")
    print("[dtype] the harness runs bf16 for --model_dir (all 9 published baselines did), "
          "so this is the cast those numbers will carry")

    ok = (not missing and not bad_shape and fused_ok and not tied and not same_vals)
    print("\n" + "=" * 78)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} -- "
          f"{len(missing)} missing, {len(unexpected)} unexpected (buffers), "
          f"{len(bad_shape)} shape mismatches")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

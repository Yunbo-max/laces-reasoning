#!/usr/bin/env python3
"""What the 2.36B in STAR-LDM-unpacked is made of, module by module.

The published Params cell prints 0.4B. A 6x gap is not a rounding convention, so the
question is whether the checkpoint bundles something that is resident but not the
diffusion network -- which would make it the same accounting question the paper already
has to answer for its own two rows.
"""
import collections
import glob
import os
import sys

import torch

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222/models"


def breakdown(d, depth=1):
    cands = (sorted(glob.glob(os.path.join(d, "*.pt")))
             + sorted(glob.glob(os.path.join(d, "*.bin")))
             + sorted(glob.glob(os.path.join(d, "*.ckpt")))
             + sorted(glob.glob(os.path.join(d, "*.pth")))
             + sorted(glob.glob(os.path.join(d, "*.safetensors"))))
    print(f"=== {d}")
    for f in cands:
        if f.endswith(".safetensors"):
            from safetensors import safe_open
            g = collections.Counter()
            n = collections.Counter()
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    m = 1
                    for x in h.get_slice(k).get_shape():
                        m *= x
                    pref = ".".join(k.split(".")[:depth])
                    g[pref] += m
                    n[pref] += 1
        else:
            sd = torch.load(f, map_location="meta", mmap=True, weights_only=False)
            if not isinstance(sd, dict):
                print(f"  {os.path.basename(f)}: {type(sd).__name__}, skipped")
                continue
            # a lightning-style ckpt hides the tensors under state_dict
            if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                print(f"  (tensors live under ['state_dict']; top keys: "
                      f"{sorted(k for k in sd if k != 'state_dict')[:8]})")
                sd = sd["state_dict"]
            g = collections.Counter()
            n = collections.Counter()

            def walk(path, obj):
                if hasattr(obj, "numel"):
                    pref = ".".join(path.split(".")[:depth])
                    g[pref] += obj.numel()
                    n[pref] += 1
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        walk(f"{path}.{k}" if path else str(k), v)

            walk("", sd)
        tot = sum(g.values())
        print(f"  {os.path.basename(f)}  ({os.path.getsize(f)/1e9:.2f} GB on disk)")
        for k, v in g.most_common(25):
            print(f"    {k:40s} {v:>14,}  {v/tot*100:5.1f}%  ({n[k]} tensors)")
        print(f"    {'TOTAL':40s} {tot:>14,}  = {tot/1e9:.3f}B")
    if not cands:
        print("  no weight files")


for d in sys.argv[1:] or ["STAR-LDM-unpacked"]:
    breakdown(os.path.join(ROOT, d) if not d.startswith("/") else d,
              depth=int(os.environ.get("DEPTH", "1")))
    print()

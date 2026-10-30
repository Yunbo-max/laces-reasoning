#!/usr/bin/env python3
"""Does a printed Params cell match ANY subset of its checkpoint's modules?

Two cells disagree with the measured total by far more than a rounding: STAR-LDM prints
0.4B against 2.36B, CoLA-DLM prints 2.0B against 2.33B. A printed size that is smaller
than the file is not automatically wrong -- it may be a defensible sub-count, e.g. "the
diffusion network only, excluding a frozen pretrained encoder". That is a testable claim:
enumerate every subset of top-level modules and see whether any sum rounds to the cell.

If no subset does, the cell is not a sub-count under any partition of that checkpoint and
has to come from somewhere else. This is the same test that showed the paper's 0.06B
matched no subset of the 14 interface modules (0 hits in 2^14).
"""
import collections
import glob
import itertools
import os
import sys

import torch

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222/models"


def modules(d, want_literal):
    per = collections.Counter()
    st = sorted(glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True))
    if st:
        from safetensors import safe_open
        for f in st:
            with safe_open(f, framework="pt") as g:
                for k in g.keys():
                    m = 1
                    for x in g.get_slice(k).get_shape():
                        m *= x
                    per[k.split(".")[0]] += m
    else:
        for f in sorted(glob.glob(os.path.join(d, "*.pt"))):
            sd = torch.load(f, map_location="meta", mmap=True, weights_only=False)
            for k, v in sd.items():
                if hasattr(v, "numel"):
                    per[str(k).split(".")[0]] += v.numel()
                elif isinstance(v, dict):
                    for k2, t in v.items():
                        if hasattr(t, "numel"):
                            per[str(k).split(".")[0]] += t.numel()
    names = sorted(per, key=lambda k: -per[k])
    tot = sum(per.values())
    print(f"=== {os.path.basename(d)}  total {tot/1e9:.4f}B, {len(names)} modules")
    for n in names:
        print(f"    {n:28s} {per[n]:>14,}  {per[n]/1e9:.4f}B")

    dp = len(want_literal.split(".")[1]) if "." in want_literal else 0
    hits = []
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            s = sum(per[c] for c in combo)
            if f"{s/1e9:.{dp}f}" == want_literal:
                hits.append((s, combo))
    print(f"  subsets of {len(names)} modules summing to {want_literal}B "
          f"(at {dp}dp): {len(hits)} hit(s)")
    for s, combo in hits[:8]:
        print(f"    {s/1e9:.4f}B = {' + '.join(combo)}")
    if not hits:
        sums = sorted({sum(per[c] for c in combo)
                       for r in range(1, len(names) + 1)
                       for combo in itertools.combinations(names, r)})
        target = float(want_literal) * 1e9
        below = max((x for x in sums if x <= target), default=None)
        above = min((x for x in sums if x >= target), default=None)
        print(f"    nearest achievable sums straddling {want_literal}B: "
              f"{below/1e9 if below else 0:.4f}B and "
              f"{above/1e9 if above else 0:.4f}B")
    return hits


if __name__ == "__main__":
    # a positive control first: a cell that SHOULD match, so a run of 0 hits everywhere
    # cannot be the enumerator silently failing
    print("POSITIVE CONTROL -- the full total must always be findable as a subset:")
    modules(os.path.join(ROOT, "STAR-LDM-unpacked"), "2.36")
    print()
    for d, want in [("STAR-LDM-unpacked", "0.4"), ("cola-dlm", "2.0")]:
        modules(os.path.join(ROOT, d), want)
        print()

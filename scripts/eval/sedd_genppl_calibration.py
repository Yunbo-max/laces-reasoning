#!/usr/bin/env python3
"""Calibrate the SEDD-medium sampler against the literature.

SEDD's repo ships no reference generations, so the anchor has to be the paper.
Figure 1 of Lou et al. (arXiv:2310.16834) says SEDD "matches GPT-2 quality using
32x fewer network evaluations" and "outperforms by 6-8x when using the full 2048
steps". 2048/32 = 64 steps, which is the step count this harness runs SEDD at, so
the falsifiable prediction is that SEDD-medium at 64 steps lands next to
GPT-2-medium on generative perplexity, both scored by GPT-2-large -- the scorer
the paper uses. MDLM (arXiv:2406.07524) puts well-trained diffusion LMs at Gen.
PPL ~27-35 on this scale.

Controls, because a single number proves nothing:
  * real text scored by GPT-2-large -- a scorer sanity floor (~20-30 expected);
    if this is not sane the whole calibration is void.
  * GPT-2-medium sampled BOTH plainly (temperature 1.0, as the paper samples its
    GPT-2 baseline) and with nucleus top-p 0.95, to separate "my sampler is
    broken" from "unfiltered ancestral sampling is just bad".
  * SEDD at 64 and at 256 steps, to reproduce the paper's step/quality pareto.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sedd_native import load_sedd, mask_id_of, sedd_denoise  # noqa: E402

MODELS = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/models")
TASK_DATA = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv"
                 "/baseline/Cola-DLM/generate_task_data")
SEDD = MODELS / "SEDD-medium"
G2 = MODELS / "gpt2-medium"
G2L = MODELS / "gpt2-large"


@torch.no_grad()
def perplexity(scorer, ids, batch=4):
    total, ntok = 0.0, 0
    for i in range(0, ids.shape[0], batch):
        chunk = ids[i:i + batch]
        logits = scorer(chunk).logits[:, :-1].float()
        tgt = chunk[:, 1:]
        total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       tgt.reshape(-1), reduction="sum"))
        ntok += tgt.numel()
    return float(torch.exp(torch.tensor(total / ntok)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    step_list = ([int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3
                 else [128])
    device = torch.device("cuda")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(G2)
    scorer = AutoModelForCausalLM.from_pretrained(
        G2L, torch_dtype=torch.float32).to(device).eval()

    # ---- control 1: real text under the scorer -------------------------------
    txt = ""
    for line in (TASK_DATA / "squad.jsonl").open():
        item = json.loads(line)
        c = str(item.get("prompt") or item.get("context") or "")
        if len(c) > 1500:
            txt = c
            break
    real = tok(txt, return_tensors="pt").input_ids[:, :length].to(device)
    print(f"[control] real text  tokens={real.shape[1]}  "
          f"PPL={perplexity(scorer, real):.2f}   (sane floor ~20-30)", flush=True)

    # ---- control 2: GPT-2-medium, two sampling schemes -----------------------
    g2 = AutoModelForCausalLM.from_pretrained(
        G2, torch_dtype=torch.float32).to(device).eval()
    bos = torch.full((n, 1), 50256, dtype=torch.long, device=device)
    torch.manual_seed(0)
    plain = g2.generate(input_ids=bos, do_sample=True, temperature=1.0, top_k=0,
                        top_p=1.0, max_new_tokens=length - 1, pad_token_id=50256)
    torch.manual_seed(0)
    topp = g2.generate(input_ids=bos, do_sample=True, temperature=1.0, top_k=0,
                       top_p=0.95, max_new_tokens=length - 1, pad_token_id=50256)
    p_plain = perplexity(scorer, plain)
    p_topp = perplexity(scorer, topp)
    print(f"[control] GPT-2-medium temp1.0 top_p=1.0 -> Gen.PPL {p_plain:.2f}", flush=True)
    print(f"[control] GPT-2-medium temp1.0 top_p=0.95 -> Gen.PPL {p_topp:.2f}", flush=True)
    print(f"          e.g. plain: {tok.decode(plain[0][:70])!r}", flush=True)
    del g2
    torch.cuda.empty_cache()

    # ---- SEDD-medium at two step counts --------------------------------------
    sedd, cfg = load_sedd(SEDD, device)
    mask = mask_id_of(cfg)
    empty = torch.zeros((1, 0), dtype=torch.long, device=device)
    for steps in step_list:
        torch.manual_seed(0)
        ids = torch.tensor([sedd_denoise(sedd, empty, length, steps, mask, device, 1.0)
                            for _ in range(n)], device=device)
        p = perplexity(scorer, ids)
        print(f"[sedd]    steps={steps:4d} -> Gen.PPL {p:.2f}   ratio/gpt2plain={p/p_plain:.3f}"
              f"   e.g. {tok.decode(ids[0][:70])!r}", flush=True)

    print("=" * 70, flush=True)
    print("paper: SEDD matches GPT-2 at 32x fewer NFE (2048/32 = 64 steps); MDLM reports"
          " well-trained diffusion LMs at Gen.PPL ~27-35", flush=True)


if __name__ == "__main__":
    main()

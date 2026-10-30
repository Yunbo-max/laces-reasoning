#!/usr/bin/env python3
"""Is the Nominal column what the prose says it is -- "the size in the checkpoint's
public name"?

That is a checkable claim, and it is checkable WITHOUT trusting the row label: each row's
weight dir comes from its own run_summary.json, and for an HF cache dir the repo name is
recoverable from the `models--org--name` ancestor. So for each row:
    public repo name -> size token in that name (if any) -> compare to printed Nominal cell

Two failure modes this guards:
  * a dir that is a bare snapshot hash (GLA, Pythia, DCLM, OLMo all are) would make the
    repo name look absent if only the basename were read -> walk up to models--*;
  * "no size token in the name" is NOT the same as "the cell is wrong" (Phi-2, MDLM, SEDD,
    ELF, STAR carry no size in their names), so those are reported separately and never
    counted as mismatches.

CONTROLS. Positive: rows whose name plainly carries its size (Qwen2.5-1.5B -> 1.5,
Mamba 2.8B -> 2.8, GLA-1.3B -> 1.3) must be extracted, or the extractor is inert.
Negative: a fabricated name with no size must yield None, or the extractor is inventing.
"""
import json
import os
import re
import sys

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222"
BENCH = os.path.join(ROOT, "research/laces-reasoning/results/benchmarks")
TEX = os.path.join(ROOT, "research/DiffRwkv/paper_ICLR27/lalaland/main.tex")

ROWS = [
    ("Qwen2.5-3B", "baseline-qwen2.5-3b"),
    ("Qwen2.5-1.5B", "baseline-qwen25-15b"),
    ("Llama-3.2-3B", "baseline-llama-3.2-3b"),
    ("Phi-2", "baseline-phi-2"),
    ("Gemma-2-2B", "baseline-gemma-2-2b"),
    ("GLA-1.3B", "baseline-gla-13b"),
    ("RWKV-7 World3", "baseline-rwkv7-2.9b-hf"),
    ("RWKV-6 World2.1", "baseline-rwkv-6-world21"),
    ("Mamba 2.8B", "baseline-mamba-2.8b"),
    ("Pythia-2.8B", "baseline-pythia-28b"),
    ("DCLM-1B", "baseline-dclm-1b"),
    ("OLMo-2-1B", "baseline-olmo-2-1b"),
    ("MDLM-OWT", "baseline-mdlm-owt"),
    ("SEDD-medium", "baseline-sedd-medium"),
    ("CoLA-DLM", "baseline-cola-dlm"),
    ("ELF-L", "baseline-elf-l"),
    ("ELF-M", "baseline-elf-m"),
    ("STAR-LDM", "baseline-star-ldm"),
]

# version tokens that are NOT sizes and would otherwise be read as one
NOT_A_SIZE = re.compile(
    r"(?:qwen|llama|gemma|rwkv|olmo|phi|mamba|pythia|sedd|mdlm|elf|star|world|v)"
    r"[-_.]?\d+(?:\.\d+)?$", re.I)


def repo_name(d):
    """The public repo name for a weight dir: models--org--name if it is an HF cache
    snapshot, else the directory's own basename."""
    p = os.path.abspath(d)
    while p and p != "/":
        b = os.path.basename(p)
        if b.startswith("models--"):
            return b[len("models--"):].replace("--", "/")
        p = os.path.dirname(p)
    return os.path.basename(os.path.abspath(d))


def size_token(name):
    """The PARAMETER size the NAME advertises, in B, or None if it advertises none.

    A size token is a number immediately followed by b/m at a token boundary. Taking the
    last such token is wrong and the positive control caught it: `gla-1.3B-100B` carries
    two, and the trailing one is the TRAINING TOKEN count, not the parameter count. A
    parameter size for these checkpoints is under 10B, so when several candidates exist the
    ones at token-count magnitude are dropped; if that still leaves more than one the name
    is genuinely ambiguous and it says so rather than picking."""
    cands = []
    for m in re.finditer(r"(?<![0-9.])(\d+(?:\.\d+)?)\s*([BbMm])(?![a-zA-Z])", name):
        val = float(m.group(1))
        if m.group(2).lower() == "m":
            val /= 1000.0
        cands.append((val, m.group(0)))
    if not cands:
        return None
    small = [c for c in cands if c[0] < 10.0]
    if not small:
        return None                      # only token-count-magnitude numbers: no size
    if len(small) > 1:
        return (small[0][0], f"AMBIGUOUS{[c[1] for c in small]}")
    return small[0]


def controls():
    ok = True
    print("CONTROLS")
    for name, want in [("Qwen/Qwen2.5-1.5B", 1.5), ("state-spaces/mamba-2.8b", 2.8),
                       ("fla-hub/gla-1.3B-100B", 1.3), ("allenai/OLMo-2-0425-1B", 1.0),
                       ("EleutherAI/pythia-2.8b", 2.8)]:
        got = size_token(name)
        good = got is not None and abs(got[0] - want) < 1e-9
        ok &= good
        print(f"  positive  {name:28s} want {want:<5} got "
              f"{got[0] if got else None:<6} {'OK' if good else 'FAIL (BAD)'}")
    for name in ["microsoft/phi-2", "SEDD-medium", "ELF-L-owt-torch", "STAR-LDM-unpacked",
                 "somefamily-owt-100B"]:
        got = size_token(name)
        good = got is None
        ok &= good
        print(f"  negative  {name:28s} want None  got "
              f"{got[0] if got else None!s:<6} {'OK' if good else 'FAIL (BAD)'}")
    print(f"  -> controls {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    if not controls():
        return 1

    tex = open(TEX).read()
    i = tex.index(r"\label{tab:models}")
    j = tex.index(r"\bottomrule", i)
    printed = {}
    for line in tex[i:j].splitlines():
        s = line.strip()
        if not s.endswith("\\\\") or s.startswith(("\\multicolumn", "\\midrule",
                                                   "\\toprule", "Checkpoint")):
            continue
        c = [x.strip() for x in s[:-2].split("&")]
        if len(c) == 5:
            printed[c[0]] = (c[1], c[2])
    if len(printed) != 20:
        print(f"ABORT: parsed {len(printed)} rows, expected 20")
        return 1

    print(f"{'row':18s} {'public repo name':34s} {'name size':>9s} {'cell':>7s}  verdict")
    agree = differ = nosize = 0
    for label, sub in ROWS:
        p = os.path.join(BENCH, sub, "run_summary.json")
        md = json.load(open(p)).get("model_dir")
        rn = repo_name(md)
        st = size_token(rn)
        cell = printed[label][0]
        cell_v = re.sub(r"\$\^?\{?\\dagger\}?\$|B$", "", cell).strip()
        if st is None:
            v = "name carries no size -> cell is not from the name"
            nosize += 1
        elif f"{st[0]:g}" == cell_v or f"{st[0]:.1f}" == cell_v:
            v = "cell == name"
            agree += 1
        else:
            v = f"CELL DISAGREES WITH NAME ({cell_v} vs {st[0]:g})"
            differ += 1
        print(f"{label:18s} {rn:34s} {(f'{st[0]:g}' if st else '--'):>9s} "
              f"{cell_v:>7s}  {v}")
    print(f"\n  cell == public name: {agree}   cell != public name: {differ}   "
          f"name has no size: {nosize}   (total {agree+differ+nosize})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

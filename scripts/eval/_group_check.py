"""Does the prose's family enumeration match tab:models' own group headers?

The prose at app:models says "Every row of every table is one of these" and then lists the
families in parentheses. That is a set equality, so it is checkable. It was false: three
dense-attention Transformers (Pythia, DCLM, OLMo) sat under the table's
"Linear-attention and state-space decoders" header and appeared in no prose group at all.

Architecture is read from each row's own config.json, resolved via its run_summary.json --
never from the row label, since the label is exactly what was wrong.

CONTROLS. Positive: Mamba and GLA must be classified non-attention and Qwen attention, or
the architecture test is inert. Negative: a row deliberately moved to the wrong group must
be reported, checked by re-running the comparison on a mutated copy of the group map.
"""
import json
import os
import re
import sys

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222"
BENCH = os.path.join(ROOT, "research/laces-reasoning/results/benchmarks")
TEX = os.path.join(ROOT, "research/DiffRwkv/paper_ICLR27/lalaland/main.tex")

SUB = {
    "Qwen2.5-3B": "baseline-qwen2.5-3b", "Qwen2.5-1.5B": "baseline-qwen25-15b",
    "Llama-3.2-3B": "baseline-llama-3.2-3b", "Phi-2": "baseline-phi-2",
    "Gemma-2-2B": "baseline-gemma-2-2b", "GLA-1.3B": "baseline-gla-13b",
    "RWKV-7 World3": "baseline-rwkv7-2.9b-hf", "RWKV-6 World2.1": "baseline-rwkv-6-world21",
    "Mamba 2.8B": "baseline-mamba-2.8b", "Pythia-2.8B": "baseline-pythia-28b",
    "DCLM-1B": "baseline-dclm-1b", "OLMo-2-1B": "baseline-olmo-2-1b",
    "MDLM-OWT": "baseline-mdlm-owt", "SEDD-medium": "baseline-sedd-medium",
    "CoLA-DLM": "baseline-cola-dlm", "ELF-L": "baseline-elf-l", "ELF-M": "baseline-elf-m",
    "STAR-LDM": "baseline-star-ldm",
}
# model_type values that are NOT softmax attention
NONATTN = {"mamba", "mamba2", "rwkv", "rwkv5", "rwkv6", "rwkv7", "gla", "retnet",
           "hgrn", "hgrn2", "deltanet", "based", "linear_attention"}


def arch(label):
    """(model_type, has_softmax_attention) from the row's own config.json."""
    p = os.path.join(BENCH, SUB[label], "run_summary.json")
    md = json.load(open(p)).get("model_dir")
    cfg = os.path.join(md or "", "config.json")
    if not md or not os.path.exists(cfg):
        return None, None
    c = json.load(open(cfg))
    mt = str(c.get("model_type", "")).lower()
    heads = c.get("num_attention_heads") or c.get("n_heads") or c.get("n_head")
    attn = mt not in NONATTN and bool(heads)
    return mt, attn


def table_groups():
    tex = open(TEX).read()
    i = tex.index(r"\label{tab:models}")
    j = tex.index(r"\bottomrule", i)
    g, out = None, {}
    for line in tex[i:j].splitlines():
        s = line.strip()
        m = re.match(r"\\multicolumn\{5\}\{l\}\{\\emph\{(.+?)\}\}", s)
        if m:
            g = m.group(1)
            out[g] = []
            continue
        if s.endswith("\\\\") and not s.startswith(("\\midrule", "\\toprule",
                                                    "Checkpoint", "\\multicolumn")):
            c = [x.strip() for x in s[:-2].split("&")]
            if len(c) == 5 and g:
                out[g].append(c[0])
    return out


def prose_groups():
    tex = open(TEX).read()
    line = next(l for l in tex.splitlines()
                if "Baselines are drawn from" in l)
    return [[x.strip() for x in m.group(1).split(",")]
            for m in re.finditer(r"\(([^()]*?)\)", line)]


def compare(tg, pg, quiet=False):
    """Returns the list of defects. A defect is a row in a group whose architecture
    contradicts the header, or a table row named in no prose group."""
    defects = []
    for header, rows in tg.items():
        if "Linear-attention" in header or "Autoregressive" in header:
            want_attn = "Autoregressive" in header
            for r in rows:
                if r not in SUB:
                    continue
                mt, attn = arch(r)
                if attn is None:
                    defects.append(f"{r}: no config.json, cannot verify group")
                elif attn != want_attn:
                    defects.append(f"{r} (model_type={mt}, softmax-attn={attn}) is under "
                                   f"'{header}'")
    prose_all = {x for grp in pg for x in grp}
    for header, rows in tg.items():
        if "This work" in header:
            continue
        for r in rows:
            # prose abbreviates MDLM-OWT as MDLM and SEDD-medium as SEDD
            if not any(r.startswith(p) or p.startswith(r.split("-")[0]) and
                       r.split("-")[0] == p.split("-")[0] for p in prose_all):
                defects.append(f"{r} appears in tab:models but in no prose family group")
    if not quiet:
        for d in defects:
            print(f"  DEFECT: {d}")
    return defects


def main():
    print("CONTROLS")
    ok = True
    for label, want in [("Mamba 2.8B", False), ("GLA-1.3B", False),
                        ("RWKV-6 World2.1", False), ("Qwen2.5-3B", True),
                        ("Pythia-2.8B", True), ("DCLM-1B", True), ("OLMo-2-1B", True)]:
        mt, attn = arch(label)
        good = attn == want
        ok &= good
        print(f"  {label:18s} model_type={mt!s:10s} softmax-attn={attn!s:6s} "
              f"want {want!s:6s} {'OK' if good else 'FAIL (BAD)'}")
    if not ok:
        print("  -> controls FAIL")
        return 1

    tg, pg = table_groups(), prose_groups()
    print("\nNEGATIVE CONTROL -- move Mamba into the Transformer group; must be reported:")
    mut = {k: list(v) for k, v in tg.items()}
    for k in mut:
        if "Linear-attention" in k and "Mamba 2.8B" in mut[k]:
            mut[k].remove("Mamba 2.8B")
        if "Autoregressive" in k:
            mut[k].append("Mamba 2.8B")
    nd = compare(mut, pg, quiet=True)
    caught = any("Mamba" in d and "Autoregressive" in d for d in nd)
    print(f"  mutated table -> {len(nd)} defect(s), Mamba caught: {caught} "
          f"{'OK' if caught else 'FAIL (BAD)'}")
    if not caught:
        return 1

    print("\nAS WRITTEN")
    for header, rows in tg.items():
        print(f"  {header}\n      {rows}")
    print("  prose groups:")
    for grp in pg:
        print(f"      {grp}")
    d = compare(tg, pg)
    print(f"\n  defects: {len(d)}   "
          f"{'group assignment and prose enumeration agree' if not d else 'MUST FIX'}")
    return 0 if not d else 1


if __name__ == "__main__":
    sys.exit(main())

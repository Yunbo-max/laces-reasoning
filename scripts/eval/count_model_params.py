#!/usr/bin/env python3
"""Count parameters for every checkpoint the paper tabulates, from the weight files.

Appendix app:models is promised at main.tex:272 to list "every checkpoint with its total,
frozen and trainable parameter counts". Nothing in the repo produced those numbers, and the
published Params cells turned out to contain at least two that disagree with the files
(Phi-2 printed 2.7B against a measured 2.78B; RWKV-6 printed 3.0B against 3.10B). This
script is the instrument for that table: it resolves each row's weight directory from the
`model_dir` recorded in that row's own capability run_summary.json -- never from the row
name, which is how two nonexistent paths were guessed earlier -- then counts tensors.

For the two \method{} rows, three counts are distinguished and all three printed, because
they differ by more than a rounding and the paper needs a stated convention:
  total     frozen decoder + the whole resident interface
  frozen    the decoder, plus interface modules that stage 2 left frozen
  trainable only the modules that receive gradients under the run's own stage-2 config
             (routed denoiser + low-rank writer + state scale; the encoder trunk and the
             posterior heads are frozen whenever s2_unfreeze_s0 is false)

The state basis is stored twice in each checkpoint, at the top level and inside
`trainable_state`, bit-identically. It is counted once; summing the top level and the nested
dict double-counts 83.9M.
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222"
BENCH = os.path.join(ROOT, "research/laces-reasoning/results/benchmarks")

# paper row label -> the results dir whose run_summary.json records the weight path
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
ARMS = [
    ("\\method{} Global", "outputs_dynamic_basis/"
     "laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000"),
    ("\\method{} Chunked", "outputs_dynamic_basis/"
     "laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000"),
]
DECODER = os.path.join(ROOT, "models/RWKV7-Goose-World3-2.9B-HF")


def resolve_dir(results_subdir):
    """The weight dir this row was actually evaluated with, per its own run_summary."""
    p = os.path.join(BENCH, results_subdir, "run_summary.json")
    if not os.path.exists(p):
        return None, "no run_summary.json"
    d = json.load(open(p))
    md = d.get("model_dir") or d.get("ckpt") or d.get("model")
    if not md:
        return None, "run_summary records no model_dir"
    return str(md), None


def _shallowest(paths):
    """Only the paths at the smallest directory depth. Two model dirs here contain a
    complete nested copy of themselves (models/RWKV7-Goose-World3-2.9B-HF holds another
    RWKV7-Goose-World3-2.9B-HF/ with its own index and shards), so a recursive glob sees
    each shard twice and returns exactly 2x. Depth alone is not enough -- the same dir
    also holds `model.safetensors.old`, a full duplicate at the SAME depth -- so this is
    only the first of two independent defences; the second is key deduplication below."""
    if not paths:
        return paths
    lo = min(p.count(os.sep) for p in paths)
    return [p for p in paths if p.count(os.sep) == lo]


def count_dir(d, warn=True):
    """Total DISTINCT parameters in a weight directory.

    Two failure modes are guarded, both of which produced wrong answers in this file's
    own history:

    UNDERCOUNT. CoLA-DLM, ELF-L and ELF-M first reported "no weight files found" because
    the search was too shallow and too extension-bound: CoLA's shards sit two levels down
    in cola_dlm/cola_dit/, and each ELF checkpoint is a single 2.7-5.2 GB file named
    `checkpoint_57051` with no suffix at all. "Not found" from a search whose depth and
    filename assumptions were never checked is a statement about the search.

    OVERCOUNT. Fixing that by recursing then reported RWKV-7 World3 as 5.90B, exactly 2x
    its true 2.95B, because that dir holds both a nested self-copy and a `.old` duplicate.
    A size-based or depth-based rule cannot separate a shard from a copy; identical tensor
    KEYS can. Every tensor is therefore counted once per (key, shape), and a duplicate is
    reported rather than silently dropped.
    """
    if not d or not os.path.isdir(d):
        return None, f"dir missing: {d}"

    seen, tot, dup = {}, 0, 0

    def add(key, n, shape):
        """Count (key, shape) once. Returns True if this was a fresh tensor."""
        nonlocal tot, dup
        sig = (key, tuple(shape))
        if sig in seen:
            dup += n
            return False
        seen[sig] = n
        tot += n
        return True

    st = _shallowest(sorted(glob.glob(os.path.join(d, "**", "*.safetensors"),
                                      recursive=True)))
    if st:
        # An index is authoritative only for its OWN directory. Globbing indexes
        # separately and taking the shallowest is wrong: with the top-level index absent
        # but a nested copy present, the nested index becomes authoritative and excludes
        # every shard beside it. The positive control below caught exactly that.
        dirs = {os.path.dirname(f) for f in st}
        named = set()
        for sub in sorted(dirs):
            for i in sorted(glob.glob(os.path.join(sub, "*.safetensors.index.json"))):
                for f in set(json.load(open(i))["weight_map"].values()):
                    named.add(os.path.join(sub, f))
        if named:
            # keep the indexed shards, plus any unindexed sibling that is not merely a
            # renamed copy of one (`model.safetensors.old` is, and is dropped by name)
            st = sorted(named | {f for f in st if f not in named
                                 and not f.endswith(".old")})
        from safetensors import safe_open
        for f in st:
            with safe_open(f, framework="pt") as g:
                for k in g.keys():
                    shape = g.get_slice(k).get_shape()
                    m = 1
                    for x in shape:
                        m *= x
                    add(k, m, shape)
        if dup and warn:
            print(f"    [dedup] {os.path.basename(d)}: dropped {dup:,} duplicate params")
        return tot, None

    cands = []
    for pat in ("*.bin", "*.pt", "*.pth", "*.ckpt"):
        cands += glob.glob(os.path.join(d, "**", pat), recursive=True)
    if not cands:
        # extensionless torch saves: ELF ships `checkpoint_57051` with no suffix
        cands = [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                 if os.path.isfile(p) and os.path.getsize(p) > 50_000_000]
    cands = _shallowest(sorted(cands))
    if not cands:
        return None, "no weight files found (recursive, any extension, >50MB)"
    import torch
    for f in cands:
        try:
            sd = torch.load(f, map_location="meta", mmap=True, weights_only=True)
        except Exception:
            try:
                sd = torch.load(f, map_location="meta", mmap=True, weights_only=False)
            except Exception as e:
                return None, f"unreadable: {type(e).__name__}"
        # lightning / custom trainers hide the tensors one level down. ELF stores them in
        # `params` alongside a bit-shaped EMA shadow in `ema_params1` with IDENTICAL inner
        # keys -- so key dedup cannot separate them (it would keep whichever came first)
        # and a substring test for "ema" misses `ema_params1` as a top-level container.
        # Resolve by naming the canonical container and skipping shadow containers.
        SHADOW = ("ema", "ema_params", "ema_params1", "ema_params2", "ema_model",
                  "model_ema", "averaged_model", "opt_state", "optimizer",
                  "grad_accum_buffers")
        if isinstance(sd, dict):
            for key in ("params", "state_dict", "model", "module"):
                if isinstance(sd.get(key), dict) and any(
                        hasattr(t, "numel") for t in sd[key].values()):
                    sd = sd[key]
                    break
        if not isinstance(sd, dict):
            continue
        for k, v in sd.items():
            # an EMA shadow and an optimizer moment are storage, not model capacity
            if isinstance(k, str) and k.split(".")[0].lower() in SHADOW:
                continue
            if hasattr(v, "numel"):
                add(k, v.numel(), tuple(v.shape))
            elif isinstance(v, dict):
                for k2, t in v.items():
                    if hasattr(t, "numel"):
                        add(f"{k}.{k2}", t.numel(), tuple(t.shape))
    if dup and warn:
        print(f"    [dedup] {os.path.basename(d)}: dropped {dup:,} duplicate params")
    return (tot, None) if tot else (None, "no tensors")


def arm_counts(ckpt_dir):
    """(total, frozen, trainable) for one \\method{} checkpoint, plus the routing it used."""
    import torch
    sd = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="meta",
                    mmap=True, weights_only=False)
    cfg = sd.get("config", "")

    def opt(k, dflt=None):
        m = re.search(rf"^\s*{k}:\s*(\S+)\s*$", cfg, re.M)
        return m.group(1) if m else dflt

    ts = sd["trainable_state"]
    per = {}
    for k, v in ts.items():
        per[k.split(".")[0]] = per.get(k.split(".")[0], 0) + v.numel()
    stored = sum(per.values())          # state_basis counted once: it lives inside ts too

    traj = str(opt("trajectory_mode")).lower() in ("true", "1", "yes")
    gdt = str(opt("global_denoiser_type")).lower()
    dit = "trajectory_dit" if (traj or gdt in ("rwkv", "birwkv")) else "latent_dit"
    grad_mods = [dit, "s1_trunk", "s1_u_head", "s1_v_head", "state_scale"]
    if str(opt("s2_unfreeze_s0")).lower() in ("true", "1", "yes"):
        grad_mods += ["encoder_trunk", "mu_head", "logvar_head"]
    if str(opt("s2_unfreeze_s1")).lower() not in ("true", "1", "yes"):
        grad_mods = [dit]
    trainable = sum(per.get(m, 0) for m in grad_mods)

    dec, err = count_dir(DECODER)
    if dec is None:
        raise SystemExit(f"cannot count the frozen decoder: {err}")
    return {"total": dec + stored, "frozen": dec + (stored - trainable),
            "trainable": trainable, "stored_interface": stored, "decoder": dec,
            "routed_denoiser": dit, "stage": opt("stage"),
            "s2_unfreeze_s1": opt("s2_unfreeze_s1"), "s2_unfreeze_s0": opt("s2_unfreeze_s0")}


def self_test():
    """Calibrate the counter against known answers and against known traps.

    This exists because two successive edits to count_dir's file selection each produced a
    confidently wrong table: first three rows read as "no weight files" (undercount), then
    RWKV-7 read as 5.90B, exactly 2x (overcount). Both looked like findings. A counter
    whose selector has never been shown to fail on a duplicate cannot be trusted to have
    counted once.
    """
    ok = True

    # ANCHORS: published configurations that are independently known, to 2 decimals.
    # These are the values the architectures imply, not values read back from this script.
    anchors = [
        ("Llama-3.2-3B", "baseline-llama-3.2-3b", 3.21),
        ("Gemma-2-2B", "baseline-gemma-2-2b", 2.61),
        ("RWKV-7 World3", "baseline-rwkv7-2.9b-hf", 2.95),   # nested-copy 2x trap
        ("Mamba 2.8B", "baseline-mamba-2.8b", 2.77),
        # ELF stores a bit-shaped EMA shadow under `ema_params1` with identical inner
        # keys, so this row reads exactly 2x if the shadow is not excluded by container
        # name. Key dedup CANNOT catch it; only the container rule can.
        ("ELF-M (EMA trap)", "baseline-elf-m", 0.34),
    ]
    print("anchors -- each MUST match the known configuration to 2dp:")
    for label, sub, want in anchors:
        d, err = resolve_dir(sub)
        n = None if err else count_dir(d, warn=False)[0]
        got = None if n is None else round(n / 1e9, 2)
        good = got == want
        ok &= good
        print(f"  {label:18s} want {want:5.2f}B  got {str(got):>6}B  "
              f"{'OK' if good else 'FAIL (BAD)'}")

    # NEGATIVE CONTROL: a directory in which every shard is duplicated must still return
    # the single-copy count. Built by symlinking a real dir's shards into a nested copy of
    # itself, which is the exact shape that produced the 5.90B.
    import shutil
    import tempfile
    d, err = resolve_dir("baseline-gemma-2-2b")
    if err:
        print(f"  negative control SKIPPED: {err}")
        return 1
    base = count_dir(d, warn=False)[0]
    tmp = tempfile.mkdtemp()
    try:
        trap = os.path.join(tmp, "trap")
        os.makedirs(trap)
        for f in os.listdir(d):
            os.symlink(os.path.join(d, f), os.path.join(trap, f))
        nest = os.path.join(trap, "trap")          # nested self-copy, as on disk
        os.makedirs(nest)
        for f in os.listdir(d):
            os.symlink(os.path.join(d, f), os.path.join(nest, f))
        for f in glob.glob(os.path.join(d, "*.safetensors")):   # same-depth .old copy
            os.symlink(f, os.path.join(trap, os.path.basename(f) + ".old"))
        got = count_dir(trap, warn=False)[0]
        good = got == base
        ok &= good
        print("negative control -- a dir with a nested copy AND a .old copy:")
        print(f"  duplicated dir      want {base/1e9:5.2f}B  got {got/1e9:5.2f}B  "
              f"{'OK' if good else f'FAIL (BAD, ratio {got/base:.2f}x)'}")

        # POSITIVE CONTROL: the check must be able to fail. Add a tensor that is NOT a
        # duplicate (a distinct key) and confirm the count rises, so the dedup is not
        # simply collapsing everything it sees.
        from safetensors.torch import save_file
        import torch
        extra = os.path.join(trap, "zz-extra.safetensors")
        save_file({"a.brand.new.key": torch.zeros(1000, 1000)}, extra)
        os.remove(os.path.join(trap, "model.safetensors.index.json"))
        for f in glob.glob(os.path.join(trap, "*.safetensors.old")):
            os.remove(f)
        got2 = count_dir(trap, warn=False)[0]
        rose = got2 == base + 1_000_000
        ok &= rose
        print("positive control -- dedup must NOT hide a genuinely new tensor:")
        print(f"  +1M distinct params want {(base+1_000_000)/1e9:5.2f}B  "
              f"got {got2/1e9:5.2f}B  {'OK' if rose else 'FAIL (BAD)'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nSELF-TEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=os.path.join(
        ROOT, "research/DiffRwkv/paper_ICLR27/lalaland/main.tex"))
    ap.add_argument("--json", default=None, help="also write the counts here")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    printed = {}
    if os.path.exists(a.tex):
        tex = open(a.tex).read()
        for m in re.finditer(r"^([A-Za-z0-9][^&\n]{0,32}?)\s*&\s*(\d\.\d+)B\s*&", tex, re.M):
            printed.setdefault(m.group(1).strip(), set()).add(m.group(2) + "B")

    out = {}
    print(f"{'row':20s} {'measured':>12s} {'->2dp':>7s} {'printed':>10s}  dir / note")
    mism = 0
    for label, sub in ROWS:
        d, err = resolve_dir(sub)
        if err:
            print(f"{label:20s} {'--':>12s} {'--':>7s} {'':>10s}  SKIP: {err}")
            continue
        n, err2 = count_dir(d)
        if n is None:
            print(f"{label:20s} {'--':>12s} {'--':>7s} {'':>10s}  SKIP: {err2}")
            continue
        got = f"{n/1e9:.2f}B"
        pr = ",".join(sorted(printed.get(label, []))) or "-"
        flag = ""
        if pr != "-":
            # The paper prints ONE decimal; comparing at two would flag every row as a
            # mismatch (it did: 14 of 14 on the first run, which is a broken comparator and
            # not a finding). A cell is faithful if it equals the measurement rounded to the
            # precision the cell itself prints.
            ok = False
            for p in pr.split(","):
                lit = p.rstrip("B")
                dp = len(lit.split(".")[1]) if "." in lit else 0
                ok |= f"{n/1e9:.{dp}f}" == lit
            if not ok:
                flag = f"  <== DIFFERS FROM PRINTED {pr}"
                mism += 1
        out[label] = {"params": n, "dir": d}
        print(f"{label:20s} {n:12,} {got:>7s} {pr:>10s}  {os.path.basename(d)}{flag}")

    for label, ck in ARMS:
        full = os.path.join(ROOT, "research/laces-reasoning", ck)
        c = arm_counts(full)
        out[label] = c
        print(f"{label:20s} {c['total']:12,} {c['total']/1e9:6.2f}B "
              f"{','.join(sorted(printed.get(label.replace(chr(92)+'method{} ', ''), []))) or '-':>10s}"
              f"  total | frozen {c['frozen']/1e9:.2f}B | trainable {c['trainable']/1e9:.2f}B "
              f"| denoiser={c['routed_denoiser']}")

    print(f"\nrows measured: {len([k for k in out])}   printed cells that differ: {mism}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

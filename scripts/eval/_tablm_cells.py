#!/usr/bin/env python3
"""Per-cell provenance for tab:lm, with the structure of the output asserted.

An earlier version of this ledger serialized `cells` as a list of corpus NAMES, losing every
per-cell field it was written to record -- the dump looked populated and answered nothing.
So this writes one dict per (corpus, column) and then asserts, after loading the file back,
that each entry carries the keys the table needs: file, mode, impl, ckpt, n_tokens.

MATCHING. A cell is matched only when some result file's own `perplexity[corpus]` equals the
printed value at the precision the cell prints. Substring grep over results/ is not used: it
was used first, and "16.09" hits unrelated digits inside jsonl prose.

CONTROLS. Positive: wikitext2=14.85 must be found (>=1 hit) or the matcher is inert.
Negative: wikitext2=99.99 must not be found (0 hits) or it matches anything. Both run before
any cell is resolved and the script exits if either fails.
"""
import glob
import json
import os
import sys

ROOT = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning"
OUT = os.path.join(ROOT, "results/paper_ledger/TABLM_CELLS.json")
TEX = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/"
       "paper_ICLR27/lalaland/main.tex")

# the corpus key inside a result file, per printed row label
ROW_KEY = {"WikiText-2": "wikitext2", "WikiText-103": "wikitext103", "PG19": "pg19",
           "LM1B": "lm1b", "XSum": "xsum"}
COLS = ["RWKV-7 World3", "\\method{} Global", "\\method{} Chunked"]

# Which (ckpt, mode) each column is: taking hits[0] is not disambiguation. WikiText-103's
# 14.79 has four matches at 2dp and three of them are phi-2 -- a collision at the printed
# precision, not a second source for this cell. The column's own (ckpt suffix, mode) is the
# discriminator, and a cell whose matches do not include exactly one file with that pair is
# reported as ambiguous rather than resolved.
COL_EXPECT = {
    "RWKV-7 World3": ("global-joint-b16-50k/step_00050000", "raw"),
    "\\method{} Global": ("global-joint-b16-50k/step_00050000", "prefix_cfg"),
    "\\method{} Chunked": ("s0frozen-joint-b4-50k-fla03/step_00030000", "traj_cfg"),
}


def load_all():
    """Every result file that records a `perplexity` dict, with its metadata."""
    out = []
    for p in glob.glob(os.path.join(ROOT, "results", "**", "*.json"), recursive=True):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        ppl = d.get("perplexity")
        if not isinstance(ppl, dict) or not ppl:
            continue
        out.append({
            "file": os.path.relpath(p, ROOT),
            "perplexity": ppl,
            "bits_per_byte": d.get("bits_per_byte") or {},
            "n_tokens": d.get("n_tokens") or d.get("tokens") or {},
            "mode": d.get("mode") or (d.get("args") or {}).get("mode"),
            "impl": d.get("impl") or (d.get("args") or {}).get("impl"),
            "mtime": os.path.getmtime(p),
            "max_files": d.get("max_files"),
            "ckpt": d.get("ckpt") or d.get("model_dir")
                    or (d.get("args") or {}).get("ckpt"),
        })
    return out


def find(files, key, printed):
    """Files whose own perplexity[key] equals `printed` at the precision printed."""
    dp = len(printed.split(".")[1]) if "." in printed else 0
    hits = []
    for f in files:
        v = f["perplexity"].get(key)
        if isinstance(v, (int, float)) and f"{v:.{dp}f}" == printed:
            hits.append(f)
    return hits


def main():
    files = load_all()
    print(f"result files carrying a perplexity dict: {len(files)}")

    print("\nCONTROLS")
    pos = find(files, "wikitext2", "14.85")
    neg = find(files, "wikitext2", "99.99")
    print(f"  positive  wikitext2=14.85 -> {len(pos)} hit(s)  "
          f"{'OK' if pos else 'FAIL (BAD: matcher inert)'}")
    print(f"  negative  wikitext2=99.99 -> {len(neg)} hit(s)  "
          f"{'OK' if not neg else 'FAIL (BAD: matches anything)'}")
    if not pos or neg:
        return 1

    # NEGATIVE CONTROL on the disambiguator. COL_EXPECT was read off the data and then
    # written back, so "9 of 9 resolve" would be self-confirming unless a wrong constraint
    # is shown to resolve nothing. Demand mode='nonexistent_mode' and every cell must fail.
    bogus = [h for h in files if h["mode"] == "nonexistent_mode"]
    print(f"  negative  mode='nonexistent_mode' -> {len(bogus)} file(s)  "
          f"{'OK' if not bogus else 'FAIL (BAD)'}")
    bogus_ck = [h for h in files
                if (h["ckpt"] or "").endswith("no-such-ckpt/step_00000001")]
    print(f"  negative  ckpt suffix 'no-such-ckpt/step_00000001' -> {len(bogus_ck)} "
          f"file(s)  {'OK' if not bogus_ck else 'FAIL (BAD)'}")
    if bogus or bogus_ck:
        return 1
    # and the constraint must be able to SEPARATE: the 14.79 collision has 4 matches at 2dp,
    # of which the column's own (ckpt,mode) must select exactly 1
    coll = find(files, "wikitext103", "14.79")
    ck, md = COL_EXPECT["\\method{} Chunked"]
    sep = [h for h in coll if (h["ckpt"] or "").endswith(ck) and h["mode"] == md]
    print(f"  separation  wikitext103=14.79 -> {len(coll)} at 2dp, {len(sep)} after "
          f"(ckpt,mode)  {'OK' if len(coll) > 1 and len(sep) == 1 else 'FAIL (BAD)'}")
    if not (len(coll) > 1 and len(sep) == 1):
        return 1

    tex = open(TEX).read()
    i = tex.index(r"\label{tab:lm}")
    j = tex.index(r"\bottomrule", i)
    printed = {}
    for line in tex[i:j].splitlines():
        s = line.strip()
        if not s.endswith("\\\\") or s.startswith(("\\toprule", "\\midrule", "Corpus")):
            continue
        c = [x.strip() for x in s[:-2].split("&")]
        if len(c) == 4 and c[0] in ROW_KEY:
            printed[c[0]] = [x.replace("\\textbf{", "").replace("}", "").strip()
                             for x in c[1:]]
    if len(printed) != 5:
        print(f"ABORT: parsed {len(printed)} rows of tab:lm, expected 5")
        return 1

    cells, unsourced, ambiguous = [], 0, 0
    print(f"\n{'corpus':13s} {'col':20s} {'value':>7s} {'hits':>4s} {'ckpt+mode':>9s}  "
          f"file / mode")
    for row, vals in printed.items():
        key = ROW_KEY[row]
        for col, val in zip(COLS, vals):
            hits = find(files, key, val)
            ck_suffix, want_mode = COL_EXPECT[col]
            exact = [h for h in hits
                     if (h["ckpt"] or "").endswith(ck_suffix) and h["mode"] == want_mode]
            # Two files can legitimately share (ckpt, mode, max_files): PG19 col1 matches
            # both ppl16/global-dit-50k-raw-control.json and the older
            # ppl/global-dit-50k_raw.json. They are the same configuration measured twice,
            # and they DISAGREE elsewhere -- wikitext 16.09 vs 16.08, and the older file
            # carries the empty-input fingerprint (ppl 1.0, bpb 0.0) on lm1b/ptb/wmt14.
            # The table's own values are the ppl16 generation, so when several files remain
            # the newest is taken and every candidate is recorded with its mtime, so the
            # choice is visible rather than implicit in glob order.
            if len(exact) > 1:
                exact = sorted(exact, key=lambda h: -h["mtime"])
                ent_multi = [{"file": h["file"], "mtime": int(h["mtime"]),
                              "value": h["perplexity"].get(key)} for h in exact]
            else:
                ent_multi = None
            ent = {"corpus": row, "corpus_key": key, "column": col, "printed": val,
                   "n_hits": len(hits), "n_hits_matching_column": len(exact),
                   "expected_ckpt_suffix": ck_suffix, "expected_mode": want_mode,
                   "file": None, "mode": None, "impl": None, "ckpt": None,
                   "n_tokens": None, "empty_fingerprint": None,
                   "all_files": [h["file"] for h in hits],
                   "exact_candidates": ent_multi}
            if len(exact) >= 1:
                h = exact[0]
                bpb = h["bits_per_byte"].get(key)
                ent.update(file=h["file"], mode=h["mode"], impl=h["impl"],
                           ckpt=h["ckpt"], n_tokens=h["n_tokens"].get(key),
                           n_docs=(h.get("n_docs") or {}).get(key) if isinstance(
                               h.get("n_docs"), dict) else None,
                           empty_fingerprint=(h["perplexity"].get(key) == 1.0
                                              and bpb == 0.0))
                status = "1/1 OK" if ent_multi is None else f"{len(exact)}->newest"
            elif not hits:
                unsourced += 1
                status = "NO FILE"
            else:
                ambiguous += 1
                status = f"{len(exact)} exact"
            cells.append(ent)
            print(f"{row:13s} {col:20s} {val:>7s} {len(hits):>4d} {status:>9s}  "
                  f"{(ent['file'] or '--')}")

    print(f"\ncells: {len(cells)}   unsourced: {unsourced}   "
          f"ambiguous after (ckpt,mode) constraint: {ambiguous}")
    json.dump({"generated": "2026-09-22", "table": "tab:lm", "cells": cells,
               "unsourced": unsourced, "ambiguous": ambiguous,
               "column_expect": {k: list(v) for k, v in COL_EXPECT.items()}},
              open(OUT, "w"), indent=2)

    # the structure the earlier ledger lost: assert it after reading the file BACK
    back = json.load(open(OUT))
    need = {"corpus", "column", "printed", "n_hits", "file", "mode", "impl", "ckpt",
            "n_tokens"}
    if not isinstance(back.get("cells"), list) or len(back["cells"]) != 15:
        print(f"ABORT: reloaded ledger has {len(back.get('cells', []))} cells, not 15")
        return 1
    for c in back["cells"]:
        if not isinstance(c, dict) or not need <= set(c):
            print(f"ABORT: reloaded cell is missing keys: {c}")
            return 1
    print(f"wrote {OUT} -- reloaded 15 cells, each with {sorted(need)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

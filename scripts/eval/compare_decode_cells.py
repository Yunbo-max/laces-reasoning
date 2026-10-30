#!/usr/bin/env python3
"""Compare an hf-decode result cell against its --decode loop rebuild, one axis at a time.

Written because every "before/after" in the decode-defect line turned out to need three
separate checks before the delta could be attributed to decode at all:

  1. protocol identity   -- same model_dir/data_dir/max_samples/budget/max_new_tokens/seed
  2. prompt identity     -- same input_tokens per row AND the rows not pinned at the cap.
                            Equal input_tokens is NOT sufficient: when a prompt is truncated to
                            exactly max_input_tokens, prepending BOS keeps the length identical
                            while shifting which source tokens survive middle-truncation. That
                            is how the gla BOS confound hid (250/250 infinitebench rows at cap).
  3. metric identity     -- RULER's official string_match_all is fractional on multi-gold tasks
                            (cwe/fwe/vt) and binary on single-gold ones (niah_*/qa_*). A cell
                            that is binary *everywhere* is on the legacy max-over-golds metric.

Usage:
    compare_decode_cells.py HF_JSON LOOP_JSON [--name LABEL]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOL_KEYS = ("model_dir", "data_dir", "max_samples", "max_input_tokens",
                 "max_new_tokens", "seed", "prepend_bos")
MULTI_GOLD = ("cwe", "fwe", "vt")


def scores(cell: dict) -> dict[str, float]:
    """Per-dataset score, tolerating both aggregate shapes.

    The harness writes {n, score, ...} dicts; ``rescore_ruler_official.py`` rewrites the same
    field as bare floats. Both appear in results/benchmarks/, so a reader that assumes either
    one crashes on the other (observed: TypeError: 'float' object is not subscriptable).
    """
    out = {}
    for key, value in (cell.get("aggregates") or {}).items():
        out[key] = float(value) if isinstance(value, (int, float)) else float(value["score"])
    return out


def macro(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values) if values else float("nan")


def row_key(row: dict):
    return (row.get("dataset"), row.get("id"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_json")
    parser.add_argument("loop_json")
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    hf = json.loads(Path(args.hf_json).read_text())
    loop = json.loads(Path(args.loop_json).read_text())
    label = args.name or Path(args.loop_json).stem
    print(f"=== {label} ===")

    # 1. protocol
    hc, lc = hf.get("config") or {}, loop.get("config") or {}
    confounds = []
    for key in PROTOCOL_KEYS:
        a, b = hc.get(key), lc.get(key)
        if a == b:
            continue
        # rwkv7's tokenizer answers False to the BOS probe, so False == the older absent key.
        if key == "prepend_bos" and (a is None and b is False) or (a is False and b is None):
            print(f"  protocol {key:<17} {a!r} vs {b!r}  (equivalent: no BOS either way)")
            continue
        confounds.append(key)
        print(f"  protocol {key:<17} {str(a)[-40:]!r} vs {str(b)[-40:]!r}  ** DIFFERS **")
    if not confounds:
        print("  protocol: identical on all checked keys")

    # 2. prompts
    hm = {row_key(r): r for r in hf.get("rows") or []}
    lm = {row_key(r): r for r in loop.get("rows") or []}
    common = sorted(set(hm) & set(lm))
    tok_deltas = sorted({lm[k].get("input_tokens", 0) - hm[k].get("input_tokens", 0)
                         for k in common})
    cap = lc.get("max_input_tokens")
    at_cap = sum(1 for k in common if lm[k].get("input_tokens") == cap)
    print(f"  rows compared: {len(common)}  input_tokens delta set: {tok_deltas}")
    if tok_deltas != [0]:
        confounds.append("input_tokens")
        print("  ** prompts differ in LENGTH -- not a decode-only comparison **")
    elif at_cap:
        print(f"  ** {at_cap}/{len(common)} rows sit exactly at the {cap}-token cap: equal length "
              f"does NOT prove equal content (BOS shifts middle-truncation) **")
    else:
        print("  prompts: identical length and no row pinned at the cap")

    # 3. metric shape
    for name, cell in (("hf", hf), ("loop", loop)):
        values = {k: [r.get("score") for r in (cell.get("rows") or []) if r.get("dataset") == k]
                  for k in scores(cell)}
        multi = {k: v for k, v in values.items() if any(t in k for t in MULTI_GOLD)}
        binary_multi = [k for k, v in multi.items() if set(v) <= {0.0, 1.0} and any(v)]
        if multi:
            verdict = ("LEGACY max-over-golds" if binary_multi and
                       len(binary_multi) == len([k for k, v in multi.items() if any(v)])
                       else "official string_match_all")
            print(f"  metric ({name}): {verdict}  [multi-gold tasks: {sorted(multi)}]")

    # scores
    hs, ls = scores(hf), scores(loop)
    keys = sorted(set(hs) & set(ls))
    print(f"\n  {'dataset':<24} {'hf':>9} {'loop':>9} {'delta':>9}")
    for key in keys:
        print(f"  {key:<24} {hs[key]:>9.4f} {ls[key]:>9.4f} {ls[key] - hs[key]:>+9.4f}")
    mh, ml = macro({k: hs[k] for k in keys}), macro({k: ls[k] for k in keys})
    ratio = f"{ml / mh:.2f}x" if mh else "n/a"
    print(f"  {'MACRO':<24} {mh:>9.4f} {ml:>9.4f} {ml - mh:>+9.4f}   ratio {ratio}")

    ident = sum(1 for k in common if lm[k].get("prediction") == hm[k].get("prediction"))
    sdiff = [k for k in common if lm[k].get("score") != hm[k].get("score")]
    print(f"\n  identical predictions: {ident}/{len(common)}")
    print(f"  rows whose SCORE changed: {len(sdiff)}"
          f"  (loop higher {sum(1 for k in sdiff if lm[k]['score'] > hm[k]['score'])},"
          f" hf higher {sum(1 for k in sdiff if hm[k]['score'] > lm[k]['score'])})")
    preds = [lm[k].get("prediction", "") for k in common]
    print(f"  loop output health: empty={sum(1 for p in preds if not p.strip())}"
          f" distinct={len(set(preds))}/{len(preds)}")
    print(f"\n  VERDICT: {'CONFOUNDED on ' + ', '.join(confounds) if confounds else 'decode-only comparison'}")


if __name__ == "__main__":
    main()

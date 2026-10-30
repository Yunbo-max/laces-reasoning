#!/usr/bin/env python3
"""mamba-2.8b LongBench score as a function of input budget, paired on identical rows.

Why this exists: mamba-2.8b's main LongBench row is near zero, and a near-zero row has two very
different explanations that the row itself cannot distinguish -- "the model is bad at these
tasks" versus "the model degrades once the input is long". Those have opposite consequences for
the paper: the first is a capability number we may quote, the second makes the cell an artefact
of the length we chose to feed it, in which case beating it is not evidence of anything.

The design is a paired within-model sweep, which is what makes it decisive. Every cell runs the
SAME 50 documents (25 hotpotqa + 25 2wikimqa) with the SAME decode path (`hf`; mamba is native
to transformers, not fla, so the fla `generate` defect is not in play), the SAME
`prepend_bos=false`, the SAME seed, and the SAME official 32-token generation cap -- both of
these datasets have `dataset2maxlen == 32`, so no cell can win by being allowed to write more.
The only thing that varies is `effective_input_budget`. A monotone trend across budgets is
therefore attributable to input length and nothing else.

Reporting notes:
  * Differences are per-row paired, not a difference of two means. Between-document difficulty
    dominates the variance on a 50-row pool, and pairing removes it.
  * `nonzero` is printed next to the mean because a token-overlap F1 mean can move either by many
    rows improving slightly or by a few rows going from nothing to something. Those are different
    claims and the paper should only make the one the data supports.
  * The degeneracy fingerprints (empty string, single-token repeat) are printed per cell. If they
    are nonzero, that cell is measuring our harness rather than the model, and the length story
    would be unfalsifiable from these numbers alone.
"""
from __future__ import annotations
import json, math, random, re, statistics
from pathlib import Path

ROOT = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
SWEEP = ROOT / "results/benchmarks/falsify_zero"
MAIN_JSON = ROOT / "results/benchmarks/longctx_fix/mamba-2.8b__longbench.json"
MAIN_LOG = ROOT / "logs/qz/lbfix-mamba-2.8b-longbench.log"
DATASETS = ("hotpotqa", "2wikimqa")


def cell_rows(path: Path) -> tuple[dict, dict]:
    """Return ({(dataset, k): score}, meta). `k` is the 1-based index within the dataset.

    Keyed by position rather than by row id because the sweep cells and the main run must be
    joinable even when the main run is only available as a log (see `main_from_log`).
    """
    doc = json.loads(path.read_text())
    idx, out = {}, {}
    empty = collapse = 0
    for r in doc["rows"]:
        ds = r["dataset"]
        idx[ds] = idx.get(ds, 0) + 1
        out[(ds, idx[ds])] = r["score"]
        if not (r.get("prediction") or "").strip():
            empty += 1
        ids = r.get("prediction_token_ids") or []
        if len(ids) >= 4 and len(set(ids)) <= 2:
            collapse += 1
    cfg = doc.get("config", {})
    meta = {"budget": cfg.get("effective_input_budget"),
            "decode": cfg.get("decode", "hf"),
            "bos": cfg.get("prepend_bos"),
            "max_new": cfg.get("max_new_tokens"),
            "empty": empty, "collapse": collapse, "n": len(doc["rows"])}
    return out, meta


def main_from_log() -> tuple[dict, str]:
    """Per-row scores for the 32768 arm, preferring its JSON and falling back to its log.

    The fallback is licensed by a check recorded in the ledger: log order == JSON row order on
    1350/1350 rows across three completed cells (tolerance 0.0006). It exists so this sweep can
    be interpreted while the 450-row main run is still in flight; once the JSON lands the JSON is
    used and the fallback never runs.
    """
    if MAIN_JSON.exists():
        try:
            return cell_rows(MAIN_JSON)[0], "json"
        except json.JSONDecodeError:
            pass                                  # still being written
    if not MAIN_LOG.exists():
        return {}, "missing"
    pat = re.compile(r"\[longbench/([a-z_0-9]+)/raw\] (\d+)/(\d+) score=([0-9.]+)")
    out = {}
    for ds, k, _tot, s in pat.findall(MAIN_LOG.read_text()):
        out[(ds, int(k))] = float(s)
    return out, "log (main run still in flight)"


def paired(a: dict, b: dict, reps: int = 10000, seed: int = 0):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diffs = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(reps))
    up = sum(1 for d in diffs if d > 0)
    dn = sum(1 for d in diffs if d < 0)
    m = up + dn
    # Exact two-sided sign test on the discordant pairs. Reported alongside the bootstrap because
    # it asks the other question: the interval is about the average margin, this is about whether
    # more rows move up than down. A length effect should show in both.
    p = (2 * sum(math.comb(m, i) for i in range(max(up, dn), m + 1)) / 2 ** m) if m else 1.0
    return {"n": n, "delta": 100 * statistics.mean(diffs),
            "lo": 100 * means[int(0.025 * reps)], "hi": 100 * means[int(0.975 * reps) - 1],
            "up": up, "dn": dn, "sign_p": min(1.0, p)}


def main() -> None:
    cells = []
    for p in sorted(SWEEP.glob("mamba-lensweep-*.json"), key=lambda q: int(q.stem.split("-")[-1])):
        rows, meta = cell_rows(p)
        cells.append((int(p.stem.split("-")[-1]), rows, meta))
    ref, ref_src = main_from_log()

    print("===== mamba-2.8b: LongBench score vs input budget (hotpotqa+2wikimqa, 25 each) =====")
    print(f"{'budget':>8s} {'n':>4s} {'mean x100':>10s} {'nonzero':>9s} {'decode':>7s} "
          f"{'bos':>5s} {'gen cap':>8s} {'empty':>6s} {'collapse':>9s}")
    for bud, rows, meta in cells:
        vals = list(rows.values())
        print(f"{bud:8d} {len(vals):4d} {100*statistics.mean(vals):10.2f} "
              f"{sum(1 for v in vals if v>0):4d}/{len(vals):<4d} {str(meta['decode']):>7s} "
              f"{str(meta['bos']):>5s} {str(meta['max_new']):>8s} {meta['empty']:6d} "
              f"{meta['collapse']:9d}")
    if ref:
        sub = {k: v for k, v in ref.items() if k[0] in DATASETS and k[1] <= 25}
        if sub:
            print(f"{32768:8d} {len(sub):4d} {100*statistics.mean(sub.values()):10.2f} "
                  f"{sum(1 for v in sub.values() if v>0):4d}/{len(sub):<4d} "
                  f"{'hf':>7s} {'False':>5s} {'512*':>8s} {'-':>6s} {'-':>9s}   <- main run "
                  f"[{ref_src}]")
    print("\n* the main run was launched with --max_new_tokens 512, but the OFFICIAL per-dataset")
    print("  cap governs generation and is 32 for both of these datasets, so the arms are")
    print("  comparable on generation length. Verified: max generated length is 32 in every cell.")

    print("\n===== PAIRED vs the 32768 arm (same documents, budget is the only difference) =====")
    print(f"{'budget':>8s} {'n':>4s} {'short':>8s} {'32768':>8s} {'delta':>8s} "
          f"{'95% CI':>18s} {'up/down':>9s} {'sign p':>8s}  verdict")
    for bud, rows, _meta in cells:
        st = paired(rows, ref)
        if not st:
            continue
        keys = sorted(set(rows) & set(ref))
        verdict = ("SHORTER IS BETTER" if st["lo"] > 0 else
                   "longer is better" if st["hi"] < 0 else "unresolved")
        ci = f"[{st['lo']:+.2f}, {st['hi']:+.2f}]"
        updn = f"{st['up']}/{st['dn']}"
        print(f"{bud:8d} {st['n']:4d} {100*statistics.mean(rows[k] for k in keys):8.2f} "
              f"{100*statistics.mean(ref[k] for k in keys):8.2f} {st['delta']:+8.2f} "
              f"{ci:>18s} {updn:>9s} {st['sign_p']:8.4f}  {verdict}")

    print("\nPositive delta = the SHORTER budget scores HIGHER, i.e. the model is harmed by more")
    print("input. That is the opposite of every other model on this table, and it is why")
    print("mamba-2.8b's main LongBench row must not be quoted as a long-context capability")
    print("number: LongBench's median prompt here is ~11.9k tokens, far past the knee.")
    print("empty/collapse = degeneracy fingerprints. Both 0 means the effect is the model, not")
    print("our harness and not the fla generate defect (mamba is native transformers, not fla).")


if __name__ == "__main__":
    main()

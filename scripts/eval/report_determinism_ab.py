#!/usr/bin/env python3
"""Same-binary A/B replication test: how much of LongBench's row churn is nondeterminism?

Why this exists: two LongBench runs of the same model at the same config produced different
predictions on a large fraction of rows, and that has two incompatible explanations. Either the
harness is nondeterministic (in which case every number on the table carries an unquoted error
bar), or the two runs straddled a code change (in which case the churn is a bug fix and the
"nondeterminism" claim is a confound of my own making). The first version of this comparison was
the second case -- an uncommitted harness edit sat between the two runs -- so this script only
accepts cells that can be shown to come from the SAME binary.

Design:
  * Both reps run back-to-back in ONE job on ONE node, so device, driver, torch, transformers and
    the runner md5 are identical by construction, and the log records all five.
  * Rows are joined on (dataset, id), and the comparison is per-row, never mean-vs-mean.
  * Two different quantities are reported and must not be conflated: how often the GENERATED TEXT
    differs, and how often the SCORE differs. Decode is autoregressive, so a single differing
    logit early can rewrite an entire continuation while leaving token-overlap F1 almost
    untouched. A table of scores is only threatened by the second.
  * Stratified by the official `dataset2maxlen` generation cap, because the churn rate is a
    function of how many tokens the model is allowed to emit -- more sampled tokens means more
    chances to diverge. Pooling the strata hides that and makes the effect look uniform.
  * `first_divergence` is the generated-token index where the two reps first disagree. If churn
    were caused by input handling (truncation, tokenization, BOS) it would be at index 0; a
    divergence deep into the continuation points at floating-point nondeterminism in the kernels.
"""
from __future__ import annotations
import json, math, random, re, statistics, sys
from pathlib import Path

ROOT = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning")
DET = ROOT / "results/benchmarks/determinism"
MAXLEN = {"hotpotqa": 32, "2wikimqa": 32, "musique": 32, "passage_retrieval_en": 32,
          "narrativeqa": 128, "multifieldqa_en": 64,
          "gov_report": 512, "qmsum": 512, "multi_news": 512}
# (label, rep A json, rep B json, log). The log is required, not optional: it is the only place
# the runner md5 and the device string are recorded, and without those a pair cannot be shown to
# be same-binary, which is the whole point of the test.
PAIRS = [("cap-32 stratum (hotpotqa+2wikimqa)",
          "determinism-qwen1p5b-A.json", "determinism-qwen1p5b-B.json",
          "determinism-qwen1p5b.log"),
         ("cap-512 stratum (gov_report+qmsum+multi_news)",
          "determ-sum-qwen1p5b-A.json", "determ-sum-qwen1p5b-B.json",
          "determ-sum-qwen1p5b.log")]


def load(name: str):
    p = DET / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None                        # still being written


def provenance(logname: str) -> dict:
    """Pull the same-binary evidence out of the job log."""
    p = ROOT / "logs/qz" / logname
    if not p.exists():
        return {}
    txt = p.read_text(errors="replace")
    out = {}
    if m := re.search(r"^DEVICE (.+)$", txt, re.M):
        out["device"] = m.group(1).strip()
    if m := re.search(r"^TORCH (\S+) TFMRS (\S+)$", txt, re.M):
        out["torch"], out["tfmrs"] = m.group(1), m.group(2)
    if m := re.search(r"^([0-9a-f]{32})\s+scripts/eval/run_open_context_bench\.py", txt, re.M):
        out["runner_md5"] = m.group(1)
    # Every occurrence, so two different values would be visible rather than silently collapsed.
    out["patch"] = sorted(set(re.findall(r"fla recurrent-generate patch applied: (\w+)", txt)))
    out["peak_mb"] = sorted(set(re.findall(r'"peak_gpu_memory_mb": ([0-9.]+)', txt)))
    out["budget_lines"] = len(set(re.findall(r"^\[budget\] (\{.*\})$", txt, re.M)))
    return out


def rowmap(doc) -> dict:
    return {(r["dataset"], r["id"]): r for r in (doc or {}).get("rows", [])}


def boot(diffs: list[float], reps: int = 10000, seed: int = 0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(reps))
    up = sum(1 for d in diffs if d > 0)
    dn = sum(1 for d in diffs if d < 0)
    m = up + dn
    p = (2 * sum(math.comb(m, i) for i in range(max(up, dn), m + 1)) / 2 ** m) if m else 1.0
    return (100 * statistics.mean(diffs), 100 * means[int(0.025 * reps)],
            100 * means[int(0.975 * reps) - 1], up, dn, min(1.0, p))


def first_divergence(a: dict, b: dict):
    ia = a.get("prediction_token_ids") or []
    ib = b.get("prediction_token_ids") or []
    for i, (x, y) in enumerate(zip(ia, ib)):
        if x != y:
            return i
    return None if ia == ib else min(len(ia), len(ib))


def report_pair(label: str, an: str, bn: str, logn: str) -> None:
    print(f"\n===== {label} =====")
    A, B = load(an), load(bn)
    prov = provenance(logn)
    if prov:
        print(f"  device      {prov.get('device','?')}")
        print(f"  torch/tfmrs {prov.get('torch','?')} / {prov.get('tfmrs','?')}")
        print(f"  runner md5  {prov.get('runner_md5','?')}  (one value = same binary for both reps)")
        print(f"  fla patch   {prov.get('patch')}   budget lines (distinct) {prov.get('budget_lines')}")
        print(f"  peak MB     {prov.get('peak_mb')}")
    if A is None or B is None:
        have = [n for n, d in ((an, A), (bn, B)) if d is not None]
        print(f"  INCOMPLETE: only {have} present -- cannot compare yet.")
        return
    ra, rb = rowmap(A), rowmap(B)
    keys = sorted(set(ra) & set(rb))
    if not keys:
        print("  no shared rows")
        return
    # Identical post-truncation input length on every row is the precondition: if inputs differed,
    # any prediction churn would be an input-handling difference, not nondeterminism.
    same_in = sum(1 for k in keys if ra[k]["input_tokens"] == rb[k]["input_tokens"])
    trunc = sum(1 for k in keys if (ra[k].get("prompt_tokens_untruncated") or 0)
                > ra[k]["input_tokens"])
    print(f"  shared rows {len(keys)}   identical input_tokens {same_in}/{len(keys)}"
          f"   rows truncated {trunc}/{len(keys)}")

    print(f"\n  {'gen cap':>8s} {'n':>4s} {'pred changed':>13s} {'score changed':>14s} "
          f"{'dAVG':>7s} {'95% CI':>18s} {'up/dn':>8s} {'sign p':>7s} {'med 1st div':>12s}")
    strata = {}
    for k in keys:
        strata.setdefault(MAXLEN.get(k[0], 0), []).append(k)
    allpred = allscore = 0
    alldiffs = []
    for cap in sorted(strata):
        ks = strata[cap]
        pred = [k for k in ks if (ra[k].get("prediction") or "") != (rb[k].get("prediction") or "")]
        sc = [k for k in ks if abs(ra[k]["score"] - rb[k]["score"]) > 1e-12]
        diffs = [ra[k]["score"] - rb[k]["score"] for k in ks]
        d, lo, hi, up, dn, p = boot(diffs)
        fd = [x for x in (first_divergence(ra[k], rb[k]) for k in pred) if x is not None]
        med = f"{sorted(fd)[len(fd)//2]}" if fd else "-"
        print(f"  {cap:8d} {len(ks):4d} {f'{len(pred)}/{len(ks)} ({100*len(pred)/len(ks):.0f}%)':>13s} "
              f"{f'{len(sc)}/{len(ks)} ({100*len(sc)/len(ks):.0f}%)':>14s} {d:+7.3f} "
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>18s} {f'{up}/{dn}':>8s} {p:7.3f} {med:>12s}")
        allpred += len(pred); allscore += len(sc); alldiffs += diffs
    d, lo, hi, up, dn, p = boot(alldiffs)
    print(f"  {'POOLED':>8s} {len(alldiffs):4d} "
          f"{f'{allpred}/{len(alldiffs)} ({100*allpred/len(alldiffs):.0f}%)':>13s} "
          f"{f'{allscore}/{len(alldiffs)} ({100*allscore/len(alldiffs):.0f}%)':>14s} {d:+7.3f} "
          f"{f'[{lo:+.2f}, {hi:+.2f}]':>18s} {f'{up}/{dn}':>8s} {p:7.3f}")
    # The number the table's reader actually needs: how far the reported AVG could move.
    aggA = {k: v["score"] for k, v in A["aggregates"].items()}
    aggB = {k: v["score"] for k, v in B["aggregates"].items()}
    shared = sorted(set(aggA) & set(aggB))
    print(f"\n  per-dataset AVG A vs B (x100):")
    for ds in shared:
        print(f"    {ds:24s} {100*aggA[ds]:6.2f}  {100*aggB[ds]:6.2f}   "
              f"delta {100*(aggA[ds]-aggB[ds]):+6.3f}")
    mA = 100 * statistics.mean(aggA[ds] for ds in shared)
    mB = 100 * statistics.mean(aggB[ds] for ds in shared)
    print(f"    {'MEAN over these datasets':24s} {mA:6.2f}  {mB:6.2f}   delta {mA-mB:+6.3f}")


def main() -> None:
    print("Same-binary A/B replication: is LongBench row churn nondeterminism, or code change?")
    for label, an, bn, logn in PAIRS:
        report_pair(label, an, bn, logn)
    print("\nHow to read this:")
    print("  'pred changed' is the fraction of rows whose generated TEXT differs between two runs")
    print("  of the same binary. 'score changed' is the fraction whose SCORE differs. The second")
    print("  is the only one that threatens a table of scores; the first can be large while the")
    print("  second is ~0, because token-overlap F1 is insensitive to rewording.")
    print("  'med 1st div' is the median generated-token index of the first disagreement. A value")
    print("  well above 0 rules out input handling (truncation/tokenization/BOS) as the cause.")
    print("  dAVG is the paired per-row mean difference; its CI is the harness noise floor and no")
    print("  model-vs-model margin smaller than it should be reported as real.")


if __name__ == "__main__":
    main()

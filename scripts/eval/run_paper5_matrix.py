#!/usr/bin/env python3
"""Evaluate a RELAY checkpoint on the five-benchmark paper suite.

Runs the four multiple-choice benchmarks (MMLU-Redux, MMLU-Pro, GPQA-Diamond,
C-Eval) through the local pre-tokenized .npz harness, in both sampling modes
that a chunked checkpoint supports:

  traj    - sample_trajectory_cfg (the native mode for traj32x16 checkpoints)
  singlez - sample_ddim_cfg (single latent)

Both scripts score the frozen backbone (acc_raw_suffix) and the relay-injected
model (acc_cfg1) in one pass, so the raw control comes for free.

Work is split into shards; each shard is a directory of symlinks to the source
.npz files (the eval scripts glob a flat directory and have no shard flag).
Shards are then distributed over the visible GPUs with longest-processing-time
first balancing, and merged back into one JSON per (benchmark, mode).

Run from the laces-reasoning repo root: load_relay_model resolves the
checkpoint's relative rwkv_local_path against the process CWD.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

BENCHMARKS = {
    # name: npz dir relative to the data root
    "mmlu_redux": "mmlu_redux/test",
    "mmlu_pro": "mmlu_pro/test",
    "gpqa_diamond": "gpqa_diamond/test",
    "ceval": "ceval/val",
}

# Measured seconds per file on 1xH100, from the champ-dynamic smoke run
# (2026-09-20, 20 files per benchmark). Each entry is the sum of the raw pass and
# the cfg1 injected pass, which the eval scripts run back to back and report
# separately in elapsed_raw_suffix_s / elapsed_cfg1_s. These exclude the ~50 s
# model load per process. Re-measure if the checkpoint or sequence mix changes.
SEC_PER_FILE = {
    #                singlez   traj
    "mmlu_redux": (3.22, 6.54),
    "mmlu_pro": (4.38, 16.44),
    "gpqa_diamond": (3.34, 11.45),
    "ceval": (3.25, 10.75),
}

MODE_SCRIPT = {
    "traj": "scripts/eval/eval_mcpack_trajectory_cfg.py",
    "singlez": "scripts/eval/eval_prefix_suffix_cfg_multichoice.py",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--tag", required=True, help="label for the output directory")
    p.add_argument("--data_root", default="/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data")
    p.add_argument("--out_root", default=None, help="default results/benchmarks/paper5_<tag>")
    p.add_argument("--work_root", default=None, help="default /tmp/paper5_<tag>_work")
    p.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    p.add_argument("--modes", default="traj,singlez")
    p.add_argument("--pairs", default=None,
                   help="explicit benchmark:mode units, e.g. 'mmlu_pro:traj,ceval:singlez'. "
                        "Lets one run be split across concurrent jobs; overrides --benchmarks/--modes.")
    p.add_argument("--target_shard_seconds", type=float, default=2400.0)
    p.add_argument("--max_samples", type=int, default=None, help="cap files per benchmark (smoke runs)")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--scales", default="1")
    p.add_argument("--gpus", default=None, help="comma list; default = all visible")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_shards", action="store_true")
    p.add_argument("--shard_mod", default=None,
                   help="K:R — only run multiple-choice shards whose index %% K == R. Lets one "
                        "--pairs slice be split across two smaller concurrent jobs; shard "
                        "numbering stays identical so filenames and --resume still line up.")
    p.add_argument("--resume", action="store_true",
                   help="skip units whose output JSON is already complete (survives a restart)")
    # GSM8K is opt-in: it is far more expensive per item than the MC benchmarks,
    # so a smoke run must not pull in the full 1319-item suite by default.
    p.add_argument("--gsm8k_n", type=int, default=0, help="items per GSM8K arm; 0 disables (default)")
    p.add_argument("--gsm8k_shards", type=int, default=8)
    return p.parse_args()


def _write_json_atomic(path: Path, obj) -> None:
    """Write via a temp file + rename so concurrent jobs can't tear each other's JSON."""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def visible_gpus() -> list[int]:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    except Exception:
        # login nodes carry a non-executable nvidia-smi stub; fall back to one lane
        return [0]
    n = sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))
    return list(range(n)) or [0]


def make_shards(task_dir: Path, work: Path, n_files: int, shard_cost_s: float, per_file_s: float) -> list[Path]:
    """Symlink the first n_files .npz into shard dirs of roughly shard_cost_s each."""
    files = sorted(task_dir.glob("*.npz"))[:n_files]
    n_shards = max(1, min(len(files), math.ceil(len(files) * per_file_s / shard_cost_s)))
    shards = []
    for i in range(n_shards):
        d = work / f"shard_{i:02d}of{n_shards:02d}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for f in files[i::n_shards]:  # round-robin keeps shards even in length mix
            (d / f.name).symlink_to(f.resolve())
        shards.append(d)
    return shards


def main():
    args = parse_args()
    out_root = Path(args.out_root or f"results/benchmarks/paper5_{args.tag}")
    work_root = Path(args.work_root or f"/tmp/paper5_{args.tag}_work")
    out_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else visible_gpus()
    data_root = Path(args.data_root)

    if not Path(args.ckpt_dir).is_dir():
        sys.exit(f"FATAL: checkpoint dir not found: {args.ckpt_dir}")

    # ---- build the unit list -------------------------------------------------
    units = []
    if args.pairs:
        pairs = []
        for chunk in args.pairs.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            b, _, m = chunk.partition(":")
            if b not in BENCHMARKS or m not in MODE_SCRIPT:
                sys.exit(f"FATAL: bad --pairs entry {chunk!r}; want <benchmark>:<mode>")
            pairs.append((b, m))
    else:
        pairs = [(b, m) for b in args.benchmarks.split(",") if b
                 for m in args.modes.split(",") if m]
    bench_list = sorted({b for b, _ in pairs})
    mode_list = sorted({m for _, m in pairs})

    for bname, mode in pairs:
        task_dir = data_root / BENCHMARKS[bname]
        n_avail = len(list(task_dir.glob("*.npz")))
        n_files = min(n_avail, args.max_samples) if args.max_samples else n_avail
        if n_files == 0:
            sys.exit(f"FATAL: no .npz files in {task_dir}")
        per_file_s = SEC_PER_FILE[bname][0 if mode == "singlez" else 1]
        shards = make_shards(task_dir, work_root / f"{bname}_{mode}", n_files, args.target_shard_seconds, per_file_s)
        for si, sdir in enumerate(shards):
            units.append({
                "benchmark": bname, "mode": mode, "shard": si, "n_shards": len(shards),
                "task_dir": str(sdir), "n_files": len(list(sdir.glob("*.npz"))),
                "est_s": len(list(sdir.glob("*.npz"))) * per_file_s,
                "out": str(out_root / f"{bname}_{mode}.shard{si:02d}.json"),
                "log": str(out_root / f"{bname}_{mode}.shard{si:02d}.log"),
            })
        print(f"{bname}/{mode}: {n_files} files -> {len(shards)} shards")

    units.sort(key=lambda u: -u["est_s"])

    # Restrict to a shard subset last, so shard numbering (and therefore output
    # filenames and --resume) is identical no matter how the run is split.
    # GSM8K units are unaffected: they shard by item offset, not by index modulo.
    if args.shard_mod:
        try:
            k_str, r_str = args.shard_mod.split(":", 1)
            k, r = int(k_str), int(r_str)
            if k < 1 or not (0 <= r < k):
                raise ValueError
        except ValueError:
            sys.exit(f"FATAL: bad --shard_mod {args.shard_mod!r}; want K:R with K>=1 and 0<=R<K")
        before = len(units)
        units = [u for u in units if u.get("kind") == "gsm8k" or u["shard"] % k == r]
        print(f"--shard_mod {k}:{r} selected {len(units)}/{before} units")
        if not units:
            sys.exit(f"FATAL: --shard_mod {args.shard_mod} selected no units")
    if args.gsm8k_n > 0:
        per_item = {"raw": 3.0, "singlez": 9.0, "traj": 45.0}
        units.append({"kind": "gsm8k", "varmode": "raw", "benchmark": "gsm8k", "mode": "raw",
                      "shard": 0, "n_shards": 1, "offset": 0, "n_items": args.gsm8k_n,
                      "est_s": args.gsm8k_n * per_item["raw"],
                      "out": str(out_root / "gsm8k_raw.json"), "log": str(out_root / "gsm8k_raw.log")})
        units.append({"kind": "gsm8k", "varmode": "singlez", "benchmark": "gsm8k", "mode": "singlez",
                      "shard": 0, "n_shards": 1, "offset": 0, "n_items": args.gsm8k_n,
                      "est_s": args.gsm8k_n * per_item["singlez"],
                      "out": str(out_root / "gsm8k_singlez.json"), "log": str(out_root / "gsm8k_singlez.log")})
        n_sh = max(1, args.gsm8k_shards)
        bounds = [(round(i * args.gsm8k_n / n_sh), round((i + 1) * args.gsm8k_n / n_sh)) for i in range(n_sh)]
        for i, (lo, hi) in enumerate(bounds):
            if hi <= lo:
                continue
            units.append({"kind": "gsm8k", "varmode": "traj", "benchmark": "gsm8k", "mode": "traj",
                          "shard": i, "n_shards": n_sh, "offset": lo, "n_items": hi - lo,
                          "est_s": (hi - lo) * per_item["traj"],
                          "out": str(out_root / f"gsm8k_traj.shard{i:02d}.json"),
                          "log": str(out_root / f"gsm8k_traj.shard{i:02d}.log")})
        units.sort(key=lambda u: -u["est_s"])

    total_est = sum(u["est_s"] for u in units)
    print(f"{len(units)} units, ~{total_est/3600:.1f} GPU-hours est, {len(gpus)} GPUs, "
          f"~{total_est/len(gpus)/3600:.1f} h wall if balanced")

    # ---- greedy longest-processing-time assignment ---------------------------
    lane_load = {g: 0.0 for g in gpus}
    lanes: dict[int, list[dict]] = {g: [] for g in gpus}
    for u in units:
        g = min(lane_load, key=lambda k: lane_load[k])
        lanes[g].append(u)
        lane_load[g] += u["est_s"]
    for g in gpus:
        print(f"  gpu {g}: {len(lanes[g])} units, est {lane_load[g]/3600:.2f} h")

    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")

    def build_cmd(u: dict) -> list[str]:
        if u.get("kind") == "gsm8k":
            if u["varmode"] == "raw":
                return [sys.executable, "-u", "scripts/eval/eval_gsm8k_raw.py",
                        "--ckpt_dir", args.ckpt_dir, "--task", "gsm8k", "--n", str(u["n_items"]),
                        "--device", "cuda", "--seed", str(args.seed), "--output", u["out"]]
            if u["varmode"] == "singlez":
                return [sys.executable, "-u", "scripts/eval/eval_chain_reasoning_numeric.py",
                        "--ckpt_dir", args.ckpt_dir, "--task", "gsm8k", "--n", str(u["n_items"]),
                        "--steps", str(args.steps), "--cfg_scale", "3.0", "--k_list", "1",
                        "--device", "cuda", "--output", u["out"]]
            return [sys.executable, "-u", "scripts/eval/eval_gsm8k_trajectory.py",
                    "--ckpt_dir", args.ckpt_dir, "--task", "gsm8k",
                    "--n", str(u["n_items"]), "--offset", str(u["offset"]),
                    "--steps", str(args.steps), "--cfg_scale", "3.0",
                    "--device", "cuda", "--seed", str(args.seed), "--output", u["out"]]
        return [sys.executable, "-u", MODE_SCRIPT[u["mode"]],
                "--ckpt_dir", args.ckpt_dir,
                "--task_dir", u["task_dir"],
                "--steps", str(args.steps),
                "--scales", args.scales,
                "--seed", str(args.seed),
                "--device", "cuda",
                "--output", u["out"]]

    def unit_done(u: dict) -> bool:
        """True when the unit's output file already holds a finished result.

        The MC scripts write their JSON once per pass, so a file containing
        acc_cfg1 has completed both the raw and injected passes. The GSM8K arms
        write a single summary, keyed differently per arm.
        """
        p = Path(u["out"])
        if not p.exists():
            return False
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        if u.get("kind") == "gsm8k":
            return ("accuracy" in d) or ("accuracy_by_relay_steps" in d)
        return "acc_cfg1" in d

    def run_lane(gpu: int) -> list[str]:
        failures = []
        for u in lanes[gpu]:
            label = f"{u['benchmark']}/{u.get('varmode', u['mode'])}"
            if args.resume and unit_done(u):
                print(f"[gpu {gpu}] {label} shard {u['shard']}/{u['n_shards']} already complete, skipping",
                      flush=True)
                continue
            cmd = build_cmd(u)
            t0 = time.time()
            with open(u["log"], "w") as lf:
                rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                     env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), cwd=str(REPO_ROOT))
            dt = time.time() - t0
            status = "ok" if rc == 0 else f"FAILED rc={rc}"
            print(f"[gpu {gpu}] {label} shard {u['shard']}/{u['n_shards']} "
                  f"n={u.get('n_files', u.get('n_items'))} {status} in {dt/60:.1f} min", flush=True)
            if rc != 0:
                failures.append(f"{label}#{u['shard']} rc={rc}")
        return failures

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        all_failures = [f for res in pool.map(run_lane, gpus) for f in res]

    # ---- merge shards per (benchmark, mode) ----------------------------------
    print("\n=== MERGED RESULTS ===")
    merged_all = {}
    for bname, mode in pairs:
            parts = sorted(out_root.glob(f"{bname}_{mode}.shard*.json"))
            if not parts:
                continue
            total = {}
            elapsed = {}
            n = 0
            base = None
            for p in parts:
                d = json.loads(p.read_text())
                base = base or {k: v for k, v in d.items() if k in ("ckpt_dir", "step", "steps", "scales", "mode")}
                n += d.get("n_files", 0)
                for k, v in d.items():
                    if k.startswith("correct_"):
                        total[k] = total.get(k, 0) + int(v)
                    elif k.startswith("elapsed_"):
                        elapsed[k] = elapsed.get(k, 0.0) + float(v)
            merged = dict(base or {})
            merged.update({"benchmark": bname, "sampling": mode, "n_files": n,
                           "n_shards": len(parts), "task_dir": str(data_root / BENCHMARKS[bname])})
            for k, v in total.items():
                merged[f"acc_{k[len('correct_'):]}"] = v / max(1, n)
                merged[k] = v
            merged.update(elapsed)
            merged_all[f"{bname}_{mode}"] = merged
            (out_root / f"{bname}_{mode}.json").write_text(json.dumps(merged, indent=2))
            raw = merged.get("acc_raw_suffix")
            cfg = merged.get("acc_cfg1")
            print(f"{bname:14s} {mode:8s} n={n:6d}  raw={raw if raw is None else f'{raw:.4f}'}"
                  f"  cfg1={cfg if cfg is None else f'{cfg:.4f}'}")

    # ---- GSM8K: merge the trajectory shards, then report all three arms --------
    if args.gsm8k_n > 0:
        parts = sorted(out_root.glob("gsm8k_traj.shard*.json"))
        if parts:
            tot = sum(json.loads(p.read_text())["correct"] for p in parts)
            n = sum(json.loads(p.read_text())["n"] for p in parts)
            base = json.loads(parts[0].read_text())
            base.update({"method": "trajectory_cfg", "sampling": "trajectory",
                         "n": n, "correct": tot, "accuracy": tot / max(1, n),
                         "n_shards": len(parts)})
            (out_root / "gsm8k_traj.json").write_text(json.dumps(base, indent=2))
        for arm in ("raw", "singlez", "traj"):
            f = out_root / f"gsm8k_{arm}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            # eval_chain_reasoning_numeric.py reports per-K accuracies under
            # accuracy_by_relay_steps instead of a top-level accuracy, so read
            # that shape before falling back.
            acc = d.get("accuracy")
            if acc is None:
                by_k = d.get("accuracy_by_relay_steps") or {}
                k1 = by_k.get("1", by_k.get(1))
                acc = k1
                if k1 is not None:
                    d["accuracy_at_K1"] = k1
                    d["accuracy_by_relay_steps"] = by_k
            d.setdefault("accuracy", acc)
            merged_all[f"gsm8k_{arm}"] = d
            shown = "n/a" if acc is None else f"{acc:.4f}"
            print(f"{'gsm8k':14s} {arm:8s} n={d.get('n'):6d}  acc={shown}")

    # summary.json composes across concurrent jobs: pick up any per-pair result
    # file already on disk that this invocation did not produce itself.
    for p in sorted(out_root.glob("*_*.json")):
        key = p.stem
        if key in merged_all or key.endswith(".shard") or ".shard" in key:
            continue
        try:
            merged_all[key] = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    _write_json_atomic(out_root / "summary.json", merged_all)

    if not args.keep_shards:
        shutil.rmtree(work_root, ignore_errors=True)

    if all_failures:
        print(f"\nFAILURES ({len(all_failures)}): " + ", ".join(all_failures))
        return 1
    print("\nall units ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

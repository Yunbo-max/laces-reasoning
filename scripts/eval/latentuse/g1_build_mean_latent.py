#!/usr/bin/env python3
"""G1 -- build the training-mean latent that arm A2 ("Training-mean latent") needs.

Deliverable
-----------
mean_train_latent = mean over 4,096 TRAINING spans of the frozen champion encoder's
POSTERIOR MEAN (mu_head output, width 32).

The mean is taken in z space.  R(E z) != E R(z); the writer R is nonlinear, so this
script deliberately never touches `predict_states` / never averages states.  A later
job applies R to the vector saved here.

Encoder rule
------------
Reused verbatim from scripts/eval/sample_prefix_suffix_cfg.py::encode_prefix (imported,
not reimplemented):
    frozen decoder forward(output_hidden_states=True)
      -> model._pool_hidden(hidden_states[-1], attention_mask)   (masked mean)
      -> encoder_trunk -> mu_head                                (encoder_type == variational)
`encode_prefix` reads mu_head directly, so it is the deterministic posterior mean, NOT
mu + std*eps (which is what training used via _encode_pooled).

Corpus
------
The champion checkpoint's own configured training token_dir, read straight out of the
checkpoint config (cfg.data.token_dir), so the corpus is not asserted from the outside
but taken from the artefact being probed.  Pre-tokenized 512-token OpenWebText chunks.

Span policies
-------------
Each span is encoded at batch size 1 with an all-ones attention mask over exactly its
real (non-pad) tokens.  That is bit-for-bit how encode_prefix is called at eval time
(`attention_mask = torch.ones_like(input_ids)`), and it removes every padding/batching
ambiguity from the pooled mean.

  full          (PRIMARY, saved)  ids[:n_real]           -- the training span as the shard defines it
  train_suffix  (diagnostic)      ids[split:n_real]      -- the window S0 encoded as the diffusion target
  train_prefix  (diagnostic)      ids[:split]            -- the window S0 encoded as the DiT condition

`split` follows the champion's own stage-2 rule (train_state_hijacking_dit.py:728-739)
with usable_len = n_real: lo=min(max(1,min_prefix), n_real-1), hi=max(lo, n_real-min_suffix),
split = randint(lo, hi).  The two diagnostic policies are complementary windows of the same
document, so they bound how much the saved vector depends on the span convention.
Reported, not silently averaged together.

Usage
-----
  python scripts/eval/latentuse/g1_build_mean_latent.py \
      --ckpt_dir outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000 \
      --out_dir  scripts/eval/latentuse
"""
from __future__ import annotations

import argparse
import getpass
import glob
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The encoder rule itself -- imported so the pooling is not reimplemented here.
from scripts.eval.sample_prefix_suffix_cfg import encode_prefix  # noqa: E402
from scripts.eval.relay_utils import load_relay_model  # noqa: E402

N_SPANS_DEFAULT = 4096
MASTER_SEED = 42
SPLIT_SEED = 43  # derived stream for split_idx, so file choice and split choice are independent

# A span may never come from the evaluation side of the table this feeds.
FORBIDDEN_PATH_SUBSTRINGS = (
    "sciq", "commonsenseqa", "commonsense_qa", "cqa", "race",
    "results/", "/results", "benchmark", "gold", "answer",
    "valid", "validation", "test", "eval",
)
MAX_CONTENT_COLLISION_FRACTION = 0.01  # above this the mean is compromised -> hard fail


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_info(repo: Path) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                                  text=True, timeout=60).stdout.strip()
        except Exception as e:  # pragma: no cover
            return f"<error {e}>"
    return {
        "repo": str(repo),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": run("status", "--porcelain").count("\n") if run("status", "--porcelain") else 0,
    }


# ---------------------------------------------------------------------------
# Guards.  Each one prints its denominator, and each one is calibrated against
# an input that MUST trip it -- an uncalibrated guard is not a guard.
# ---------------------------------------------------------------------------

def guard_paths_are_training(paths: list[str]) -> dict:
    """No sampled span may come from an eval suite / results file / gold answer."""
    def offending(p: str) -> list[str]:
        low = p.lower()
        return [s for s in FORBIDDEN_PATH_SUBSTRINGS if s in low]

    # POSITIVE CONTROL: the guard must fire on planted eval-looking paths, including
    # ones that differ from a real path only in the forbidden token.
    planted = [
        "preprocessed_data/sciq/train/0001_tokens.npz",
        "results/benchmarks/champ/race_predictions.jsonl",
        "preprocessed_data/owt_rwkv_tokens/validation/00000001_tokens.npz",
        "data/cqa_gold_answers.json",
    ]
    fired = sum(1 for p in planted if offending(p))
    if fired != len(planted):
        raise RuntimeError(
            f"contamination guard is INERT: only {fired}/{len(planted)} planted eval paths "
            f"tripped it. Fix the guard before trusting any 'clean' verdict."
        )
    # NEGATIVE CONTROL: a real training path must NOT fire.
    benign = "preprocessed_data/owt_rwkv_tokens/train/00007455_tokens.npz"
    if offending(benign):
        raise RuntimeError(f"contamination guard has a FALSE POSITIVE on {benign}: {offending(benign)}")

    bad = {p: offending(p) for p in paths if offending(p)}
    log(f"[guard] contamination scan: {len(paths)} sampled paths checked "
        f"(guard calibrated: {fired}/{len(planted)} planted tripped, benign path clean); "
        f"offending={len(bad)}")
    if bad:
        for p, hits in list(bad.items())[:10]:
            log(f"  [guard] OFFENDING {p} -> {hits}")
        raise RuntimeError(f"{len(bad)} sampled span(s) look like evaluation data; refusing to continue.")
    return {"n_checked": len(paths), "n_offending": 0,
            "guard_positive_control": f"{fired}/{len(planted)} planted eval paths tripped",
            "guard_negative_control": "benign train path did not trip"}


def guard_encoder_weights_loaded(model, ckpt_state: dict) -> dict:
    """strict=False silently drops tensors. If mu_head were random-init the mean would
    still look perfectly plausible, so compare the live encoder against the ckpt bitwise."""
    names = [n for n in ckpt_state
             if n.split(".")[0] in ("encoder_trunk", "mu_head", "s0_input_adapter")]
    if not names:
        raise RuntimeError("no encoder tensors found in checkpoint trainable_state -- wrong ckpt?")
    live = dict(model.named_parameters())
    live.update(dict(model.named_buffers()))
    ok, bad = [], []
    for n in names:
        if n not in live:
            bad.append((n, "absent from live model"))
            continue
        a = live[n].detach().cpu()
        b = ckpt_state[n].detach().cpu().to(a.dtype)
        if a.shape != b.shape:
            bad.append((n, f"shape {tuple(a.shape)} vs ckpt {tuple(b.shape)}"))
        elif not torch.equal(a, b):
            bad.append((n, f"values differ (max|d|={(a.float()-b.float()).abs().max():.3e})"))
        else:
            ok.append(n)
    log(f"[guard] encoder weights: {len(ok)}/{len(names)} tensors bitwise-identical to checkpoint")
    # Calibration: the comparison must be able to fail.  Perturb a copy and re-run the
    # same predicate, otherwise "all identical" could just mean the check is a no-op.
    probe = live[names[0]].detach().cpu().clone()
    probe_ref = ckpt_state[names[0]].detach().cpu().to(probe.dtype)
    perturbed = probe.clone().float()
    flat = perturbed.reshape(-1)
    flat[0] = flat[0] + 1.0
    can_fail = not torch.equal(perturbed.reshape(probe.shape).to(probe.dtype), probe_ref)
    log(f"[guard] encoder-weight comparison can distinguish a 1-element change: {can_fail}")
    if not can_fail:
        raise RuntimeError("encoder-weight equality check is INERT (cannot detect a perturbation).")
    if bad:
        for n, why in bad:
            log(f"  [guard] ENCODER TENSOR NOT LOADED: {n}: {why}")
        raise RuntimeError(f"{len(bad)}/{len(names)} encoder tensors did not load from the checkpoint.")
    return {"n_encoder_tensors": len(names), "n_bitwise_identical": len(ok),
            "tensor_names": names, "comparison_can_fail": bool(can_fail)}


# ---------------------------------------------------------------------------

def load_span(path: str) -> tuple[np.ndarray, int, bool]:
    d = np.load(path)
    ids = d["input_ids"].astype(np.int64)
    am = d["attention_mask"].astype(bool)
    n_real = int(am.sum())
    # padding must be a right-hand run, otherwise ids[:n_real] is not the real span
    contiguous_prefix = bool(am[:n_real].all() and not am[n_real:].any())
    return ids, n_real, contiguous_prefix


@torch.no_grad()
def encode_one(model, ids_1d: np.ndarray, device: str) -> torch.Tensor:
    """encode_prefix on a single span, all-ones mask, no padding -- exactly the eval call."""
    input_ids = torch.from_numpy(np.ascontiguousarray(ids_1d)).long().unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    z = encode_prefix(model, input_ids, attention_mask)
    return z.detach().float().squeeze(0).cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=str(
        REPO_ROOT / "outputs_dynamic_basis"
        / "laces-2.9B-dynlowrank-r32-global-joint-b16-50k" / "step_00050000"))
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--n_spans", type=int, default=N_SPANS_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--token_dir", default=None,
                    help="override; default is the token_dir recorded IN the checkpoint config")
    ap.add_argument("--skip_ckpt_sha", action="store_true",
                    help="smoke only -- provenance then carries no ckpt hash")
    args = ap.parse_args()

    t_wall0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"=== G1 training-mean latent ===")
    log(f"host={socket.gethostname()} user={getpass.getuser()} python={platform.python_version()} "
        f"torch={torch.__version__} device={args.device}")
    log(f"ckpt_dir={args.ckpt_dir}")

    # ---- model -----------------------------------------------------------
    t0 = time.time()
    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    cfg_any = cast(Any, cfg)
    model_any = cast(Any, model)
    model.eval()
    log(f"[load] model loaded in {time.time()-t0:.1f}s; "
        f"encoder_type={model_any.encoder_type} latent_dim={model_any.latent_dim} "
        f"ckpt_step={ckpt.get('step')}")
    if model_any.encoder_type != "variational":
        raise RuntimeError(f"expected encoder_type=variational, got {model_any.encoder_type}")
    if int(model_any.mu_head.out_features) != 32:
        raise RuntimeError(f"mu_head width is {model_any.mu_head.out_features}, expected 32")

    enc_guard = guard_encoder_weights_loaded(model, ckpt["trainable_state"])

    # logvar is what distinguishes posterior mean from a sampled z -- record it.
    lv_b = ckpt["trainable_state"]["logvar_head.bias"].float()
    lv_w = ckpt["trainable_state"]["logvar_head.weight"].float()
    logvar_info = {
        "logvar_head_bias_unique": sorted({round(float(v), 6) for v in lv_b.tolist()}),
        "logvar_head_weight_all_zero": bool(torch.count_nonzero(lv_w) == 0),
        "implied_posterior_std_if_weight_zero": float((0.5 * lv_b).exp().mean()),
    }
    log(f"[info] logvar_head.bias unique={logvar_info['logvar_head_bias_unique']} "
        f"weight_all_zero={logvar_info['logvar_head_weight_all_zero']} "
        f"=> training z was mu + {logvar_info['implied_posterior_std_if_weight_zero']:.4f}*eps; "
        f"we take mu (posterior_mean)")

    # ---- corpus / split --------------------------------------------------
    token_dir = args.token_dir or str(cfg_any.data.token_dir)
    token_dir = os.path.abspath(token_dir)
    split_string = os.path.basename(token_dir.rstrip("/"))
    log(f"[corpus] token_dir (from checkpoint config) = {token_dir}")
    log(f"[corpus] SPLIT STRING = '{split_string}'")
    if split_string != "train":
        raise RuntimeError(f"split must be 'train', got '{split_string}' from {token_dir}")
    if not os.path.isdir(token_dir):
        raise RuntimeError(f"token_dir does not exist: {token_dir}")

    all_files = sorted(glob.glob(os.path.join(token_dir, "*_tokens.npz")))
    n_pop = len(all_files)
    log(f"[corpus] examples in split '{split_string}': {n_pop}   <-- denominator")
    if n_pop < args.n_spans:
        raise RuntimeError(f"need {args.n_spans} spans but split holds only {n_pop}")

    # ---- span sampling ---------------------------------------------------
    rng_files = random.Random(MASTER_SEED)
    sampled = rng_files.sample(all_files, args.n_spans)   # sample() => distinct FILES
    if len(set(sampled)) != args.n_spans:
        raise RuntimeError("file-level distinctness violated")
    log(f"[sample] seed={MASTER_SEED} drew {len(sampled)} distinct files from {n_pop}")
    path_guard = guard_paths_are_training(sampled)

    rng_split = random.Random(SPLIT_SEED)
    min_prefix = int(cfg_any.training.get("prefix_suffix_min_prefix", 32))
    min_suffix = int(cfg_any.training.get("prefix_suffix_min_suffix", 32))
    log(f"[sample] training split rule: min_prefix={min_prefix} min_suffix={min_suffix} "
        f"(from checkpoint config)")

    records = []
    n_noncontig = 0
    for p in sampled:
        ids, n_real, contig = load_span(p)
        if not contig:
            n_noncontig += 1
        usable = max(3, n_real)
        lo = min(max(1, min_prefix), max(1, usable - 1))
        hi = max(lo, usable - max(1, min_suffix))
        split = max(1, usable // 2) if hi <= lo else rng_split.randint(lo, hi)
        records.append({"path": p, "id": os.path.basename(p), "n_real": n_real, "split_idx": int(split),
                        "ids": ids})
    if n_noncontig:
        raise RuntimeError(f"{n_noncontig}/{len(records)} spans have non-right-side padding; "
                           f"ids[:n_real] would not be the real span.")
    log(f"[sample] all {len(records)} spans have right-side padding only "
        f"(so ids[:n_real] is the real span)")

    lens = np.array([r["n_real"] for r in records])
    log(f"[sample] n_real tokens: min={lens.min()} max={lens.max()} mean={lens.mean():.1f}")

    # ---- distinctness by content ----------------------------------------
    hashes = {}
    for r in records:
        h = hashlib.sha256(r["ids"][: r["n_real"]].astype(np.int32).tobytes()).hexdigest()
        r["token_sha256"] = h
        hashes.setdefault(h, []).append(r["id"])
    n_unique = len(hashes)
    collisions = {h: v for h, v in hashes.items() if len(v) > 1}
    n_colliding_spans = sum(len(v) for v in collisions.values())
    log(f"[distinct] token-id hashing over {len(records)} spans: {n_unique} unique digests, "
        f"{len(collisions)} colliding digest(s) covering {n_colliding_spans} span(s)")
    # calibration: hashing must be able to see a duplicate at all
    dup_probe = hashlib.sha256(records[0]["ids"][: records[0]["n_real"]].astype(np.int32).tobytes()).hexdigest()
    if dup_probe != records[0]["token_sha256"]:
        raise RuntimeError("token hashing is not reproducible within one process")
    log(f"[distinct] hash reproducibility probe: OK (re-hash of span 0 matches)")
    if collisions:
        for h, v in list(collisions.items())[:5]:
            log(f"  [distinct] COLLISION {h[:16]}... -> {v}")
        if n_colliding_spans > MAX_CONTENT_COLLISION_FRACTION * len(records):
            raise RuntimeError(f"{n_colliding_spans} duplicated spans exceeds "
                               f"{MAX_CONTENT_COLLISION_FRACTION:.0%} -- mean would be biased.")
        log(f"  [distinct] NOT deduplicated (reported, kept, per protocol)")

    # ---- positive control: determinism ----------------------------------
    log("[posctl] determinism of the posterior mean (same span encoded twice)")
    pos_rows = []
    for r in records[:8]:
        z1 = encode_one(model, r["ids"][: r["n_real"]], args.device)
        z2 = encode_one(model, r["ids"][: r["n_real"]], args.device)
        exact = bool(torch.equal(z1, z2))
        maxd = float((z1 - z2).abs().max())
        pos_rows.append({"id": r["id"], "bitwise_identical": exact, "max_abs_diff": maxd,
                         "z_l2": float(z1.norm())})
        log(f"  [posctl] {r['id']}: bitwise_identical={exact} max|d|={maxd:.3e} ||z||={z1.norm():.4f}")
    n_det = sum(1 for x in pos_rows if x["bitwise_identical"])
    log(f"[posctl] {n_det}/{len(pos_rows)} spans re-encoded bitwise identically")
    # calibration: the comparison must be able to fail
    z_a = encode_one(model, records[0]["ids"][: records[0]["n_real"]], args.device)
    z_b = encode_one(model, records[1]["ids"][: records[1]["n_real"]], args.device)
    zdiff_ok = not torch.equal(z_a, z_b)
    log(f"[posctl] comparison can distinguish two DIFFERENT spans: {zdiff_ok} "
        f"(if False the equality test is vacuous)")
    if not zdiff_ok:
        raise RuntimeError("determinism check is vacuous: two different spans compare equal.")
    if n_det != len(pos_rows):
        raise RuntimeError("encoder is NOT deterministic at the posterior mean -- something is "
                           "sampling when it should not be. Stopping (see brief).")

    # ---- main pass -------------------------------------------------------
    policies = ("full", "train_suffix", "train_prefix")
    Z = {k: [] for k in policies}
    t0 = time.time()
    for i, r in enumerate(records):
        ids, n_real, split = r["ids"], r["n_real"], r["split_idx"]
        Z["full"].append(encode_one(model, ids[:n_real], args.device))
        Z["train_suffix"].append(encode_one(model, ids[split:n_real], args.device))
        Z["train_prefix"].append(encode_one(model, ids[:split], args.device))
        if (i + 1) % 256 == 0 or i + 1 == len(records):
            el = time.time() - t0
            log(f"[encode] {i+1}/{len(records)} spans x {len(policies)} policies "
                f"elapsed={el:.0f}s eta={el/(i+1)*(len(records)-i-1):.0f}s")
    Zt = {k: torch.stack(v) for k, v in Z.items()}      # [N, 32] float32
    for k, v in Zt.items():
        if v.shape != (len(records), 32):
            raise RuntimeError(f"policy {k}: bad shape {tuple(v.shape)}")

    # mean in z space, accumulated in float64
    means = {k: v.double().mean(dim=0) for k, v in Zt.items()}
    primary = "full"
    mean_z = means[primary]
    mean_vec = mean_z.to(torch.float32).unsqueeze(0).contiguous()   # (1, 32)

    # ---- negative controls ----------------------------------------------
    stats = {}
    for k in policies:
        v = Zt[k].double()
        m = means[k]
        ind_norms = v.norm(dim=1)
        d = (v - m.unsqueeze(0)).norm(dim=1)
        cos = torch.nn.functional.cosine_similarity(v, m.unsqueeze(0).expand_as(v), dim=1)
        stats[k] = {
            "mean_l2": float(m.norm()),
            "mean_individual_l2": float(ind_norms.mean()),
            "individual_l2_min": float(ind_norms.min()),
            "individual_l2_max": float(ind_norms.max()),
            "individual_l2_std": float(ind_norms.std()),
            "ratio_mean_over_mean_individual": float(m.norm() / ind_norms.mean()),
            "min_distance_mean_to_any_single_z": float(d.min()),
            "argmin_distance_span_id": records[int(d.argmin())]["id"],
            "max_distance_mean_to_any_single_z": float(d.max()),
            "mean_cosine_individual_to_mean": float(cos.mean()),
            "mean_is_exactly_zero": bool(torch.count_nonzero(m) == 0),
            "n_dims_exactly_zero": int((m == 0).sum()),
            "per_dim_mean": [float(x) for x in m],
            "per_dim_min": [float(x) for x in v.min(dim=0).values],
            "per_dim_max": [float(x) for x in v.max(dim=0).values],
            "per_dim_std": [float(x) for x in v.std(dim=0)],
        }
        s = stats[k]
        log(f"[negctl:{k}] ||mean||={s['mean_l2']:.6f}  mean||z||={s['mean_individual_l2']:.6f}  "
            f"ratio={s['ratio_mean_over_mean_individual']:.4f}")
        log(f"[negctl:{k}] mean is zero vector: {s['mean_is_exactly_zero']}  "
            f"(dims exactly 0: {s['n_dims_exactly_zero']}/32)")
        log(f"[negctl:{k}] min ||mean - z_i|| = {s['min_distance_mean_to_any_single_z']:.6f} "
            f"(closest span {s['argmin_distance_span_id']}); "
            f"max = {s['max_distance_mean_to_any_single_z']:.6f}")
        log(f"[negctl:{k}] mean cos(z_i, mean) = {s['mean_cosine_individual_to_mean']:.4f}")

    if stats[primary]["mean_is_exactly_zero"]:
        raise RuntimeError("mean latent is the zero vector -- that is a DIFFERENT control; refusing to save.")
    if stats[primary]["min_distance_mean_to_any_single_z"] == 0.0:
        raise RuntimeError("mean latent equals an individual span's z exactly -- suspect a 1-span mean.")

    # cross-policy sensitivity: how much does the span convention move the answer?
    cross = {}
    for a in policies:
        for b in policies:
            if a < b:
                ma, mb = means[a], means[b]
                cross[f"{a}__vs__{b}"] = {
                    "l2_distance": float((ma - mb).norm()),
                    "cosine": float(torch.nn.functional.cosine_similarity(
                        ma.unsqueeze(0), mb.unsqueeze(0), dim=1)),
                    "l2_distance_relative_to_primary_norm": float((ma - mb).norm() / mean_z.norm()),
                }
    for k, v in cross.items():
        log(f"[sensitivity] {k}: ||d||={v['l2_distance']:.6f} "
            f"({v['l2_distance_relative_to_primary_norm']:.3f} of ||mean_primary||) "
            f"cos={v['cosine']:.6f}")

    # batching/padding sensitivity: would a padded batch have given the same z?
    batch_probe = {}
    try:
        k = 8
        sub = records[:k]
        L = max(r["n_real"] for r in sub)
        bi = torch.zeros(k, L, dtype=torch.long)
        bm = torch.zeros(k, L, dtype=torch.long)
        for j, r in enumerate(sub):
            bi[j, : r["n_real"]] = torch.from_numpy(r["ids"][: r["n_real"]]).long()
            bm[j, : r["n_real"]] = 1
        zb = encode_prefix(model, bi.to(args.device), bm.to(args.device)).detach().float().cpu()
        zs = torch.stack([Zt["full"][j] for j in range(k)])
        batch_probe = {
            "k": k, "padded_batch_len": int(L),
            "max_abs_diff_vs_batch1": float((zb - zs).abs().max()),
            "mean_abs_diff_vs_batch1": float((zb - zs).abs().mean()),
            "note": "batch-1 unpadded is the primary path (matches encode_prefix at eval time); "
                    "this only quantifies what padded batching would have changed.",
        }
        log(f"[batchprobe] padded batch (k={k}, L={L}) vs batch-1: "
            f"max|d|={batch_probe['max_abs_diff_vs_batch1']:.3e} "
            f"mean|d|={batch_probe['mean_abs_diff_vs_batch1']:.3e}")
    except Exception as e:
        batch_probe = {"error": repr(e)}
        log(f"[batchprobe] skipped: {e!r}")

    # ---- save ------------------------------------------------------------
    mean_path = out_dir / "g1_mean_latent.pt"
    torch.save(mean_vec, mean_path)
    log(f"[save] {mean_path}  shape={tuple(mean_vec.shape)} dtype={mean_vec.dtype}")

    zs_path = out_dir / "g1_individual_latents.pt"
    torch.save({"policy_order": list(policies),
                "z": {k: Zt[k] for k in policies},
                "span_ids": [r["id"] for r in records],
                "n_real": [r["n_real"] for r in records],
                "split_idx": [r["split_idx"] for r in records]}, zs_path)
    log(f"[save] {zs_path} (per-span z for all policies, for audit/reuse)")

    ckpt_path = os.path.join(args.ckpt_dir, "model.pt")
    if args.skip_ckpt_sha:
        ck_sha = None
    else:
        t0 = time.time()
        ck_sha = sha256_file(ckpt_path)
        log(f"[prov] ckpt sha256={ck_sha} ({time.time()-t0:.0f}s)")

    wall = time.time() - t_wall0
    prov = {
        "deliverable": "mean_train_latent for appendix.tex tab:latent-use-pending arm A2 "
                       "(Training-mean latent)",
        "saved_tensor": str(mean_path),
        "tensor_shape": list(mean_vec.shape),
        "tensor_shape_convention": "(1, 32) -- batch-first, directly usable as predict_states(z) input",
        "tensor_dtype": str(mean_vec.dtype),
        "mean_taken_in": "z space (posterior means). R(E z) != E R(z): the writer R is NOT applied "
                         "here and no states were averaged.",
        "prefix_encoding_rule": "posterior_mean",
        "encoding_rule_source": "scripts/eval/sample_prefix_suffix_cfg.py::encode_prefix (imported, "
                                "pooling not reimplemented); frozen decoder -> _pool_hidden "
                                "(masked mean) -> encoder_trunk -> mu_head",
        "encoder_type": str(model_any.encoder_type),
        "latent_width": int(model_any.mu_head.out_features),
        "logvar": logvar_info,

        "corpus_name": "OpenWebText (pre-tokenized 512-token chunks)",
        "corpus_source": "the champion checkpoint's own configured training token_dir, read from "
                         "cfg.data.token_dir inside model.pt",
        "corpus_revision": {
            "kind": "local pre-tokenized shard directory (no HF dataset revision available)",
            "token_dir": token_dir,
            "dir_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime(os.path.getmtime(token_dir))),
            "n_files_in_dir": n_pop,
            "generating_script": "NOT identifiable in the tree: neither "
                                 "scripts/preprocess/preprocess_owt_clean.py (writes "
                                 "preprocessed_data/owt/tokens/train) nor "
                                 "preprocess_owt_for_diffrwkv.py (writes "
                                 "preprocessed_data/owt_32d/tokens/train) targets this directory. "
                                 "Provenance here is by CONTENT (per-span token sha256 below), "
                                 "not by claimed script.",
        },
        "split": split_string,
        "split_examples_count": n_pop,
        "tokenizer": {
            "source": str(cfg_any.model.rwkv_local_path),
            "class": type(tokenizer).__name__,
            "vocab_size_len": len(tokenizer),
            "note": "spans are already tokenized in the shards; this is the tokenizer the "
                    "checkpoint's backbone/vocab is tied to.",
        },

        "seed_master": MASTER_SEED,
        "seed_split_idx_stream": SPLIT_SEED,
        "sampling": f"random.Random({MASTER_SEED}).sample(sorted(glob('*_tokens.npz')), {args.n_spans})",
        "n_spans_requested": args.n_spans,
        "n_spans_used": len(records),
        "span_length_policy": {
            "primary_saved": "full: ids[:n_real] -- every real (non-pad) token of the chunk, "
                             "batch size 1, all-ones attention mask (identical to how "
                             "encode_prefix is called at eval time)",
            "diagnostic_train_suffix": "ids[split_idx:n_real] -- the window S0 encoded as the "
                                       "stage-2 diffusion target",
            "diagnostic_train_prefix": "ids[:split_idx] -- the window S0 encoded as the DiT condition",
            "split_idx_rule": f"champion stage-2 rule (train_state_hijacking_dit.py:728-739) with "
                              f"usable_len=n_real, min_prefix={min_prefix}, min_suffix={min_suffix}",
            "n_real_min": int(lens.min()), "n_real_max": int(lens.max()),
            "n_real_mean": float(lens.mean()),
            "padding_side_verified": "right-side only for all spans",
        },

        "controls": {
            "positive_control_determinism": {
                "description": "same span encoded twice must give bitwise-identical z",
                "n_spans_tested": len(pos_rows),
                "n_bitwise_identical": n_det,
                "rows": pos_rows,
                "calibration_can_distinguish_different_spans": bool(zdiff_ok),
            },
            "negative_control_not_zero_not_single": {
                k: {kk: vv for kk, vv in stats[k].items() if not kk.startswith("per_dim_")}
                for k in policies
            },
            "distinctness": {
                "method": "sha256 of int32 token ids over ids[:n_real]",
                "n_spans": len(records),
                "n_unique_digests": n_unique,
                "n_colliding_digests": len(collisions),
                "n_spans_in_collisions": n_colliding_spans,
                "collisions": {h: v for h, v in list(collisions.items())[:50]},
                "deduplicated": False,
                "hash_reproducibility_probe": "OK",
            },
            "split_check": {
                "split_string_loaded": split_string,
                "examples_in_split": n_pop,
                "denominator_printed": True,
            },
            "contamination_guard": path_guard,
            "encoder_weights_guard": enc_guard,
            "span_policy_sensitivity": cross,
            "padded_batch_probe": batch_probe,
        },

        "per_dim_stats_over_individual_z": {
            k: {"min": stats[k]["per_dim_min"], "max": stats[k]["per_dim_max"],
                "std": stats[k]["per_dim_std"], "mean": stats[k]["per_dim_mean"]}
            for k in policies
        },
        "mean_l2_norm": stats[primary]["mean_l2"],
        "mean_of_individual_l2_norms": stats[primary]["mean_individual_l2"],
        "primary_policy": primary,
        "mean_vector": [float(x) for x in mean_z],
        "mean_vector_all_policies": {k: [float(x) for x in means[k]] for k in policies},

        "checkpoint": {
            "ckpt_dir": os.path.abspath(args.ckpt_dir),
            "model_pt": os.path.abspath(ckpt_path),
            "sha256": ck_sha,
            "size_bytes": os.path.getsize(ckpt_path),
            "step": ckpt.get("step"),
        },
        "git": git_info(REPO_ROOT),
        "span_identifiers": [
            {"id": r["id"], "n_real": r["n_real"], "split_idx": r["split_idx"],
             "token_sha256": r["token_sha256"]}
            for r in records
        ],
        "environment": {
            "host": socket.gethostname(), "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_HOME": os.environ.get("HF_HOME"),
        },
        "wall_clock_seconds": wall,
        "wall_clock_human": f"{int(wall)//60}m{int(wall)%60}s",
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    prov_path = out_dir / "g1_mean_latent_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)
    log(f"[save] {prov_path}")

    log("=== SUMMARY ===")
    log(f"saved            : {mean_path} shape={tuple(mean_vec.shape)}")
    log(f"corpus/split     : {token_dir}  split='{split_string}' (n={n_pop})")
    log(f"spans            : {len(records)} distinct (unique digests {n_unique}), seed {MASTER_SEED}")
    log(f"||mean||         : {stats[primary]['mean_l2']:.6f}")
    log(f"mean ||z||       : {stats[primary]['mean_individual_l2']:.6f}")
    log(f"ratio            : {stats[primary]['ratio_mean_over_mean_individual']:.4f}")
    log(f"determinism      : {n_det}/{len(pos_rows)} bitwise identical")
    log(f"wall clock       : {int(wall)//60}m{int(wall)%60}s")
    log("=== G1 DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

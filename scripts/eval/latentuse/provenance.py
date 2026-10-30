#!/usr/bin/env python3
"""D1: record the provenance of the single-plan champion checkpoint.

Everything here is COMPUTED from the artefacts on disk, never copied from a
note or a README.  Run on CPU; it loads the checkpoint's ``trainable_state``
but never builds the decoder, so no CUDA is required.

    python3 scripts/eval/latentuse/provenance.py \
        --ckpt outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000 \
        --out  scripts/eval/latentuse/provenance.json

Why ``global_denoiser_type`` presence matters: ``inject_into_cache``
(models/state_hijacking_dit.py, the ``is_rnn_denoiser`` test) branches on it.
When the key is ABSENT the attribute defaults to ``"dit"``, the branch takes
the non-RNN path, and ``conv_state``/``ffn_state`` are ZEROED on every write.
That zeroing is the entire A0-vs-A1 contrast, so the flag has to be recorded as
PRESENT/ABSENT rather than as a value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

CKPT_DEFAULT = (
    "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-global-joint-b16-50k/step_00050000"
)

# Tensors whose shapes pin the writer geometry.  A missing one is an error, not
# a blank cell: the five-arm runner multiplies through these shapes.
REQUIRED_SHAPES = [
    "s1_u_head.weight",
    "s1_v_head.weight",
    "s1_trunk.0.weight",
    "mu_head.weight",
    "state_basis",
]


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    porcelain = run("status", "--porcelain")
    return {
        "repo": str(repo),
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(porcelain),
        "n_dirty_paths": len([ln for ln in porcelain.splitlines() if ln.strip()]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT_DEFAULT,
                    help="checkpoint DIRECTORY containing model.pt")
    ap.add_argument("--out", default=str(Path(__file__).with_name("provenance.json")))
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt)
    if not ckpt_dir.is_absolute():
        ckpt_dir = REPO / ckpt_dir
    model_pt = ckpt_dir / "model.pt"
    if not model_pt.is_file():
        raise SystemExit(f"no model.pt under {ckpt_dir}")

    st = model_pt.stat()
    print(f"[provenance] hashing {model_pt} ({st.st_size} bytes) ...", flush=True)
    ckpt_sha = sha256_file(model_pt)
    print(f"[provenance] sha256={ckpt_sha}", flush=True)

    ck = torch.load(model_pt, map_location="cpu", weights_only=False)
    # cfg is stored as a YAML *string* in this checkpoint family, not a dict.
    cfg = OmegaConf.create(ck["config"])
    mcfg = cfg.model

    sd = ck["trainable_state"]

    shapes: dict[str, Any] = {}
    for key in REQUIRED_SHAPES:
        if key not in sd:
            raise KeyError(
                f"{key} absent from trainable_state; refusing to emit a provenance "
                f"record with an unknown writer geometry (keys present: "
                f"{sorted(k for k in sd if k.startswith(key.split('.')[0]))})"
            )
        shapes[key] = {"shape": list(sd[key].shape), "dtype": str(sd[key].dtype)}

    # state_scale lives both at the top level and inside trainable_state; report
    # both and assert they agree, because a silent divergence would rescale every
    # written memory by a different factor than the one the paper quotes.
    scale_sd = sd["state_scale"].float().flatten()
    if scale_sd.numel() != 1:
        raise ValueError(f"state_scale is not a scalar: numel={scale_sd.numel()}")
    state_scale = float(scale_sd[0])
    top_scale = ck.get("state_scale")
    if top_scale is not None:
        top_val = float(torch.as_tensor(top_scale).float().flatten()[0])
        if abs(top_val - state_scale) > 0:
            raise ValueError(
                f"state_scale mismatch: top-level {top_val!r} vs trainable_state {state_scale!r}"
            )

    # Resolve the decoder config from the decoder's OWN config.json.
    dec_path = Path(str(mcfg.rwkv_local_path))
    dec_json = dec_path / "config.json"
    if not dec_json.is_file():
        raise SystemExit(f"decoder config.json missing at {dec_json}")
    dec = json.loads(dec_json.read_text())
    hidden = int(dec["hidden_size"])
    head_dim = int(dec.get("head_dim", 64))
    if hidden % head_dim:
        raise ValueError(f"hidden_size={hidden} not divisible by head_dim={head_dim}")
    # config.json carries "num_heads": null; the model derives it (see
    # StateInjectionDiTRELAY.__init__, self.num_heads = hidden_size // head_dim).
    n_heads = hidden // head_dim

    # Cross-check the derived geometry against state_basis, which is the only
    # tensor that encodes (layers, heads, head_dim) all at once.
    sb = list(sd["state_basis"].shape)
    expect = [int(dec["num_hidden_layers"]), int(mcfg.get("n_basis", 16)), n_heads, head_dim, head_dim]
    if sb != expect:
        raise ValueError(
            f"state_basis shape {sb} disagrees with decoder geometry {expect}; "
            "the writer and the decoder do not describe the same model"
        )

    rec = {
        "generated_by": "scripts/eval/latentuse/provenance.py",
        "checkpoint": {
            "dir": str(ckpt_dir),
            "model_pt": str(model_pt),
            "sha256": ckpt_sha,
            "size_bytes": st.st_size,
            "mtime_epoch": st.st_mtime,
            "mtime_iso": __import__("datetime").datetime.fromtimestamp(st.st_mtime).isoformat(),
            "step_in_ckpt": ck.get("step"),
            "n_trainable_tensors": len(sd),
        },
        "git": git_state(REPO),
        "model_config": {
            "encoder_type": str(mcfg.get("encoder_type")),
            "s1_writer_type": str(mcfg.get("s1_writer_type")),
            "s1_rank": int(mcfg.get("s1_rank")),
            "s1_dyn_hidden": int(mcfg.get("s1_dyn_hidden")),
            "latent_dim": int(mcfg.get("latent_dim")),
            "n_basis": int(mcfg.get("n_basis")),
            "use_cond_boundary": bool(mcfg.get("use_cond_boundary")),
            "trajectory_mode": bool(mcfg.get("trajectory_mode")),
            # gen_type is a TRAINING field in this checkpoint family, and
            # relay_utils.load_relay_model reads it from cfg.training.
            "gen_type": str(cfg.training.get("gen_type")),
            "global_denoiser_type_present": "global_denoiser_type" in mcfg,
            "global_denoiser_type_value": (
                str(mcfg.get("global_denoiser_type")) if "global_denoiser_type" in mcfg else None
            ),
            "injection_branch": (
                "rnn(preserve conv/ffn)"
                if str(mcfg.get("global_denoiser_type", "dit")).lower() in ("rwkv", "birwkv")
                else "non-rnn(zero conv/ffn)"
            ),
        },
        "training_config": {
            "stage": cfg.training.get("stage"),
            "prefix_suffix_s2": cfg.training.get("prefix_suffix_s2"),
            "cfg_drop_prob": cfg.training.get("cfg_drop_prob"),
            "seed": cfg.training.get("seed"),
            "num_train_steps": cfg.training.get("num_train_steps"),
            "resume": cfg.training.get("resume"),
            "s2_unfreeze_s1": cfg.training.get("s2_unfreeze_s1"),
            "s2_unfreeze_s0": cfg.training.get("s2_unfreeze_s0"),
            "token_dir": cfg.data.get("token_dir"),
            "max_length": cfg.data.get("max_length"),
        },
        "state_scale": state_scale,
        "writer_shapes": shapes,
        "decoder": {
            "path": str(dec_path),
            "config_json": str(dec_json),
            "model_type": dec.get("model_type"),
            "layers": int(dec["num_hidden_layers"]),
            "d_model": hidden,
            "n_heads": n_heads,
            "n_heads_source": "derived: hidden_size // head_dim (config.json num_heads is null)",
            "head_dim": head_dim,
            "vocab_size": int(dec["vocab_size"]),
            "eos_token_id": dec.get("eos_token_id"),
            "bos_token_id": dec.get("bos_token_id"),
            "max_position_embeddings": dec.get("max_position_embeddings"),
        },
        "env": {
            "RELAY_INJECT_ZERO_RNN_CACHE": os.environ.get("RELAY_INJECT_ZERO_RNN_CACHE", "0"),
            "torch": torch.__version__,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rec, indent=2, sort_keys=True))
    print(f"[provenance] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

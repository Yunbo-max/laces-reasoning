"""S0 information-fidelity probe: where along the chain does a decisive token vanish?

For paired samples that differ only in ONE answer-deciding token (digit, operand
order, `<` vs `<=`), measure how separable the two are at each stage of the LACES
encoding chain:

    full hidden (per-position RWKV hidden)  ->  mean-pooled hidden  ->  S0 mu (32-dim)

"separable" = 1 - cosine_similarity. If pooling is the culprit, the full-hidden
min-cosine drops sharply once mean-pooled (the decisive token's delta is divided
by L). If the 32-dim projection is the culprit, the drop happens at mu instead.

Read-only: S0/RWKV are frozen; no training. Runs on qz H100.
"""
from __future__ import annotations
import sys, os, json, time
import torch

REPO = "/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/laces-reasoning"
sys.path.insert(0, REPO)
from laces_posttrain.native import load_native

CKPT = os.path.join(REPO, "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000")
RWKV = os.path.join(REPO, "models/RWKV7-Goose-World3-2.9B-HF")

# (label, text_a, text_b) — differ in one decisive token
PAIRS = [
    ("digit 16->18",      "Janet's ducks lay 16 eggs per day.", "Janet's ducks lay 18 eggs per day."),
    ("digit 42->43",      "The final answer is 42.",            "The final answer is 43."),
    ("operand a-b->b-a",  "Let x equal a minus b.",             "Let x equal b minus a."),
    ("add->sub",          "Compute three plus five.",           "Compute three minus five."),
    ("lt->le (math)",     "The rule is if x is less than 10.",  "The rule is if x is less than or equal to 10."),
    ("code digit 16->18", "def f(n): return n * 16",            "def f(n): return n * 18"),
    ("code lt->le",       "while x < limit:",                   "while x <= limit:"),
    ("code a-b->b-a",     "diff = total - cost",                "diff = cost - total"),
]


def cos(a, b):
    a = a.float().reshape(1, -1); b = b.float().reshape(1, -1)
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-9))


@torch.no_grad()
def probe(native, text):
    ids = native.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(native.device)
    mask = torch.ones_like(ids).bool()
    out = native.model.rwkv_model(input_ids=ids, attention_mask=mask,
                                  output_hidden_states=True, use_cache=True, return_dict=True)
    h = out.hidden_states[-1]                       # [1, L, D]
    pooled = native.model._pool_hidden(h, mask)     # [1, D]
    mu, _ = native.model._encode_pooled(pooled)     # [1, 32]
    return h, pooled, mu


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    native = load_native(CKPT, device, rwkv_path=RWKV, expected_step=30000,
                         expected_writer="dynlowrank", blend=0.7)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    rows = []
    for label, a, b in PAIRS:
        ha, pa, ma = probe(native, a)
        hb, pb, mb = probe(native, b)
        L = min(ha.shape[1], hb.shape[1])
        per_tok = [cos(ha[:, t], hb[:, t]) for t in range(L)]
        row = {
            "label": label,
            "full_hidden_min_cos": min(per_tok),
            "full_hidden_mean_cos": sum(per_tok) / len(per_tok),
            "pooled_cos": cos(pa, pb),
            "mu_cos": cos(ma, mb),
            # separability = 1 - cos (higher = more separable)
            "sep_full_min": 1 - min(per_tok),
            "sep_pooled": 1 - cos(pa, pb),
            "sep_mu": 1 - cos(ma, mb),
        }
        rows.append(row)
        print(f"{label:22s} sep_full_min={row['sep_full_min']:.4f}  sep_pooled={row['sep_pooled']:.4f}  sep_mu={row['sep_mu']:.4f}", flush=True)

    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "results/s0_info_fidelity.json")
    payload = {"pairs": rows, "elapsed_seconds": time.time() - t0}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}", flush=True)
    native.close()


if __name__ == "__main__":
    main()

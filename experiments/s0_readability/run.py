"""S0 readability probe: can a linear probe decode the decisive symbol from each layer?

Reads the answer-deciding symbol from (a) mean-pooled RWKV hidden, (b) the S0 32-dim
latent (via _encode_pooled). Templates split train/test so the probe cannot memorize
surface wording. Probe is a torch linear layer trained on-GPU (no sklearn dependency).

Tasks: digit(8-way), operator(+,-,*), order(a-b vs b-a), boundary(< vs <=).
Decision: pooled reads well but 32-dim fails -> fix the projection/objective;
          pooled already fails -> keep positional latent slots per EOS segment.
"""
from __future__ import annotations
import sys, os, json, time
import torch
import torch.nn.functional as F

REPO = "/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/laces-reasoning"
sys.path.insert(0, REPO)
from laces_posttrain.native import load_native

CKPT = os.path.join(REPO, "outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000")
RWKV = os.path.join(REPO, "models/RWKV7-Goose-World3-2.9B-HF")

DIGITS = [16, 18, 42, 43, 7, 8, 3, 5]
OPS = ["+", "-", "*"]

DIGIT_TRAIN = ["The number is {x}.", "We have {x} apples.", "Set n = {x}.", "The constant is {x}.",
               "There are {x} items.", "x equals {x}.", "n is {x}.", "The value is {x}."]
DIGIT_TEST  = ["The result is {x}.", "Count is {x}.", "The total is {x}.", "We need {x} units.",
               "Quantity is {x}.", "The amount is {x}.", "Size is {x}.", "The score is {x}."]
OP_TRAIN = ["Compute a {op} b.", "x = a {op} b.", "What is a {op} b?", "a {op} b equals",
            "Let z = a {op} b.", "Find a {op} b.", "a {op} b is", "Solve a {op} b."]
OP_TEST  = ["Evaluate a {op} b.", "y = a {op} b.", "a {op} b gives", "the result of a {op} b",
            "p = a {op} b.", "Calculate a {op} b.", "value of a {op} b", "a {op} b yields"]
ORDER_TRAIN = ["Let x equal {w}.", "We compute {w}.", "The expression is {w}.", "{w} is the result.",
               "Set y to {w}.", "Consider {w}.", "Find {w}.", "The value is {w}."]
ORDER_TEST  = ["Define x as {w}.", "We need {w}.", "Take {w}.", "The answer is {w}.",
               "Let p be {w}.", "Compute {w}.", "Evaluate {w}.", "The result is {w}."]
BOUND_TRAIN = ["The rule is if x is {c} 10.", "We run while x {c} 10.", "Check x {c} 10 first.",
               "The guard is x {c} 10.", "Loop if x {c} 10.", "Stop when x {c} 10.",
               "Condition: x {c} 10.", "If x {c} 10 then go."]
BOUND_TEST  = ["Test whether x {c} 10.", "We stop at x {c} 10.", "The limit is x {c} 10.",
               "Proceed if x {c} 10.", "Exit when x {c} 10.", "The check is x {c} 10.",
               "Gate: x {c} 10.", "Wait while x {c} 10."]


def make_samples(templates, values):
    out = []
    for v in values:
        for t in templates:
            out.append(t.format(**v))
    return out


def encode(native, texts):
    pooled_list, z_list = [], []
    for txt in texts:
        ids = native.tokenizer(txt, return_tensors="pt", add_special_tokens=False).input_ids.to(native.device)
        mask = torch.ones_like(ids).bool()
        with torch.no_grad():
            out = native.model.rwkv_model(input_ids=ids, attention_mask=mask,
                                          output_hidden_states=True, use_cache=True, return_dict=True)
            h = out.hidden_states[-1]
            pooled = native.model._pool_hidden(h, mask)
            z, _ = native.model._encode_pooled(pooled)
        pooled_list.append(pooled.squeeze(0))
        z_list.append(z.squeeze(0))
    return torch.stack(pooled_list), torch.stack(z_list)


def train_probe(Xtr, ytr, Xte, yte, n_classes, device, epochs=300):
    Xtr = Xtr.float().to(device); Xte = Xte.float().to(device)
    ytr = ytr.to(device); yte = yte.to(device)
    lin = torch.nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(epochs):
        loss = F.cross_entropy(lin(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        a = (lin(Xtr).argmax(-1) == ytr).float().mean().item()
        b = (lin(Xte).argmax(-1) == yte).float().mean().item()
    return a, b


def run_task(native, name, train_texts, test_texts, labels_tr, labels_te, n_classes):
    ytr = torch.tensor(labels_tr); yte = torch.tensor(labels_te)
    pool_tr, z_tr = encode(native, train_texts)
    pool_te, z_te = encode(native, test_texts)
    res = {"task": name, "n_train": len(train_texts), "n_test": len(test_texts), "n_classes": n_classes}
    for feat, Xtr, Xte in [("pooled", pool_tr, pool_te), ("z32", z_tr, z_te)]:
        a, b = train_probe(Xtr, ytr, Xte, yte, n_classes, native.device)
        res[f"{feat}_train_acc"] = a
        res[f"{feat}_test_acc"] = b
    print(f"{name:10s} pooled={res['pooled_test_acc']:.3f} z32={res['z32_test_acc']:.3f}", flush=True)
    return res


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    native = load_native(CKPT, device, rwkv_path=RWKV, expected_step=30000,
                         expected_writer="dynlowrank", blend=0.7)
    print(f"loaded in {time.time()-t0:.1f}s device={native.device}", flush=True)
    rows = []
    rows.append(run_task(native, "digit",
                         make_samples(DIGIT_TRAIN, [{"x": d} for d in DIGITS]),
                         make_samples(DIGIT_TEST, [{"x": d} for d in DIGITS]),
                         [i for i in range(len(DIGITS)) for _ in DIGIT_TRAIN],
                         [i for i in range(len(DIGITS)) for _ in DIGIT_TEST], len(DIGITS)))
    rows.append(run_task(native, "operator",
                         make_samples(OP_TRAIN, [{"op": o} for o in OPS]),
                         make_samples(OP_TEST, [{"op": o} for o in OPS]),
                         [i for i in range(len(OPS)) for _ in OP_TRAIN],
                         [i for i in range(len(OPS)) for _ in OP_TEST], len(OPS)))
    rows.append(run_task(native, "order",
                         make_samples(ORDER_TRAIN, [{"w": "a minus b"}, {"w": "b minus a"}]),
                         make_samples(ORDER_TEST, [{"w": "a minus b"}, {"w": "b minus a"}]),
                         [0]*len(ORDER_TRAIN) + [1]*len(ORDER_TRAIN),
                         [0]*len(ORDER_TEST) + [1]*len(ORDER_TEST), 2))
    rows.append(run_task(native, "boundary",
                         make_samples(BOUND_TRAIN, [{"c": "less than"}, {"c": "less than or equal to"}]),
                         make_samples(BOUND_TEST, [{"c": "less than"}, {"c": "less than or equal to"}]),
                         [0]*len(BOUND_TRAIN) + [1]*len(BOUND_TRAIN),
                         [0]*len(BOUND_TEST) + [1]*len(BOUND_TEST), 2))
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "results/s0_readability.json")
    payload = {"tasks": rows, "elapsed_seconds": time.time() - t0}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}", flush=True)
    native.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        with open(os.path.join(REPO, "results/s0_readability_err.txt"), "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        raise

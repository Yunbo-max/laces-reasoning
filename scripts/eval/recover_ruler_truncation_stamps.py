#!/usr/bin/env python3
"""Recover the missing truncation stamps for the 4 cells a peer reported as unauditable.

The peer's finding: champ-dynamic-30k/global-dit-50k `16384_vt` and qwen2.5-{1.5b,3b} `16384_cwe`
have `truncation_reason` absent, `effective_input_budget` null and no `prompt_tokens_untruncated`,
yet `input_tokens` is pinned (distinct==1) at 16384. Their conclusion was that nothing on disk
distinguishes "not truncated" from "not recorded".

That conclusion is right about those FILES but wrong about the information: the gap is a runner
VERSION gap (those four were written 07:13:30; the stamp keys first appear at 09:23:24 and are
present in everything from 10:33 on), and the inputs that determine truncation -- the RULER parquet,
the prompt construction, the tokenizer and the cap -- are all still on disk. So recompute the
untruncated prompt length directly and settle each cell.

WHY ONLY FOUR CASES, when 45 cells lack `prompt_tokens_untruncated` (peer's sweep, verified):
there is a cheap arithmetic discriminator that settles 41 of them with no tokenizer at all --

    if max(input_tokens) < CAP:  nothing was cut, seen_fraction = 1.0000, done.

`input_tokens` is recorded even in the old files, and the runner clamps to the cap, so a cell whose
maximum sits strictly below the cap cannot have been truncated. Only cells PINNED at the cap
(`distinct == 1` and `max == CAP`) are ambiguous, and there are exactly four of those. Run the cheap
test first; this script is only for the residue. (llama-3.2-3b's "untruncated" verdict also falls
out of the cheap test: its maxima are 16265/16323/15687, all < 16384.)

Zero GPU: tokenizer only, no model is loaded.

PROVENANCE OF THIS FILE: the first committed version raised `NameError` on a debug print and could
not run, even though the numbers quoted from it were correct -- I had run a working copy by hand and
committed a different one. A peer caught it. The fix is not just the missing name: `self_check()`
below now re-derives a stamped oracle cell on every run and hard-exits on mismatch, so this file can
never again print numbers it has not just validated.
"""
import json, sys
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

RULER = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/"
             "datasets--simonjegou--ruler/snapshots/24adceac8a0e6532936e8d721cd9e9084d2e4686")
CAP = 16384
# NOTE: rwkv7's stamped config has reserved_for_generation=128 but
# effective_input_budget=16384, i.e. the reserve is NOT subtracted from the input cap.
# Do not "helpfully" subtract it -- that would invent a 16256 budget that the runner
# never applied and shift every recovered seen_fraction.
EFFECTIVE = CAP

# (label, tokenizer_dir, cell, length_dir, task, poswin_from_config)
CASES = [
    ("champ-dynamic-30k", "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/"
     "laces-reasoning/models/RWKV7-Goose-World3-2.9B-HF-fla-v2", "16384_vt", "16384", "vt", None),
    ("global-dit-50k", "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/"
     "laces-reasoning/models/RWKV7-Goose-World3-2.9B-HF-fla-v2", "16384_vt", "16384", "vt", None),
    ("qwen2.5-1.5b", "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/qwen2.5-1.5b",
     "16384_cwe", "16384", "cwe", 131072),
    ("qwen2.5-3b", "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/Qwen2.5-3B",
     "16384_cwe", "16384", "cwe", 32768),
]

def prompts_for(length_dir, task, n):
    """Exactly the runner's construction: context + "\\n" + question + "\\n" + answer_prefix."""
    files = sorted((RULER / length_dir).glob("*.parquet"))
    assert files, f"no parquet in {RULER/length_dir}"
    out = []
    for f in files:
        df = pd.read_parquet(f)
        df = df[df["task"] == task]
        for item in df.to_dict("records"):
            out.append(f"{item.get('context','')}\n{item.get('question','')}\n"
                       f"{item.get('answer_prefix','')}")
            if len(out) >= n:
                return out
    return out

N = 200  # full cell. Validated against ground truth: this reproduces rwkv7's stamped
         # untruncated mean (16993.3) and llama's stamped input_tokens maxima
         # (16265/16323/15687) exactly.

# Oracle for the built-in self-check: rwkv7-world3 IS stamped, and its tokenizer.json is
# byte-identical (md5 2a91093dbd189729c63018aace31ef14) to the -fla-v2 directory the relay arms
# load, so its stamped numbers are ground truth for the relay arms' unstamped ones.
ORACLE = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/"
          "results/benchmarks/longctx_decodefix/rwkv7-world3__ruler.json")
ORACLE_CELL = "16384_vt"
ORACLE_TOKDIR = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/"
                 "models/RWKV7-Goose-World3-2.9B-HF-fla-v2")

tok_cache = {}
print(f"cap={CAP} effective_budget={EFFECTIVE} (reserved_for_generation is NOT subtracted; see "
      f"the NOTE above)\n")


def tokenizer_for(tokdir):
    if tokdir not in tok_cache:
        tok_cache[tokdir] = AutoTokenizer.from_pretrained(
            tokdir, trust_remote_code=True, local_files_only=True)
    return tok_cache[tokdir]


def lengths_for(tokdir, length_dir, task, n=N):
    tok = tokenizer_for(tokdir)
    return [len(tok(p, add_special_tokens=False)["input_ids"])
            for p in prompts_for(length_dir, task, n)], tok


# ---------------------------------------------------------------------------
# SELF-CHECK FIRST, and HARD-FAIL on mismatch.
#
# This block exists because the first committed version of this script raised NameError on line 62
# and could not produce any output at all, while the ledger cited numbers from a working copy I had
# run by hand. A peer caught it. The lesson is not "test your script" -- it is that a recovery tool
# whose output is being pasted into a paper must PROVE itself against a stamped oracle every time it
# runs, so that "the numbers in the ledger" and "what this file produces" cannot drift apart again.
# ---------------------------------------------------------------------------
def self_check():
    blob = json.load(open(ORACLE))
    rows = [r for r in blob["rows"] if r["dataset"] == ORACLE_CELL]
    stamped = [r["prompt_tokens_untruncated"] for r in rows
               if r.get("prompt_tokens_untruncated") is not None]
    if len(stamped) != N:
        raise SystemExit(f"SELF-CHECK CANNOT RUN: oracle {ORACLE} has {len(stamped)} stamped rows "
                         f"at {ORACLE_CELL}, expected {N}. Refusing to emit recovered numbers "
                         f"without a validated method.")
    recovered, _ = lengths_for(ORACLE_TOKDIR, ORACLE_CELL.split("_", 1)[0],
                               ORACLE_CELL.split("_", 1)[1])
    got = (round(sum(recovered) / len(recovered), 1), min(recovered), max(recovered))
    exp = (round(sum(stamped) / len(stamped), 1), min(stamped), max(stamped))
    print(f"SELF-CHECK vs stamped oracle ({ORACLE_CELL} of rwkv7-world3, byte-identical tokenizer):")
    print(f"  recovered (mean,min,max) = {got}")
    print(f"  stamped   (mean,min,max) = {exp}")
    if got != exp:
        raise SystemExit(
            "SELF-CHECK FAILED: recovered lengths do not match the stamped oracle. The prompt "
            "construction, tokenizer or parquet snapshot has changed. Every number below would be "
            "wrong -- nothing was printed. Fix the method, do not relax this check.")
    print("  MATCH on mean AND min AND max over 200 rows -> method validated.\n")


self_check()

for label, tokdir, cell, length_dir, task, poswin in CASES:
    lens, tok = lengths_for(tokdir, length_dir, task)
    over_cap  = sum(1 for L in lens if L > CAP)
    over_eff  = sum(1 for L in lens if L > EFFECTIVE)
    mean = sum(lens) / len(lens)
    reason = ("positional_window" if poswin is not None and poswin < CAP
              else "cli_cap_only")
    # seen_fraction as the runner computes it: a ratio of MEANS, not a mean of per-row ratios.
    seen = (CAP / mean) if over_cap == len(lens) else 1.0
    print(f"{label:20s} {cell:14s} n={len(lens)}  mean={mean:10.1f}  min={min(lens)}  max={max(lens)}")
    print(f"{'':20s} over {CAP}: {over_cap}/{len(lens)}   over effective {EFFECTIVE}: {over_eff}/{len(lens)}")
    print(f"{'':20s} seen_fraction = {seen:.4f}")
    print(f"{'':20s} declared positional_window={poswin!r}  =>  truncation_reason = {reason}")
    print(f"{'':20s} tokenizer = {tokdir.split('/')[-1]}  vocab={tok.vocab_size}")
    print()

#!/bin/bash
# OLMo-2-0425-1B baseline: the cells this model can actually be measured on.
#
# WHY THIS SCRIPT RUNS SO FEW LONG-CONTEXT CELLS
# ----------------------------------------------
# OLMo-2-0425-1B declares `max_position_embeddings: 4096`, and so do OLMo-2-1124-7B,
# -13B and -0325-32B (all four verified from their config.json on the Hub). The whole
# family is a 4k-context family; no size choice escapes it. `positional_window()` in
# run_open_context_bench.py therefore clamps the input budget to `4096 - max_new_tokens`.
#
# scripts/eval/audit_olmo2_window.py tokenised every planned cell with OLMo-2's OWN
# tokenizer over the same local dataset files the harness reads, before any GPU time was
# spent, and wrote results/benchmarks/longctx/olmo-2-1b__window_audit.json. Cells whose
# rows do not fit are NOT run here: a score earned on 2.5% of a document is not a
# capability measurement. The precedent is phi-2, whose RULER aggregate moved
# 0.0000 -> 0.4103 purely by clamping, with one cell going 0.000 -> 0.990 while the model
# saw 44% of the document. That number was published and is wrong.
#
# Cells RUN here (audit says every row fits whole, seen_fraction == 1.0):
#   babilong  qa1/4k, qa2/4k
#   ruler     4096/niah_single_1, 4096/niah_multikey_1, 4096/vt, 4096/qa_1
#
# ruler 4096/cwe is ALSO run, but is a special case that must be read from the per-row
# fields: 199 of its 200 rows fit whole and exactly 1 does not (mean 3778 vs budget 3968).
# Its aggregate `seen_fraction` rounds to 1.0000, which is precisely the kind of number that
# hides a truncated row, so the reportable value for this cell is the mean over the 199
# rows with `dropped_tokens == 0`, and the aggregate printed by the harness is NOT the number
# to quote. Running it and excluding one row recovers a real cell; dropping the cell would
# throw away 199 good rows.
#
# WHY cwe IS KEPT BUT fwe IS NOT, despite both being "partly truncated":
# the deciding issue is SELECTION BIAS, not the raw seen_fraction. Restricting a cell to its
# untruncated rows keeps only the SHORT documents, so the retained subset is systematically
# easier -- and for the RULER word-extraction tasks, fewer words to scan is strictly easier.
# cwe drops 1 row of 200 (0.5%), so that bias is negligible and the remaining 199 are still a
# fair sample. fwe drops 117 of 200 (58.5%) and its aggregate seen_fraction of 0.9889 looks
# almost clean while hiding exactly that: the surviving 83 rows are the short tail, so a number
# computed over them is NOT comparable to another model's full-200 fwe score and would flatter
# this model. Same reasoning rules out longbench/qasper (11 untruncated of 50),
# multifieldqa_en (8 of 50) and hotpotqa (2 of 50) -- all three have some rows that fit, but
# too few, and length-selected.
#
# Cells DELIBERATELY NOT RUN (audit seen_fraction < 1.0 -- reported as
# "not measurable within this model's 4096-token window", never as a score):
#   babilong  qa1/8k (0.549, 100/100 rows truncated), qa1/16k (0.269, 100/100)
#   ruler     4096/fwe (0.9889 but 117/200 truncated -- see the selection-bias note above),
#             16384/niah_single_1 (0.2531), 16384/vt (0.2432), 16384/cwe (0.2457), all 200/200
#   longbench narrativeqa (0.130, 50/50), qasper (0.626, 39/50),
#             multifieldqa_en (0.476, 42/50), hotpotqa (0.266, 48/50)
#   infinitebench  all five, every row truncated: longbook_qa_eng 0.0253,
#             longbook_choice_eng 0.0212, kv_retrieval 0.0320, code_debug 0.0232,
#             math_find 0.0441 -- i.e. this model would see ~2-4% of each document
#   memeval   longmemeval_oracle (0.564, 50/50) -- and separately, its 50-row slice has a
#             random floor of 0.220 with known scorer defects, so it is uninformative
#             regardless of window.
#
# The two suites whose cells all fit (short-context generate-then-match, and PPL) are run
# in full: they are where this baseline actually contributes a comparable number.
#
# BOS: probed empirically, not read off a flag. OLMo-2's tokenizer returns identical ids
# for add_special_tokens=True and False ([87] both ways for "x"), so tokenizer_wants_bos()
# is False and no BOS is prepended. This is the correct answer for this model, unlike
# gemma-2-2b (+40pp with BOS) -- the flag itself is zero information (7/7 models read False).
#
# DTYPE: the two paths below deliberately differ, and this matches every existing baseline.
# OLMo-2's config.json declares `torch_dtype: "float32"` (the dtype the weights actually ship
# in). run_open_context_bench.py hardcodes bfloat16 for --model_dir, while
# eval_lm_ppl.py / eval_generic_gen_match.py share a _load_hf_model that HONOURS the declared
# dtype (its in-code reason: "mamba-2.8b ships float32 weights, and loading them as bf16 makes
# the first forward die"). So the long-context cells run bf16 and the PPL/gen-match cells run
# fp32. That asymmetry is pre-existing, not introduced here: gemma-2-2b and mamba-2.8b also
# declare float32, so their published ppl16/ rows were produced through exactly this fp32 path
# while their longctx/ rows were bf16. Forcing either side to match the other would make this
# row incomparable with the nine baselines already in the tables. At 1B both fit one GPU
# (fp32 is ~5.9 GB of weights).
set -uo pipefail

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning
export HF_HOME=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# ${PYTHONPATH:+:$PYTHONPATH}, never a bare $PYTHONPATH: PYTHONPATH is UNSET in this image
# and under `set -u` a bare reference aborts the shell before any log redirect exists,
# which kills the job with no log at all.
export PYTHONPATH=/inspire/hdd/global_user/zhangjiaquan-253108540222/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MODEL=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf
TAG=olmo-2-1b
OUT=$REPO/results/benchmarks/longctx
GENOUT=$REPO/results/benchmarks/baseline-$TAG
PPLOUT=$REPO/results/benchmarks/ppl16
LOGD=$REPO/logs/qz
TASK_DATA=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/baseline/Cola-DLM/generate_task_data
mkdir -p "$OUT" "$GENOUT" "$PPLOUT" "$LOGD"
cd "$REPO"

BAB=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/datasets--RMT-team--babilong/snapshots/ee0d588794c7ac098062ee0d247c733d62e94fe2/data
RULER=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/datasets--simonjegou--ruler/snapshots/24adceac8a0e6532936e8d721cd9e9084d2e4686

echo "=== OLMo-2-1B eval, started $(date -u) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# ---- byte-level checkpoint gate, fatal --------------------------------------------------
# Runs BEFORE the load check, because "loads cleanly" does not imply "is the right bytes":
# a shard that is short by one chunk, or right-length-but-corrupt in the middle, can still
# produce a model object. This compares every file against the HF API's x-linked-size and
# both shards' sha256 against x-linked-etag (constants baked in, so it needs no network),
# and refuses to go further on any mismatch. ~15 s for 5.94 GB on this filesystem.
echo "=== checkpoint gate (sizes + sha256 + residue + index) ==="
python3 -u "$REPO/.omo/evidence/olmo-2-1b/verify_checkpoint.py" \
    > "$LOGD/${TAG}-gate.log" 2>&1
gate_rc=$?
tail -40 "$LOGD/${TAG}-gate.log"
if [ "$gate_rc" -ne 0 ]; then
  echo "FATAL checkpoint gate failed (rc=$gate_rc) -- refusing to score this checkpoint"
  exit 1
fi

# ---- load check, fatal ------------------------------------------------------------------
# A strict=False load silently swallowing 473 tensors already produced a wrong published
# result in this project, so the run aborts here rather than scoring a half-loaded model.
echo "=== load check ==="
python3 -u - <<PY > "$LOGD/${TAG}-loadcheck.log" 2>&1
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
p = "$MODEL"
cfg = AutoConfig.from_pretrained(p, local_files_only=True)
print("model_type", cfg.model_type, "window", cfg.max_position_embeddings)
tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
out = AutoModelForCausalLM.from_pretrained(
    p, torch_dtype=torch.bfloat16, local_files_only=True,
    output_loading_info=True)
model, info = out if isinstance(out, tuple) else (out, None)
if info is None:
    raise SystemExit("FATAL: no loading info returned; cannot prove key completeness")
# transformers 5.3.0 returns LoadStateDictInfo.to_dict(), whose missing_keys/unexpected_keys/
# mismatched_keys are SETS (only error_msgs is a list). Slicing a set raises TypeError, which
# would blow up this check inside the very failure path it exists to diagnose, so sort first.
def peek(v, k=20):
    try:
        return sorted(v)[:k] if not isinstance(v, dict) else list(v)[:k]
    except TypeError:
        return [repr(x) for x in list(v)[:k]]
nonempty = {k: peek(v) for k, v in info.items() if v}
print("loading_info_nonempty:", json.dumps(nonempty, default=str))
# All four categories are fatal, not just the two obvious ones: mismatched_keys means a tensor
# was silently NOT loaded because its shape disagreed, and error_msgs is where the loader puts
# failures it chose not to raise. A strict=False load silently swallowing 473 tensors already
# produced a wrong published result in this project.
for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
    got = info.get(key)
    if got:
        raise SystemExit(f"FATAL {key} (n={len(got)}): {peek(got)}")
print("loading_info_clean: missing/unexpected/mismatched/error_msgs all empty")
n = sum(p_.numel() for p_ in model.parameters())
print(f"params={n} ({n/1e9:.3f} B)")
# a real forward, so "loads" cannot mean "loads and emits garbage"
model = model.cuda().eval()
ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    g = model.generate(ids, max_new_tokens=8, do_sample=False)
print("smoke:", repr(tok.decode(g[0][ids.shape[1]:])))
print("LOADCHECK OK")
PY
tail -20 "$LOGD/${TAG}-loadcheck.log"
grep -q "LOADCHECK OK" "$LOGD/${TAG}-loadcheck.log" || { echo "FATAL load check failed"; exit 1; }

run_lc() {   # suite datasets max_samples max_new data_dir
  local suite=$1 ds=$2 n=$3 mnt=$4 dd=$5
  echo "[launch] $suite"
  python3 -u scripts/eval/run_open_context_bench.py \
      --suite "$suite" --datasets "$ds" --data_dir "$dd" \
      --model_dir "$MODEL" --arm raw --decode hf \
      --output "$OUT/${TAG}__${suite}.json" \
      --max_samples "$n" --max_input_tokens 4096 --max_new_tokens "$mnt" \
      --seed 42 --device cuda \
      > "$LOGD/${TAG}-${suite}.log" 2>&1
  echo "[done] $suite exit=$?"
}

# --max_input_tokens 4096 is the model's real window, not a guess: the harness still clamps
# to window - max_new_tokens internally and records the result per row. Passing 16384 here
# (as the published baselines did) would be harmless for the clamp but would make
# `config.max_input_tokens` misleading -- it is the raw CLI arg, not the effective budget.
run_lc babilong "qa1/4k,qa2/4k" 100 32 "$BAB"
run_lc ruler "4096/niah_single_1,4096/niah_multikey_1,4096/vt,4096/qa_1,4096/cwe" 200 128 "$RULER"

# ---- short context: generate-then-match, the tab:capability protocol --------------------
# 200 items/cell, 32 new tokens, repetition_penalty 1.1 (the script's default), temperature
# 0.0 -- copied from run_baseline_evals.py's own invocation for the HF baselines.
echo "=== short-context generate-then-match ==="
# PROVENANCE OF THIS ONE OUTPUT DIR (matters for which paper cell may cite it):
# The other nine baselines needed TWO directories, because six tasks (arc_easy, arc_challenge,
# winogrande, sciq, commonsenseqa, piqa) ship a `choices` field that the original prompts never
# showed the model; they were re-run into results/benchmarks/fix2choices/<slug>/ and
# results/paper_ledger/cell_sources.json points those six columns there and the other eight at
# results/benchmarks/baseline-<slug>/.
# `inline_choices()` now lives INSIDE eval_generic_gen_match.py's build_prompt(), so this fresh
# run emits the FIXED prompts for those six and leaves the other eight untouched. Verified by
# tokenising all 14 tasks through build_prompt() with OLMo-2's tokenizer: the eight
# non-choice tasks are byte-identical to the raw `prompt` field (mmlu 291/291, obqa 176/176,
# siqa 152/152, story_cloze 211/211, hellaswag 283/283, lambada 147/147, squad 528/528,
# race 854/854 max tokens) while all six choice tasks grow (piqa 27 -> 314, arc_easy 98 -> 146,
# arc_challenge 124 -> 157, winogrande 38 -> 46, sciq 69 -> 87, commonsenseqa 44 -> 67).
# So this SINGLE directory is protocol-equivalent to fix2choices/ for the six and to
# baseline-<slug>/ for the eight -- exactly the mix cell_sources.json cites. No second run is
# needed, and no cell here should be compared against a pre-fix number for those six columns.
# Every one of the 14 cells also fits the 4096 window with room to spare (max 854 tokens on
# race vs a 4064 budget), so unlike the long-context suites nothing is truncated here.
for T in mmlu obqa siqa story_cloze hellaswag piqa arc_easy arc_challenge winogrande sciq commonsenseqa lambada squad race; do
  echo "[gen] $T"
  python3 -u scripts/eval/eval_generic_gen_match.py \
      --model_dir "$MODEL" \
      --task_data_dir "$TASK_DATA" \
      --output_dir "$GENOUT" \
      --tasks "$T" \
      --max_samples 200 --max_new_tokens 32 --temperature 0.0 \
      > "$LOGD/${TAG}-gen-${T}.log" 2>&1
  echo "[gen] $T exit=$?"
done

echo "=== score generate-then-match ==="
# The scorer's own main() only prints 8 of the 14 tasks, so call score_file directly to get
# every cell. The default rule set (SCORE_LETTER_FIRST unset) is what every previously
# reported number used, so it stays unset -- turning it on would make this row incomparable
# with the other nine baselines.
python3 -u scripts/eval/score_gen_match.py "$GENOUT" \
    > "$LOGD/${TAG}-score.log" 2>&1
echo "[score/8task] exit=$?"
cat "$LOGD/${TAG}-score.log"

python3 -u - <<'PY' > "$LOGD/olmo-2-1b-score-all14.log" 2>&1
import json, os, sys
sys.path.insert(0, '/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning')
from scripts.eval.score_gen_match import score_file
base = ('/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning'
        '/results/benchmarks/baseline-olmo-2-1b')
tasks = ["mmlu", "obqa", "siqa", "story_cloze", "hellaswag", "piqa", "arc_easy",
         "arc_challenge", "winogrande", "sciq", "commonsenseqa", "lambada", "squad", "race"]
out = {}
for t in tasks:
    p = os.path.join(base, f"{t}.jsonl")
    if not os.path.exists(p):
        print(f"{t:16s} MISSING {p}"); continue
    # score_file returns (task, acc_or_f1, em, n) -- for free-form tasks (squad) the second
    # value is token-F1 and the third is EM; for choice tasks the second is accuracy and EM
    # is nan. Unpacking this in the wrong order would silently report n as the score.
    task, acc, em, n = score_file(p)
    out[t] = {"score": acc, "em": None if em != em else em, "n": n}
    print(f"{t:16s} n={n:4d} score={acc:.2f}" + ("" if em != em else f" em={em:.2f}"))
dest = os.path.join(base, "scores_all14.json")
json.dump(out, open(dest, "w"), indent=2)
print("[written]", dest)
PY
echo "[score/14task] exit=$?"
cat "$LOGD/olmo-2-1b-score-all14.log"

# ---- blank-generation exposure audit (does NOT change any score) -------------------------
# Another owner found, and I independently reproduced, that score_gen_match.letter_of() maps
# a BLANK generation to option A: `"" in "ABCDEFGH"` is substring containment, so it is True,
# the caller's `if lt is not None` guard passes on `""`, and `LETTERS.index("")` is 0. A file
# of 4 all-empty generations with gold=A on 2 rows scores 50.0 on the default path (verified
# directly). So a blank answer is silently credited whenever the gold happens to be A.
# The scorer is deliberately NOT patched here: it is shared, several agents are scoring against
# it concurrently, and a mid-flight fix would split the table into pre- and post-fix numbers --
# the exact comparability break this ledger exists to prevent. Instead this reports the
# exposure per cell so every number above is auditable, and whoever owns the scoring pass can
# apply the one-character fix (`len(t) == 1`) and rescore all variants together.
python3 -u - <<'PY' > "$LOGD/olmo-2-1b-blank-audit.log" 2>&1
import json, os, sys
sys.path.insert(0, '/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning')
from scripts.eval.score_gen_match import truncate, normalize_text, match_choice
base = ('/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning'
        '/results/benchmarks/baseline-olmo-2-1b')
tot_blank = tot_credited = tot_rows = 0
for t in ["mmlu", "obqa", "siqa", "story_cloze", "hellaswag", "piqa", "arc_easy",
          "arc_challenge", "winogrande", "sciq", "commonsenseqa", "lambada", "squad", "race"]:
    p = os.path.join(base, f"{t}.jsonl")
    if not os.path.exists(p):
        continue
    blank = credited = n = 0
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ch = r.get("choices") or []
        if not ch:
            continue
        n += 1
        if truncate(r.get("generate", "")).strip():
            continue
        blank += 1
        # would this blank row be credited? gold index resolved exactly as score_file does
        gt = str(r.get("ground_truth", ""))
        ng = normalize_text(gt)
        gold = next((i for i, c in enumerate(ch) if normalize_text(c) == ng), None)
        if gold is None:
            gold = match_choice(gt, ch)
        if gold == 0:
            credited += 1
    tot_blank += blank; tot_credited += credited; tot_rows += n
    flag = "  <-- INFLATES THIS CELL" if credited else ""
    print(f"{t:16s} choice_rows={n:4d} blank={blank:3d} falsely_credited={credited:3d}{flag}")
print(f"TOTAL choice_rows={tot_rows} blank={tot_blank} falsely_credited={tot_credited}")
print("Every falsely_credited row is worth +1/n on that cell; if 0, this defect did not touch "
      "this row at all and the scores above stand as printed.")
PY
echo "[blank-audit] exit=$?"
cat "$LOGD/olmo-2-1b-blank-audit.log"

# ---- PPL: the same 7-corpus protocol as results/benchmarks/ppl16/*.json ------------------
# NOTE for the reader, not a bug in this script: in every existing ppl16 file wikitext2 and
# wikitext103 are numerically identical, because both corpus dirs' first chunks are the same
# data and `--max_files 100` takes a prefix. That artefact is inherited here so the row stays
# comparable; it is not evidence about this model.
echo "=== PPL ==="
python3 -u scripts/eval/eval_lm_ppl.py --impl hf --model_dir "$MODEL" \
    --mode raw --corpora wikitext2,wikitext103,pg19,lm1b,ptb,wmt14,xsum \
    --output "$PPLOUT/${TAG}.json" \
    > "$LOGD/${TAG}-ppl.log" 2>&1
echo "[ppl] exit=$?"
tail -20 "$LOGD/${TAG}-ppl.log"

echo "=== summary ==="
python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob('/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/results/benchmarks/longctx/olmo-2-1b__*.json')):
    if 'window_audit' in f:
        continue
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f, 'UNREADABLE', e); continue
    c = d.get('config', {})
    print('==', os.path.basename(f))
    print('   window=%s budget=%s bos=%s decode=%s' % (
        c.get('positional_window'), c.get('effective_input_budget'),
        c.get('prepend_bos'), d.get('decode_path')))
    # Audit from the PER-ROW fields, never from config.max_input_tokens (which is the raw
    # CLI arg). A cell is only quotable if every row it kept was shown whole; where some
    # rows were truncated, print the score over the untruncated rows separately and say how
    # many were dropped, so a reader can never mistake a mixed cell for a clean one.
    rows = d.get('rows', [])
    # Aggregate SCHEMAS DIFFER BETWEEN FILE VINTAGES. The current runner writes a dict per
    # cell ({n, score, rows_truncated, seen_fraction, ...}), but older longctx files -- e.g.
    # every existing *__ruler.json, including qwen2.5-3b's and champ-dynamic-30k's -- store a
    # BARE FLOAT. Calling .get() on those raises AttributeError: 'float' object has no
    # attribute 'get'. This run writes the new form, so normalising here is belt-and-braces;
    # it matters because the same block gets pointed at the published files when comparing.
    def as_dict(v, cell_rows):
        if isinstance(v, dict):
            return v
        return {'n': len(cell_rows), 'score': float(v),
                'seen_fraction': None, 'rows_truncated': None}
    for k, v_raw in d.get('aggregates', {}).items():
        cell = [r for r in rows if r['dataset'] == k]
        v = as_dict(v_raw, cell)
        whole = [r for r in cell if not r.get('dropped_tokens')]
        clean = len(whole) == len(cell) and bool(cell)
        line = ('   %-24s n=%-4s score=%.4f seen=%s trunc=%s'
                % (k, v.get('n'), v.get('score', -1), v.get('seen_fraction'),
                   v.get('rows_truncated')))
        if clean:
            print(line + '   REPORTABLE')
        elif whole:
            sub = sum(r['score'] for r in whole) / len(whole)
            frac_kept = len(whole) / len(cell)
            # Restricting to untruncated rows keeps only the SHORT documents, so the subset is
            # length-selected and systematically easier. Tolerable when almost nothing is
            # dropped (cwe: 199/200); not tolerable once a large share is gone, because the
            # surviving rows are then the easy tail and the number is not comparable to another
            # model's full-cell score. 0.95 is the line, and the reason is printed either way
            # so no reader has to infer it.
            if frac_kept >= 0.95:
                print(line + '   REPORTABLE over the %d/%d untruncated rows -> quote %.4f'
                      ' (aggregate above is NOT the number to quote)'
                      % (len(whole), len(cell), sub))
            else:
                print(line + '   NOT COMPARABLE: only %d/%d rows (%.0f%%) fit, and those are'
                      ' the SHORT ones -- length-selected subset scores %.4f, do not quote'
                      % (len(whole), len(cell), 100 * frac_kept, sub))
        else:
            print(line + '   NOT MEASURABLE (no row seen whole)')
    # Trap: a zero can mean "model cannot do the task" or "model emitted nothing", and a
    # degeneracy detector's verdict swings ~55x with its threshold, so print RAW counts and
    # let a human read samples instead of trusting a heuristic. Note RULER niah gold IS a bare
    # number, so "output is one short token" is a CORRECT answer there, not degeneracy --
    # which is exactly why this reports counts and examples rather than a degenerate flag.
    for k in d.get('aggregates', {}):
        cell = [r for r in rows if r['dataset'] == k]
        if not cell:
            continue
        empty = sum(1 for r in cell if not str(r.get('prediction', '')).strip())
        ws_only = sum(1 for r in cell if str(r.get('prediction', '')).strip('\n\r\t ') == ''
                      and str(r.get('prediction', '')) != '')
        zero = sum(1 for r in cell if not r.get('score'))
        print('      %-20s empty=%d whitespace_only=%d zero_score=%d/%d'
              % (k, empty, ws_only, zero, len(cell)))
        for r in cell[:2]:
            print('        sample pred=%r gold=%r score=%s'
                  % (str(r.get('prediction', ''))[:90], r.get('answers'), r.get('score')))
p = '/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/results/benchmarks/ppl16/olmo-2-1b.json'
if os.path.exists(p):
    print('== ppl16/olmo-2-1b.json'); print('  ', json.dumps(json.load(open(p)).get('perplexity')))
PY
echo "=== ALL DONE $(date -u) ==="

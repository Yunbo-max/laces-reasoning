#!/usr/bin/env bash
# Gemma-2-2B, the 14 capability tasks of tab:capability, WITH the leading BOS restored.
#
# WHY. eval_generic_gen_match.py tokenizes with add_special_tokens=False (needed so that
# middle-truncation offsets stay inside the document). For Gemma that strips the BOS the model
# expects. This paper's own LongBench appendix measures that omission as worth +8.00 points to
# Gemma alone -- model-specific, not extrapolable -- so the published Gemma capability row
# (27.3 avg) is a floor set by our harness, not a capability measurement. The generations show
# it: sciq row 0 is 'Scopes! Scopes! Scopes! ...'.
#
# The confound points OUR way: it depresses a baseline we compare ourselves against. That is
# why this re-run happens before the row is used to support any claim.
#
# WHAT IS HELD FIXED. Everything except the BOS: same task files, same --max_samples 200,
# --max_new_tokens 32, --temperature 0.0 (greedy), --repetition_penalty 1.1 default, same
# seed 42, same weights. --restore_bos is a new flag that defaults OFF, so no previously
# published row changes; the flag is recorded in run_summary.json for provenance.
#
# The 14 tasks are exactly the columns of tab:capability, in table order. RACE is included
# even though the table's RACE column is being dropped for a separate decode-path defect:
# the cell costs two minutes and its absence would be harder to explain later than its
# presence.
set -euo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222
export ROOT
REPO=$ROOT/research/laces-reasoning
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

TASKS=mmlu,obqa,siqa,story_cloze,hellaswag,piqa,arc_easy,arc_challenge,winogrande,sciq,commonsenseqa,lambada,race,squad
DATA=$ROOT/research/DiffRwkv/baseline/Cola-DLM/generate_task_data
MODEL=$ROOT/models/gemma-2-2b
OUT=$REPO/results/benchmarks/bosfix/baseline-gemma-2-2b-bos
LOGS=$REPO/logs/qz
mkdir -p "$OUT" "$LOGS"
cd "$REPO"

python3 scripts/eval/eval_generic_gen_match.py \
  --model_dir "$MODEL" \
  --task_data_dir "$DATA" \
  --output_dir "$OUT" \
  --tasks "$TASKS" \
  --max_samples 200 \
  --max_new_tokens 32 \
  --temperature 0.0 \
  --seed 42 \
  --restore_bos \
  > "$LOGS/gemma-capability-bos.log" 2>&1

# Do NOT echo "exit=$?" here: under `set -e` this line is only reached on success, so the
# stamp would always read 0. Job status is the authority.
echo "gemma BOS capability run completed -> $OUT"

python3 - <<'EOF'
import json, os, sys, statistics
sys.path.insert(0, 'scripts/eval')
import score_gen_match as S

ROOT = os.environ['ROOT'] if 'ROOT' in os.environ else '/inspire/hdd/global_user/zhangjiaquan-253108540222'
R = f'{ROOT}/research/laces-reasoning/results/benchmarks'
NEW = f'{R}/bosfix/baseline-gemma-2-2b-bos'
# The published row is NOT one directory. Verified by reproducing it: the six option-inlined
# tasks (piqa arc_easy arc_challenge winogrande sciq commonsenseqa) come from fix2choices/,
# the other eight from baseline-gemma-2-2b/, and the mean over the 11 tasks the table's Avg
# keeps (it drops hellaswag, arc_challenge, squad) is 27.27 == the printed 27.3. Resolving
# fix2choices first per task is what reproduces it; pointing OLD at either dir alone does not.
OLD_DIRS = [f'{R}/fix2choices/baseline-gemma-2-2b', f'{R}/baseline-gemma-2-2b']
def old_path(t):
    for d in OLD_DIRS:
        p = f'{d}/{t}.jsonl'
        if os.path.exists(p):
            return p
    return None
TASKS = "mmlu obqa siqa story_cloze hellaswag piqa arc_easy arc_challenge winogrande sciq commonsenseqa lambada race squad".split()
AVG_TASKS = [t for t in TASKS if t not in ('hellaswag', 'arc_challenge', 'squad')]

s = json.load(open(f'{NEW}/run_summary.json'))
print('PROVENANCE restore_bos =', s.get('restore_bos'), ' rep_pen =', s.get('repetition_penalty'))
assert s.get('restore_bos') is True, 'restore_bos not recorded -- the flag did not reach the run'

# The control that proves the intervention landed: the prompt must be exactly one token longer.
for t in ('sciq', 'piqa'):
    a = [json.loads(l) for l in open(old_path(t))]
    b = [json.loads(l) for l in open(f'{NEW}/{t}.jsonl')]
    m = {r['id']: r for r in a}
    d = [b_r['prompt_tokens'] - m[b_r['id']]['prompt_tokens'] for b_r in b if b_r['id'] in m]
    ch = sum(1 for b_r in b if b_r['id'] in m and b_r.get('generate') != m[b_r['id']].get('generate'))
    print(f'  LANDED {t}: prompt_tokens delta set={sorted(set(d))} (want [1]); '
          f'predictions changed {ch}/{len(d)}')

print(f"\n{'task':16s} {'old':>7s} {'BOS':>7s} {'delta':>7s}  {'in Avg':>6s}")
o_all, n_all = {}, {}
for t in TASKS:
    po, pn = old_path(t), f'{NEW}/{t}.jsonl'
    if not os.path.exists(pn):
        continue
    _, an, _, nn = S.score_file(pn)
    vn = 100 * an if an <= 1 else an
    tag = 'yes' if t in AVG_TASKS else '--'
    if po:
        _, ao, _, no = S.score_file(po)
        vo = 100 * ao if ao <= 1 else ao
        o_all[t] = vo; n_all[t] = vn
        print(f'{t:16s} {vo:7.2f} {vn:7.2f} {vn - vo:+7.2f}  {tag:>6s}')
    else:
        print(f'{t:16s} {"--":>7s} {vn:7.2f} {"n/a":>7s}  {tag:>6s}   (no old cell)')

# Two means, both printed: the table's own Avg definition, and all 14, so that a change in
# the row cannot be confused with a change in which columns the row averages over.
for label, sel in (('AVG(11, as printed)', AVG_TASKS), ('MEAN(all 14)', TASKS)):
    ks = [t for t in sel if t in o_all]
    if not ks:
        continue
    mo, mn = statistics.mean(o_all[t] for t in ks), statistics.mean(n_all[t] for t in ks)
    print(f'{label:20s} {mo:7.2f} {mn:7.2f} {mn - mo:+7.2f}   (n_tasks={len(ks)})')
EOF

#!/bin/bash
# Alignment experiment driver: 4 decode/dtype arms in parallel on 8 GPUs,
# plus the weight bit-identity check. One arm per GPU.
set -uo pipefail

export HF_HOME=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/inspire/hdd/global_user/zhangjiaquan-253108540222/shared_pylibs${PYTHONPATH:+:$PYTHONPATH}

REPO=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning
LOGDIR=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/logs/qz
S=$REPO/scripts/eval/align_rwkv_two_harnesses.py
OUT=$REPO/results/benchmarks/harness_align

mkdir -p "$LOGDIR" "$OUT"
cd "$REPO" || exit 1

echo "=== START $(date -Is) host=$(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# (1) Is the relay checkpoint's "frozen decoder" bit-identical to the released HF
#     weights?  Runs on GPU 7 while the arms occupy 0-3.
CUDA_VISIBLE_DEVICES=7 python "$S" --mode verify_weights \
  > "$LOGDIR/align-verify-weights.log" 2>&1 &
PID_V=$!

# (2) Per-step top-5 trace, fp32 vs bf16, one prompt.
CUDA_VISIBLE_DEVICES=6 python "$S" --mode trace \
  > "$LOGDIR/align-trace.log" 2>&1 &
PID_T=$!

# (3) The 2x2: {hf, loop} x {fp32, bf16}, same weight file, 6 tasks x 200.
i=0
PIDS=()
for ARM in hf_fp32 hf_bf16 loop_fp32 loop_bf16; do
  CUDA_VISIBLE_DEVICES=$i python "$S" --mode run --arm "$ARM" \
    --output_dir "$OUT" --max_samples 200 --max_new_tokens 32 \
    --repetition_penalty 1.1 \
    > "$LOGDIR/align-arm-$ARM.log" 2>&1 &
  PIDS+=($!)
  echo "launched arm=$ARM gpu=$i pid=${PIDS[-1]}"
  i=$((i+1))
done

RC=0
wait $PID_V || { echo "verify_weights FAILED rc=$?"; RC=1; }
wait $PID_T || { echo "trace FAILED rc=$?"; RC=1; }
for idx in "${!PIDS[@]}"; do
  wait "${PIDS[$idx]}" || { echo "arm[$idx] FAILED rc=$?"; RC=1; }
done

echo "=== ARMS DONE rc=$RC $(date -Is) ==="

# (4) Score every arm on the six option-bearing columns, both scoring rules,
#     and report token-level agreement between arms.
python - <<'PY'
import json, os, sys
REPO='/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning'
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,'scripts','eval'))
T=["piqa","arc_easy","arc_challenge","winogrande","sciq","commonsenseqa"]
ARMS=["hf_fp32","hf_bf16","loop_fp32","loop_bf16"]
ROOT=os.path.join(REPO,'results','benchmarks','harness_align')
out={}
for mode in ('0','1'):
    os.environ['SCORE_LETTER_FIRST']=mode
    for m in [k for k in list(sys.modules) if 'score_gen_match' in k]: del sys.modules[m]
    from scripts.eval.score_gen_match import score_file
    print(f"===== SCORE_LETTER_FIRST={mode}")
    print(f"{'arm':12s} " + " ".join(f"{t[:9]:>9s}" for t in T) + "   MEAN6")
    for a in ARMS:
        vals=[]
        for t in T:
            p=os.path.join(ROOT,a,f'{t}.jsonl')
            if not (os.path.exists(p) and os.path.getsize(p)>0): vals.append(float('nan')); continue
            _,acc,_,_=score_file(p); vals.append(acc)
        ok=[v for v in vals if v==v]
        mean=sum(ok)/len(ok) if ok else float('nan')
        out[(mode,a)]=(vals,mean)
        print(f"{a:12s} " + " ".join(f"{v:9.1f}" for v in vals) + f" {mean:7.2f}")
    print()

print("===== token-level agreement between arms (gen_ids), per task")
def load(a,t):
    p=os.path.join(ROOT,a,f'{t}.jsonl')
    if not (os.path.exists(p) and os.path.getsize(p)>0): return None
    return [json.loads(l) for l in open(p) if l.strip()]
import itertools
for x,y in itertools.combinations(ARMS,2):
    tot=same=0
    for t in T:
        A,B=load(x,t),load(y,t)
        if not A or not B: continue
        for ra,rb in zip(A,B):
            tot+=1; same += (ra.get('gen_ids')==rb.get('gen_ids'))
    if tot: print(f"  {x:10s} vs {y:10s}  identical_gen_ids={same}/{tot} ({100.0*same/tot:.1f}%)")

print("===== non-empty file census")
for a in ARMS:
    ne=[t for t in T if os.path.exists(os.path.join(ROOT,a,f'{t}.jsonl')) and os.path.getsize(os.path.join(ROOT,a,f'{t}.jsonl'))>0]
    print(f"  {a:12s} non_empty={len(ne)}/6 {ne}")
PY

echo "=== ALL DONE $(date -Is) rc=$RC ==="

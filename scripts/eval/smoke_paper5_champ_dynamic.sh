#!/usr/bin/env bash
# Smoke test: champ-dynamic relay checkpoint on the four MC paper benchmarks,
# both sampling modes, tiny sample counts. Run from the laces-reasoning repo root
# (load_relay_model resolves the checkpoint's relative rwkv_local_path against CWD).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000}"
DATA_ROOT="${DATA_ROOT:-/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data}"
OUT_ROOT="${OUT_ROOT:-results/benchmarks/paper5_smoke}"
STEPS="${STEPS:-100}"
N_SMOKE="${N_SMOKE:-20}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -d "$CKPT" ]]; then
  echo "FATAL: checkpoint dir not found: $CKPT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# run_one <gpu> <mode:traj|singlez> <task_name> <task_dir> <n>
run_one() {
  local gpu="$1" mode="$2" name="$3" tdir="$4" n="$5"
  local out="$OUT_ROOT/${name}_${mode}.json"
  local log="$OUT_ROOT/${name}_${mode}.log"
  local script
  if [[ "$mode" == "traj" ]]; then
    script="scripts/eval/eval_mcpack_trajectory_cfg.py"
  else
    script="scripts/eval/eval_prefix_suffix_cfg_multichoice.py"
  fi
  echo "=== [gpu $gpu] $name / $mode (n=$n) -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$script" \
    --ckpt_dir "$CKPT" \
    --task_dir "$tdir" \
    --max_samples "$n" \
    --steps "$STEPS" \
    --scales 1 \
    --device cuda \
    --output "$out" >"$log" 2>&1
  echo "--- $name/$mode exit=$? ; tail:"
  tail -n 4 "$log"
}

# Spread the five checks over four GPUs; each falls back to gpu0 if fewer are visible.
run_one 0 traj    gpqa_diamond "$DATA_ROOT/gpqa_diamond/test" "$N_SMOKE" &
run_one 1 singlez gpqa_diamond "$DATA_ROOT/gpqa_diamond/test" "$N_SMOKE" &
run_one 2 traj    mmlu_pro     "$DATA_ROOT/mmlu_pro/test"     10 &
run_one 3 singlez ceval        "$DATA_ROOT/ceval/val"         "$N_SMOKE" &
wait

run_one 0 traj mmlu_redux "$DATA_ROOT/mmlu_redux/test" "$N_SMOKE"

echo "=== SMOKE SUMMARY ==="
python - "$OUT_ROOT" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
for p in sorted(root.glob("*.json")):
    d = json.loads(p.read_text())
    keys = {k: v for k, v in d.items() if k.startswith("acc_") or k.startswith("elapsed_")}
    print(f"{p.stem:28s} n={d.get('n_files')} {keys}")
PY

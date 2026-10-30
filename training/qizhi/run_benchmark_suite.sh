#!/usr/bin/env bash
set -euo pipefail

# --- Runner for the Qizhi Benchmark Suite ---
# This script iterates through specified models, prepares a single Qizhi job
# that runs all specified benchmarks sequentially, and submits it.
#
# Example (Dry Run for one model):
#   bash training/qizhi/run_benchmark_suite.sh --model-type dynamic-basis --dry-run
#
# Example (Submit a specific job):
#   bash training/qizhi/run_benchmark_suite.sh --model-type fixed-basis
#

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PREPARE_SCRIPT="${REPO_ROOT}/training/qizhi/prepare_benchmark_suite.py"

# Models and runnable benchmark bundles to test.
ALL_MODELS=("dynamic-basis" "fixed-basis" "raw-rwkv")
ALL_BENCHMARKS=("gsm8k" "cola8" "mcpack")

# --- Argument Parsing ---
DRY_RUN=0
TARGET_MODELS=()

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dry-run)
        DRY_RUN=1
        shift
        ;;
        --model-type)
        TARGET_MODELS+=("$2")
        shift; shift
        ;;
        *)
        echo "Unknown option: $1" >&2
        exit 1
        ;;
    esac
done

# If no specific targets are given, run all models
if [ ${#TARGET_MODELS[@]} -eq 0 ]; then
    TARGET_MODELS=("${ALL_MODELS[@]}")
fi

# --- Main Loop ---
echo "Starting Qizhi benchmark suite submission..."

for model in "${TARGET_MODELS[@]}"; do
    echo "--------------------------------------------------"
    echo "Preparing job for Model=${model} with all benchmarks"
    echo "--------------------------------------------------"

    # The python script will now generate a single command for all benchmarks
    if [ $DRY_RUN -eq 1 ]; then
        python3 "${PREPARE_SCRIPT}" --model-type "${model}" --benchmark "${ALL_BENCHMARKS[@]}" --dry-run
    else
        echo "Submitting job for ${model}..."
        # Generate JSON for the combined job and pipe to qz
        python3 "${PREPARE_SCRIPT}" --model-type "${model}" --benchmark "${ALL_BENCHMARKS[@]}" | tr -d '\n' | xargs -0 qz train CreateJob --data
        echo "Job submitted for ${model} to run all benchmarks."
    fi
    echo ""
done

echo "All specified model benchmark jobs have been processed."

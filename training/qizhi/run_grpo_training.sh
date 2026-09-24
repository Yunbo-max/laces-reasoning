#!/usr/bin/env bash
# Runner for LACES GRPO post-training on a specified dataset.

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PREPARE_SCRIPT="${REPO_ROOT}/training/qizhi/prepare_grpo_training.py"

# --- Argument Parsing ---
DRY_RUN=0
TARGET_DATASET=""

# Simple loop for arguments. For more complex needs, use getopts.
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dry-run)
        DRY_RUN=1
        shift
        ;;
        --dataset)
        TARGET_DATASET="$2"
        shift; shift
        ;;
        *)
        echo "Usage: $0 [--dry-run] --dataset <dataset_name>" >&2
        echo "Available datasets: deepmath, gsm8k, math" >&2
        exit 1
        ;;
    esac
done

if [ -z "${TARGET_DATASET}" ]; then
    echo "Error: --dataset is a required argument." >&2
    echo "Usage: $0 [--dry-run] --dataset <dataset_name>" >&2
    exit 1
fi

# --- Main Execution ---
echo "--------------------------------------------------"
echo "Starting LACES GRPO Training Job Preparation"
echo "Dataset: ${TARGET_DATASET}"
if [ $DRY_RUN -eq 1 ]; then
    echo "Mode: Dry Run"
fi
echo "--------------------------------------------------"

CMD_ARGS=("--dataset" "${TARGET_DATASET}")
if [ ${DRY_RUN} -eq 1 ]; then
    CMD_ARGS+=("--dry-run")
fi

# Generate the qz command or JSON body
if [ $DRY_RUN -eq 1 ]; then
    python3 "${PREPARE_SCRIPT}" "${CMD_ARGS[@]}"
else
    echo "Generating job configuration and submitting to Qizhi..."
    # The python script prints the JSON, xargs passes it to --data
    # Note: Using xargs -0 is safer for complex JSON strings.
    python3 "${PREPARE_SCRIPT}" "${CMD_ARGS[@]}" | tr -d '\n' | xargs -0 qz train CreateJob --data
    echo "Job submitted for dataset ${TARGET_DATASET}."
fi

echo "GRPO training job for ${TARGET_DATASET} has been processed."

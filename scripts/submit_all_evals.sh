#!/usr/bin/env bash

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREPARE_SCRIPT="${REPO_ROOT}/training/qizhi/prepare_benchmark_suite.py"

MODELS_TO_RUN=("raw-rwkv" "fixed-basis" "dynamic-basis")

# --- Main Loop ---
echo "##################################################"
echo "#    Submitting Full LACES Benchmark Suite     #"
echo "##################################################"

for model in "${MODELS_TO_RUN[@]}"; do
    echo "
---> Preparing and submitting job for Model: ${model}"
    
    job_json=$(python3 "${PREPARE_SCRIPT}" --model-type "${model}" --benchmark gsm8k cola8 mcpack)

    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate job JSON for model ${model}. Aborting." >&2
        exit 1
    fi

    echo "Submitting job to Qizhi..."
    qz train CreateJob --data "${job_json}"

    if [ $? -ne 0 ]; then
        echo "Error: Qizhi job submission failed for model ${model}." >&2
        # Optional: decide whether to continue with other models or exit
        # exit 1 
    else
        echo "Successfully submitted job for ${model}."
    fi
    echo "--------------------------------------------------"
    sleep 2 # Small delay to avoid overwhelming the API
done

echo "
All evaluation jobs submitted."

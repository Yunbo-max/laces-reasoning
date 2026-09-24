#!/usr/bin/env bash
# Prepares the reasoning-aware dataset mixture for LACES pretraining.
# This script downloads, preprocesses, and shards multiple datasets
# into the format expected by `train_state_hijacking_dit.py`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/training/common.sh"

# --- Configuration ---
: "${CONTEXT_LENGTH:=512}"
: "${BACKBONE:=2.9B}" # For tokenizer selection
: "${OUT_DIR:=preprocessed_data/laces_reasoning_pretrain_p0}"
: "${FORCE_REPROCESS:=0}"

# Dataset sources from Hugging Face and their respective max samples for the mix.
# Using placeholder names and splits for now.
# These should be adjusted based on the final dataset selection and splits.
DATASETS=(
    # prefix, dataset_name, max_samples, extra_args
    "fineweb_edu,HuggingFaceFW/fineweb-edu,1000000, --split train"
    "openwebmath,openwebmath/openwebmath,500000,"
    "proofpile2,EleutherAI/proof-pile-2,300000, --split train"
    "starcoder_edu,bigcode/the-stack-edu,300000,"
    "openthoughts,open-thoughts/OpenThoughts3-1.2M,300000,"
    "science_texts,EleutherAI/proof-pile-2,200000, --split arxiv"
)

# --- Main Preprocessing Logic ---
CHUNK="$(trajectory_chunk_for "${CONTEXT_LENGTH}")"
MODEL_PATH="${MODEL_PATH:-$(realpath "${REPO_ROOT}/models/RWKV7-Goose-World3-${BACKBONE}-HF-fla-v2")}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Error: RWKV backbone not found at ${MODEL_PATH}" >&2
    echo "Please ensure the base model is downloaded and available." >&2
    exit 1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUT_DIR}"
fi

preprocess_dataset() {
  local prefix="$1"
  local dataset="$2"
  local max_samples="$3"
  shift 3
  local manifest_path="${OUT_DIR}/manifest_${prefix}.json"

  if [[ "${FORCE_REPROCESS}" != "1" && -f "${manifest_path}" ]]; then
    echo "Skipping ${prefix}: found ${manifest_path}. Set FORCE_REPROCESS=1 to regenerate." >&2
    return 0
  fi

  local args=(
    scripts/preprocess/preprocess_response_sft.py
    --dataset "${dataset}"
    --output_dir "${OUT_DIR}"
    --model_path "${MODEL_PATH}"
    --max_length "${CONTEXT_LENGTH}"
    --min_response_tokens "${CHUNK}" # Use chunk size as min length for consistency
    --filename_prefix "${prefix}"
    --local_files_only # Assume datasets are cached locally
    --streaming
    "$@"
  )

  if [[ "${max_samples}" != "0" ]]; then
    args+=(--max_samples "${max_samples}")
  fi

  run_cmd python3 "${args[@]}"
}

print_header "LACES Reasoning Pre-training Dataset Preparation"

for d in "${DATASETS[@]}"; do
    IFS=',' read -r prefix dataset_name max_samples extra_args <<< "$d"
    # shellcheck disable=SC2086
    preprocess_dataset "${prefix}" "${dataset_name}" "${max_samples}" ${extra_args}
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Data preparation dry-run complete for mix: ${OUT_DIR}"
else
  echo "Data preparation complete for mix: ${OUT_DIR}"
  # TODO: Add a manifest merge step here to create a final mix manifest.
fi

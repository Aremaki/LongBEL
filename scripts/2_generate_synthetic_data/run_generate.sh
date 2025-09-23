#!/bin/bash
# Launcher to submit one sbatch per chunk across all datasets/variants.
# Usage:
#   ./scripts/2_generate_synthetic_data/run_genereate.sh
# Optional env overrides (propagated to sbatch jobs):
#   MODEL_PATH=... BATCH_SIZE=4 MAX_NEW_TOKENS=1024 MAX_RETRIES=5 ./scripts/2_generate_synthetic_data/run_genereate.sh

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SLURM_SCRIPT="${SCRIPT_DIR}/generate_slurm.sh"

# Ensure slurm script exists
if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "Missing ${SLURM_SCRIPT}"
  exit 1
fi

# Helper to submit all chunks in a directory
submit_for_dir() {
  local dataset="$1"     # MM | QUAERO
  local variant="$2"     # def | no_def
  local user_prompts_dir="$3"

  echo "[SUBMIT] dataset=${dataset} variant=${variant} dir=${user_prompts_dir}"
  for chunk_file in "${user_prompts_dir}"/sample_*.parquet; do
    local chunk
    chunk=$(basename "$chunk_file" .parquet | sed 's/sample_//')
    echo "  -> sbatch ${dataset} ${variant} chunk=${chunk}"
    sbatch --export=ALL,MODEL_PATH,BATCH_SIZE,MAX_NEW_TOKENS,MAX_RETRIES \
      "${SLURM_SCRIPT}" "${dataset}" "${variant}" "${chunk}"
  done
}

# MM
submit_for_dir MM def    "data/synthetic_data/SynthMM/user_prompts_def"
submit_for_dir MM no_def "data/synthetic_data/SynthMM/user_prompts_no_def"

# QUAERO
submit_for_dir QUAERO def    "data/synthetic_data/SynthQUAERO/user_prompts_def"
submit_for_dir QUAERO no_def "data/synthetic_data/SynthQUAERO/user_prompts_no_def"

echo "All jobs submitted."

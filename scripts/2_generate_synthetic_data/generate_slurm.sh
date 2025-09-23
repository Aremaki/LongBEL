#!/bin/bash
#SBATCH --job-name=generate_synth        # Job name
#SBATCH -C h100                           # Target H100 nodes
#SBATCH --ntasks=1                        # Total tasks (GPUs)
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH --gres=gpu:1                      # GPUs per node
#SBATCH --cpus-per-task=24                # CPU cores per task
#SBATCH --hint=nomultithread              # Disable SMT
#SBATCH --time=20:00:00                   # Max wall time (HH:MM:SS)
#SBATCH --output=logs/log_out%j.out       # STDOUT log
#SBATCH --error=logs/log_err%j.out        # STDERR log

# Fail fast and be verbose about failures
set -euo pipefail

# Args: DATASET (MM|QUAERO), VARIANT (def|no_def), CHUNK (number)
DATASET=${1:-}
VARIANT=${2:-}
CHUNK=${3:-}

if [[ -z "${DATASET}" || -z "${VARIANT}" || -z "${CHUNK}" ]]; then
  echo "Usage: sbatch $(basename "$0") <MM|QUAERO> <def|no_def> <chunk>"
  exit 1
fi

if [[ "${DATASET}" != "MM" && "${DATASET}" != "QUAERO" ]]; then
  echo "DATASET must be MM or QUAERO"
  exit 1
fi

if [[ "${VARIANT}" != "def" && "${VARIANT}" != "no_def" ]]; then
  echo "VARIANT must be def or no_def"
  exit 1
fi

# Environment setup (adapt to your cluster modules)
module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1

# Common args (allow env overrides)
MODEL_PATH=${MODEL_PATH:-models/Llama-3.3-70B-Instruct}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_RETRIES=${MAX_RETRIES:-5}

if [[ "${DATASET}" == "MM" ]]; then
  BASE_DIR="data/synthetic_data/SynthMM"
  DATASET_PREFIX="SynthMM"
  SYSTEM_PROMPT_PATH="scripts/2_generate_synthetic_data/prompts/system_prompt_mm.txt"
else
  BASE_DIR="data/synthetic_data/SynthQUAERO"
  DATASET_PREFIX="SynthQUAERO"
  SYSTEM_PROMPT_PATH="scripts/2_generate_synthetic_data/prompts/system_prompt_quaero_fr.txt"
fi

USER_PROMPTS_DIR="${BASE_DIR}/user_prompts_${VARIANT}"
OUT_DIR="${BASE_DIR}/generated_${VARIANT}"

if [[ ! -f "${USER_PROMPTS_DIR}/sample_${CHUNK}.parquet" ]]; then
  echo "Missing input chunk: ${USER_PROMPTS_DIR}/sample_${CHUNK}.parquet"
  exit 2
fi

mkdir -p "${OUT_DIR}"

echo "[INFO] Launching generation for ${DATASET} ${VARIANT} chunk=${CHUNK}"
python scripts/2_generate_synthetic_data/generate.py run \
  --chunk "${CHUNK}" \
  --user-prompts-dir "${USER_PROMPTS_DIR}" \
  --out-dir "${OUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --system-prompt-path "${SYSTEM_PROMPT_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-retries "${MAX_RETRIES}"

echo "SCRIPT FINISHED: ${DATASET} ${VARIANT} chunk=${CHUNK}"

#!/bin/bash
set -e

# Change to the script's directory so relative paths work properly
cd "$(dirname "$0")"

BASE_MODEL="${1:-../../models/deberta-v3-large}"

DATASETS=(
  "medmentions"
  "quaero_emea"
  "quaero_medline"
  "spaccc"
)

echo "Submitting NER training jobs for datasets with base model: $BASE_MODEL"

for DATASET in "${DATASETS[@]}"; do
  echo "Submitting urgent 2h job for: $DATASET"

  sbatch \
    --job-name="ner_${DATASET}" \
    -A ssq@h100 \
    --qos=qos_gpu_h100-dev \
    --time=02:00:00 \
    run_ner_experiments.slurm "$DATASET" "$BASE_MODEL"
done

echo "All jobs submitted!"

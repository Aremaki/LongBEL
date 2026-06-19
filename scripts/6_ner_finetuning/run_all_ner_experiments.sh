#!/bin/bash
set -e

# Change to the script's directory so relative paths work properly
cd "$(dirname "$0")"

DATASETS=(
  "MedMentions"
  "EMEA"
  "MEDLINE"
  "SPACCC"
)

echo "Submitting NER training jobs for datasets: ${DATASETS[*]}"

for DATASET in "${DATASETS[@]}"; do
  echo "Submitting job for: $DATASET"

  sbatch \
    --job-name="ner_${DATASET}" \
    -A ssq@h100 \
    run_ner_experiments.slurm "$DATASET"
done

echo "All jobs submitted!"

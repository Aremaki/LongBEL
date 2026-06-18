#!/usr/bin/env bash
set -euo pipefail

# Run all NER experiments for one base model.
#
# Usage:
#   bash scripts/run_ner_experiments.sh
#   bash scripts/run_ner_experiments.sh ../../models/word-embedding/deberta-v3-large
#   bash scripts/run_ner_experiments.sh microsoft/deberta-v3-large

CONFIG="configs/ner/train_ner.cfg"
BASE_MODEL="${1:-../../models/word-embedding/deberta-v3-large}"

DATASETS=(
  "medmentions"
  "quaero_emea"
  "quaero_medline"
  "spaccc"
)

for DATASET in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "Training dataset=${DATASET} base_model=${BASE_MODEL}"
  echo "============================================================"

  python train_ner_edsnlp.py train "${CONFIG}" \
    --vars.dataset "${DATASET}" \
    --vars.base_model "${BASE_MODEL}"

  echo "============================================================"
  echo "Evaluating dataset=${DATASET} base_model=${BASE_MODEL}"
  echo "============================================================"

  python train_ner_edsnlp.py evaluate "${CONFIG}" \
    --vars.dataset "${DATASET}" \
    --vars.base_model "${BASE_MODEL}"
done

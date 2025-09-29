#!/bin/bash

# This script runs the evaluation for the SynCABEL model with a comprehensive set of parameters.
# It iterates through different datasets, model configurations, and generation strategies
# to produce a detailed evaluation report.

python scripts/6_evaluate_syncabel/evaluate.py \
    --datasets MedMentions MEDLINE EMEA \
    --model_names "mt5-large" \
    --selection_methods "embedding" "levenshtein" "tfidf" "title" \
    --num_beams 5 \
    --with_group True False \
    --best True False \
    --augmented_data True False \
    --constraints True False

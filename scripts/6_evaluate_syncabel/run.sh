#!/bin/bash

# This script runs the evaluation for the SynCABEL model with a comprehensive set of parameters.
# It iterates through different datasets, model configurations, and generation strategies
# to produce a detailed evaluation report.

python scripts/6_evaluate_syncabel/evaluate.py \
    --datasets SPACCC \
    --model_names "mt5-large" "bart-large" "mbart-large-50" "ANGEL_pretrained" "biobart-v2-large" \
    --selection_methods "tfidf" \
    --num_beams 1 5 10 15 20 \
    --with_group True False \
    --best True False \
    --augmented_data True False \
    --constraints True False

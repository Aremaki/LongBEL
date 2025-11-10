#!/bin/bash

# This script runs the evaluation for the SynCABEL model with a comprehensive set of parameters.
# It iterates through different datasets, model configurations, and generation strategies
# to produce a detailed evaluation report.

python scripts/6_evaluate_syncabel/evaluate.py \
    --datasets SPACCC \
    --model_names "mt5-large" "bart-large" "mbart-large-50" "ANGEL_pretrained_bart" "biobart-v2-large" \
    --selection_methods "tfidf" \
    --num_beams 1 5 10 15 20 \
    --with_group True False \
    --best True False \
    --augmented_data "human_only" "synth_only" "full" \
    --constraints True False \
    --output "evaluation_results_spaccc.csv" \
    --add_group_column True

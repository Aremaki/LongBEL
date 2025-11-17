#!/bin/bash

model_names=("mt5-large" "bart-large" "mbart-large-50" "ANGEL_pretrained_bart" "biobart-v2-large")
augmented_data_options=("human_only" "synth_only" "full" "human_only_ft" "full_upsampled")

for model_name in "${model_names[@]}"; do
    for augmented_data in "${augmented_data_options[@]}"; do
        RESULT_FILE="results/all_norm_csv/norm_evaluation_results_${model_name}_${augmented_data}.csv"

        # Skip if file exists
        if [ -f "$RESULT_FILE" ]; then
            echo "Result file $RESULT_FILE already exists. Skipping."
            continue
        fi
        
        SCRIPT_ARGS="--datasets SPACCC \
            --model_names $model_name \
            --selection_methods tfidf \
            --num_beams 1 2 5 10 \
            --with_group True \
            --best True False \
            --augmented_data $augmented_data \
            --constraints True False \
            --tasks norm ner+norm"
        sbatch --export=SCRIPT_ARGS="$SCRIPT_ARGS" -A agr@cpu scripts/6_evaluate_syncabel/run.slurm
    done
done

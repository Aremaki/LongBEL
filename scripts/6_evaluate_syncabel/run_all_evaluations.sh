

#!/bin/bash

model_names=('mt5-large' 'bart-large' 'mbart-large-50' 'ANGEL_pretrained_bart' 'biobart-v2-large')
augmented_data_options=("human_only" "synth_only" "full" "human_only_ft" "full_upsampled")

for model_name in "${model_names[@]}"; do
    for augmented_data in "${augmented_data_options[@]}"; do
        RESULT_DIR="results/norm_results_${model_name}_${augmented_data}"
        if [ -d "$RESULT_DIR" ]; then
            echo "Results directory $RESULT_DIR already exists. Skipping."
            continue
        fi

        SCRIPT_ARGS="--datasets 'SPACCC' \
            --model_names $model_name \
            --selection_methods 'tfidf' \
            --num_beams 1 2 5 10 \
            --with_group True \
            --best True False \
            --augmented_data $augmented_data \
            --constraints True False \
            --tasks 'norm' 'ner+norm'"
        sbatch --export=SCRIPT_ARGS="$SCRIPT_ARGS" scripts/6_evaluate_syncabel/run.slurm
    done
done

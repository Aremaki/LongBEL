#!/bin/bash

# Define arrays for the parameters
MODELS=("google/mt5-large" "facebook/bart-large" "facebook/mbart-large-50" "dmis-lab/ANGEL_pretrained_bart" "GanjinZero/biobart-v2-large")
DATASETS=("SPACCC")
SELECTION_METHODS=("tfidf")
AUGMENTED_OPTIONS=(false)
WITH_GROUP_OPTIONS=(true)

# Loop over all combinations
for dataset in "${DATASETS[@]}"; do
    for selection in "${SELECTION_METHODS[@]}"; do
        for augmented in "${AUGMENTED_OPTIONS[@]}"; do
            for group in "${WITH_GROUP_OPTIONS[@]}"; do
                for model in "${MODELS[@]}"; do
                    # Build the arguments for the python script
                    ARGS="--model-name ${model} --dataset-name ${dataset} --selection-method ${selection}"
                    if [ "$augmented" = true ]; then
                        ARGS="${ARGS} --augmented-data"
                    fi
                    if [ "$group" = true ]; then
                        ARGS="${ARGS} --with-group"
                    fi

                    # Submit the Slurm job
                    sbatch --export=ALL,SCRIPT_ARGS="${ARGS}" -A ssq@h100 scripts/4_training_seq2seq/run.slurm
                    echo "Submitted job for: ${ARGS}"
                    sleep 1 # To avoid overwhelming the scheduler
                done
            done
        done
    done
done

echo "All training jobs submitted."

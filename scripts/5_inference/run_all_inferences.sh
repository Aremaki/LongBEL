#!/bin/bash

# Define arrays for the parameters
DATASETS=("MedMentions" "EMEA" "MEDLINE")
NUM_BEAMS=(5)
SELECTION_METHODS=("embedding" "tfidf" "levenshtein" "title")
AUGMENTED_OPTIONS=(false)
WITH_GROUP_OPTIONS=(true)

# Loop over all combinations
for dataset in "${DATASETS[@]}"; do
    for selection in "${SELECTION_METHODS[@]}"; do
        for augmented in "${AUGMENTED_OPTIONS[@]}"; do
            for group in "${WITH_GROUP_OPTIONS[@]}"; do
                for num_beams in "${NUM_BEAMS[@]}"; do
                    # Build the arguments for the python script
                    ARGS="--model-name google/mt5-large --dataset-name ${dataset} --selection-method ${selection} --num-beams ${num_beams}"
                    if [ "$augmented" = true ]; then
                        ARGS="${ARGS} --augmented-data"
                    fi
                    if [ "$group" = true ]; then
                        ARGS="${ARGS} --with-group"
                    fi

                    # Submit the Slurm job
                    sbatch --export=ALL,SCRIPT_ARGS="${ARGS}" -A ssq@h100 scripts/5_inference/run.slurm
                    echo "Submitted job for: ${ARGS}"
                    sleep 1 # To avoid overwhelming the scheduler
                done
            done
        done
    done
done

echo "All inference jobs submitted."

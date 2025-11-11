#!/bin/bash

# Define arrays for the parameters
MODELS=("mt5-large" "bart-large" "mbart-large-50" "ANGEL_pretrained_bart" "biobart-v2-large")
DATASETS=("SPACCC")
NUM_BEAMS=(1 5 10)
SELECTION_METHODS=("tfidf")
AUGMENTED_OPTIONS=("human_only" "human_only_ft" "synth_only" "full" "full_upsampled")
WITH_GROUP_OPTIONS=(true)
BEST_OPTIONS=(true false)
BATCH_SIZE=64
# Loop over all combinations
for dataset in "${DATASETS[@]}"; do
    for selection in "${SELECTION_METHODS[@]}"; do
        for augmented in "${AUGMENTED_OPTIONS[@]}"; do
            for group in "${WITH_GROUP_OPTIONS[@]}"; do
                for num_beams in "${NUM_BEAMS[@]}"; do
                    for model in "${MODELS[@]}"; do
                        for best in "${BEST_OPTIONS[@]}"; do
                            # Build the arguments for the python script
                            ARGS="--model-name ${model} --dataset-name ${dataset} --selection-method ${selection} --num-beams ${num_beams} --batch-size ${BATCH_SIZE} --augmented-data ${augmented}"
                            if [ "$group" = true ]; then
                                ARGS="${ARGS} --with-group"
                            fi
                            if [ "$best" = true ]; then
                                ARGS="${ARGS} --best"
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
    done
done

echo "All inference jobs submitted."

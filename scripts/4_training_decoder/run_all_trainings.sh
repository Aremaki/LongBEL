#!/bin/bash

BASE_OUTPUT_DIR="models/NED"

MODELS=("meta-llama/Meta-Llama-3-8B-Instruct")
DATASETS=("SPACCC")
SELECTION_METHODS=("tfidf")
AUGMENTED_OPTIONS=("human_only" "full_upsampled")
HUMAN_RATIOS=(0.2 0.4 0.6 0.8)

for dataset in "${DATASETS[@]}"; do
    for selection in "${SELECTION_METHODS[@]}"; do
        for augmented in "${AUGMENTED_OPTIONS[@]}"; do
            for human_ratio in "${HUMAN_RATIOS[@]}"; do
                for model in "${MODELS[@]}"; do
                    # Construct arguments for the training script (pass full model id)
                    ARGS="--model-name ${model} --dataset-name ${dataset} --selection-method ${selection} --augmented-data ${augmented} --human-ratio ${human_ratio}"

                    # Submit job
                    echo "Submitting training job (missing): ${MODEL_DIR}"
                    echo "  ARGS=${ARGS}"
                    sbatch --export=ALL,SCRIPT_ARGS="${ARGS}" -A ssq@h100 scripts/4b_training_decoder/run.slurm
                    sleep 1
                done
            done
        done
    done
done

echo "All training jobs checked and submitted if needed."

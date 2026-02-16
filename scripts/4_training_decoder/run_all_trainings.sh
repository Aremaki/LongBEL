#!/bin/bash

BASE_OUTPUT_DIR="models/NED"

MODELS=("Llama-3.1-8B-Instruct", "Llama-3.2-1B-Instruct")
DATASETS=("MedMentions")
SELECTION_METHODS=("tfidf")
AUGMENTED_OPTIONS=("human_only")
LONG_FORMAT=("true")

for dataset in "${DATASETS[@]}"; do
    for selection in "${SELECTION_METHODS[@]}"; do
        for augmented in "${AUGMENTED_OPTIONS[@]}"; do
            for long_format in "${LONG_FORMAT[@]}"; do
                for model in "${MODELS[@]}"; do
                    LONG_FORMAT_ARG=""
                    if [[ "${long_format}" == "true" ]]; then
                        LONG_FORMAT_ARG=" --long-format"
                    fi

                    # Construct arguments for the training script (pass full model id)
                    ARGS="--model-name ${model} --dataset-name ${dataset} --selection-method ${selection} --augmented-data ${augmented}${LONG_FORMAT_ARG}"

                    # Submit job
                    echo "Submitting training job (missing): ${MODEL_DIR}"
                    echo "  ARGS=${ARGS}"
                    sbatch --export=ALL,SCRIPT_ARGS="${ARGS}" -A ssq@h100 scripts/4_training_decoder/run.slurm
                    sleep 1
                done
            done
        done
    done
done

echo "All training jobs checked and submitted if needed."

#!/bin/bash

BASE_OUTPUT_DIR="models/NED"

MODELS=("google/mt5-large" "facebook/bart-large" "facebook/mbart-large-50" "dmis-lab/ANGEL_pretrained_bart" "GanjinZero/biobart-v2-large")
DATASETS=("SPACCC_UMLS")
SELECTION_METHODS=("tfidf")
AUGMENTED_OPTIONS=("human_only")
WITH_GROUP_OPTIONS=(true)

for dataset in "${DATASETS[@]}"; do
    for selection in "${SELECTION_METHODS[@]}"; do
        for augmented in "${AUGMENTED_OPTIONS[@]}"; do
            for with_group in "${WITH_GROUP_OPTIONS[@]}"; do
                for model in "${MODELS[@]}"; do

                    # Use only the last path component of the model string
                    model_name="${model##*/}"   # e.g. google/mt5-large -> mt5-large

                    # Compute expected output directory:
                    # models/NED/<dataset>_<augmented>_<selection>[_with_group]/<model_name>/model_best
                    MODEL_DIR="${BASE_OUTPUT_DIR}/${dataset}_${augmented}_${selection}"
                    if [ "$with_group" = true ]; then
                        MODEL_DIR="${MODEL_DIR}_with_group"
                    fi
                    MODEL_DIR="${MODEL_DIR}/${model_name}/model_best"

                    # Check if training already completed
                    if [ -d "$MODEL_DIR" ]; then
                        echo "Skipping (already trained): $MODEL_DIR"
                        continue
                    fi

                    # Construct arguments for the training script (pass full model id)
                    ARGS="--model-name ${model} --dataset-name ${dataset} --selection-method ${selection} --augmented-data ${augmented}"
                    if [ "$with_group" = true ]; then
                        ARGS="${ARGS} --with-group"
                    fi

                    # Submit job
                    echo "Submitting training job (missing): ${MODEL_DIR}"
                    echo "  ARGS=${ARGS}"
                    sbatch --export=ALL,SCRIPT_ARGS="${ARGS}" -A ssq@h100 scripts/4_training_seq2seq/run.slurm
                    sleep 1

                done
            done
        done
    done
done

echo "All training jobs checked and submitted if needed."

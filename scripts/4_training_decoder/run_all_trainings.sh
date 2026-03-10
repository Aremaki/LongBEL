#!/bin/bash

BASE_OUTPUT_DIR="models/NED"

MODELS=("Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct" "Llama-3.1-8B-Instruct")
DATASETS=("MedMentions" "SPACCC" "EMEA" "MEDLINE")
SELECTION_METHODS=("tfidf")
AUGMENTED_OPTIONS=("human_only")
CONTEXT_FORMAT=("short" "long" "hybrid_short" "hybrid_long")
COMPLETE_MODE=(true false)

for dataset in "${DATASETS[@]}"; do
    for selection in "${SELECTION_METHODS[@]}"; do
        for augmented in "${AUGMENTED_OPTIONS[@]}"; do
            for context_format in "${CONTEXT_FORMAT[@]}"; do
                for complete_mode in "${COMPLETE_MODE[@]}"; do
                for model in "${MODELS[@]}"; do
                    COMPLETE_MODE_ARG=""
                    if [[ "${complete_mode}" == true ]]; then
                        COMPLETE_MODE_ARG=" --complete-mode"
                    fi
                    # Check if model already exists
                    MODEL_DIR="${BASE_OUTPUT_DIR}/${dataset}_${augmented}_${selection}_${context_format}"
                    if [[ "${complete_mode}" == true ]]; then
                        MODEL_DIR="${MODEL_DIR}_complete"
                    fi
                    MODEL_DIR="${MODEL_DIR}/${model}/model_best"
                    if [ -d "$MODEL_DIR" ]; then
                        echo "Skipping: Model already exists → ${MODEL_DIR}"
                        continue
                    fi

                    # Construct arguments for the training script (pass full model id)
                    ARGS="--model-name ${model} --dataset-name ${dataset} --selection-method ${selection} --augmented-data ${augmented} --context-format ${context_format} ${COMPLETE_MODE_ARG}"

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

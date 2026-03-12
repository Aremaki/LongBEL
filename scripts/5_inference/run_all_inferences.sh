#!/bin/bash

BASE_OUTPUT_DIR="results/inference_outputs"
BASE_MODEL_DIR="models/NED"

DATASETS=("SPACCC" "MedMentions" "EMEA" "MEDLINE")
SELECTION_METHODS=("tfidf")
SPLIT_NAMES=("test")
AUGMENTED_OPTIONS=("human_only")
CONTEXT_FORMAT=("short" "long" "hybrid_short" "hybrid_long" "hybrid_medium")
COMPLETE_MODE=(true false)
ADD_HEADERS=(true false)
NUM_BEAMS=(5)
MODELS=("Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct" "Llama-3.1-8B-Instruct")
BEST_OPTIONS=(true false)
BATCH_SIZE=16

for dataset in "${DATASETS[@]}"; do
    for selection in "${SELECTION_METHODS[@]}"; do
        for split in "${SPLIT_NAMES[@]}"; do
            for augmented in "${AUGMENTED_OPTIONS[@]}"; do
                for context_format in "${CONTEXT_FORMAT[@]}"; do
                    for complete_mode in "${COMPLETE_MODE[@]}"; do
                        for num_beams in "${NUM_BEAMS[@]}"; do
                            for model in "${MODELS[@]}"; do
                                for best in "${BEST_OPTIONS[@]}"; do

                                    # ----------------------------
                                    # Compute complete mode string
                                    # ----------------------------
                                    complete_mode_str=""
                                    if [[ "${complete_mode}" == true ]]; then
                                        complete_mode_str="_complete"
                                    fi

                                    # ----------------------------
                                    # Compute add headers string
                                    # ----------------------------
                                    add_headers_str=""
                                    if [[ "${add_headers}" == true ]]; then
                                        add_headers_str="_addheaders"
                                    fi

                                    # Determine the output folder
                                    FOLDER="${BASE_OUTPUT_DIR}/${dataset}/${augmented}_${selection}_${context_format}${complete_mode_str}${add_headers_str}"

                                    MODEL_FOLDER="${FOLDER}/${model}_"
                                    if [ "$best" = true ]; then
                                        MODEL_FOLDER="${MODEL_FOLDER}best"
                                    else
                                        MODEL_FOLDER="${MODEL_FOLDER}last"
                                    fi

                                    OUTPUT_FILE="${MODEL_FOLDER}/pred_${split}_constraint_${num_beams}_beams.tsv"

                                    # Check if job already done
                                    if [ -f "$OUTPUT_FILE" ]; then
                                        echo "Skipping: Output already exists → $OUTPUT_FILE"
                                        continue
                                    fi
                                    # Check if model already exists
                                    MODEL_DIR="${BASE_MODEL_DIR}/${dataset}_${augmented}_${selection}_${context_format}"
                                    if [[ "${complete_mode}" == true ]]; then
                                        MODEL_DIR="${MODEL_DIR}_complete"
                                    fi
                                    if [[ "${add_headers}" == true ]]; then
                                        MODEL_DIR="${MODEL_DIR}_addheaders"
                                    fi
                                    if [ "$best" = true ]; then
                                        MODEL_FOLDER="${MODEL_FOLDER}/${model}/model_best"
                                    else
                                        MODEL_FOLDER="${MODEL_FOLDER}/${model}/model_last"
                                    fi
                                    if [ ! -d "$MODEL_DIR" ]; then
                                        echo "Skipping: Model doesn't exist → ${MODEL_DIR}"
                                        continue
                                    fi

                                    # Build Slurm args
                                    ARGS="--model-name ${model} --dataset-name ${dataset} --selection-method ${selection} --split-name ${split} --num-beams ${num_beams} --batch-size ${BATCH_SIZE} --augmented-data ${augmented} --context-format ${context_format}"

                                    if [ "$best" = true ]; then
                                        ARGS="${ARGS} --best"
                                    fi

                                    if [[ "${complete_mode}" == true ]]; then
                                        ARGS="${ARGS} --complete-mode"
                                    fi

                                    if [[ "${add_headers}" == true ]]; then
                                        ARGS="${ARGS} --add-headers"
                                    fi

                                    # Submit job
                                    echo "Submitting: ${ARGS} (missing output)"
                                    sbatch --export=ALL,SCRIPT_ARGS="${ARGS}" -A ssq@h100 scripts/5_inference/run.slurm
                                    sleep 1

                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All available jobs checked and submitted if necessary."

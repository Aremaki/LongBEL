#!/bin/bash

BASE_OUTPUT_DIR="results/inference_outputs"

DATASETS=("SPACCC" "MedMentions" "EMEA" "MEDLINE")
SELECTION_METHODS=("tfidf")
SPLIT_NAMES=("test")
AUGMENTED_OPTIONS=("human_only")
LONG_FORMAT=(true false)
NUM_BEAMS=(5)
MODELS=("Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct" "Llama-3.1-8B-Instruct")
BEST_OPTIONS=(true false)
BATCH_SIZE=16

for dataset in "${DATASETS[@]}"; do
    for selection in "${SELECTION_METHODS[@]}"; do
        for split in "${SPLIT_NAMES[@]}"; do
            for augmented in "${AUGMENTED_OPTIONS[@]}"; do
                for long_format in "${LONG_FORMAT[@]}"; do
                    for num_beams in "${NUM_BEAMS[@]}"; do
                        for model in "${MODELS[@]}"; do
                            for best in "${BEST_OPTIONS[@]}"; do

                                # ----------------------------
                                # Compute long format string
                                # ----------------------------
                                long_format_str=""
                                if [[ "${long_format}" == true ]]; then
                                    long_format_str="_long"
                                fi
 
                                # Determine the output folder
                                FOLDER="${BASE_OUTPUT_DIR}/${dataset}/${augmented}_${selection}${long_format_str}"

                                MODEL_FOLDER="${FOLDER}/${model}_"
                                if [ "$best" = true ]; then
                                    MODEL_FOLDER="${MODEL_FOLDER}best"
                                else
                                    MODEL_FOLDER="${MODEL_FOLDER}last"
                                fi
                                
                                # Check if model folder exists
                                if [ ! -d "$MODEL_FOLDER" ]; then
                                    echo "Skipping: Model folder does not exist → ${MODEL_FOLDER}"
                                    continue
                                fi

                                OUTPUT_FILE="${MODEL_FOLDER}/pred_${split}_constraint_${num_beams}_beams.tsv"

                                # Check if job already done
                                if [ -f "$OUTPUT_FILE" ]; then
                                    echo "Skipping: Output already exists → $OUTPUT_FILE"
                                    continue
                                fi

                                # Build Slurm args
                                ARGS="--model-name ${model} --dataset-name ${dataset} --selection-method ${selection} --split-name ${split} --num-beams ${num_beams} --batch-size ${BATCH_SIZE} --augmented-data ${augmented}"

                                if [ "$best" = true ]; then
                                    ARGS="${ARGS} --best"
                                fi

                                if [[ "${long_format}" == true ]]; then
                                    ARGS="${ARGS} --long-format"
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

echo "All available jobs checked and submitted if necessary."

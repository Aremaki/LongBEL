#!/bin/bash

BASE_OUTPUT_DIR="results/inference_outputs"

MODELS=("mt5-large" "bart-large" "mbart-large-50" "ANGEL_pretrained_bart" "biobart-v2-large")
DATASETS=("SPACCC")
NUM_BEAMS=(1 2 5 10)
SELECTION_METHODS=("tfidf")
SPLIT_NAMES=("test" "test_ner")
AUGMENTED_OPTIONS=("human_only" "human_only_ft" "synth_only" "full" "full_upsampled")
WITH_GROUP_OPTIONS=(true)
BEST_OPTIONS=(true false)
BATCH_SIZE=8

for dataset in "${DATASETS[@]}"; do
    for selection in "${SELECTION_METHODS[@]}"; do
        for split in "${SPLIT_NAMES[@]}"; do
            for augmented in "${AUGMENTED_OPTIONS[@]}"; do
                for with_group in "${WITH_GROUP_OPTIONS[@]}"; do
                    for num_beams in "${NUM_BEAMS[@]}"; do
                        for model in "${MODELS[@]}"; do
                            for best in "${BEST_OPTIONS[@]}"; do

                                # Determine the output folder
                                FOLDER="${BASE_OUTPUT_DIR}/${dataset}/${augmented}_${selection}"
                                if [ "$with_group" = true ]; then
                                    FOLDER="${FOLDER}_with_group"
                                fi

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

                                # Build Slurm args
                                ARGS="--model-name ${model} --dataset-name ${dataset} --selection-method ${selection} --split-name ${split} --num-beams ${num_beams} --batch-size ${BATCH_SIZE} --augmented-data ${augmented}"

                                if [ "$with_group" = true ]; then
                                    ARGS="${ARGS} --with-group"
                                fi

                                if [ "$best" = true ]; then
                                    ARGS="${ARGS} --best"
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

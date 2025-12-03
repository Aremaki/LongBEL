#!/bin/bash

BASE_OUTPUT_DIR="results/inference_outputs"

MODELS=("Meta-Llama-3-8B-Instruct")
DATASETS=("SPACCC" "MedMentions" "EMEA" "MEDLINE")
NUM_BEAMS=(5 10)
SELECTION_METHODS=("tfidf")
SPLIT_NAMES=("test")
AUGMENTED_OPTIONS=("human_only" "full" "full_upsampled")
HUMAN_RATIOS=(0.2 0.4 0.6 0.8 1.0)
BEST_OPTIONS=(true false)
BATCH_SIZE=16

for dataset in "${DATASETS[@]}"; do
    for selection in "${SELECTION_METHODS[@]}"; do
        for split in "${SPLIT_NAMES[@]}"; do
            for augmented in "${AUGMENTED_OPTIONS[@]}"; do
                for human_ratio in "${HUMAN_RATIOS[@]}"; do
                    for num_beams in "${NUM_BEAMS[@]}"; do
                        for model in "${MODELS[@]}"; do
                            for best in "${BEST_OPTIONS[@]}"; do

                                # ----------------------------
                                # Compute human_ratio string
                                # ----------------------------
                                human_ratio_str=""
                                # Use awk for floating point comparison
                                if awk "BEGIN {exit !($human_ratio < 1.0)}"; then
                                    # Multiply by 100 and round to nearest integer
                                    pct=$(awk "BEGIN {printf \"%d\", $human_ratio * 100 + 0.5}")
                                    human_ratio_str="_${pct}pct"
                                fi

                                # Determine the output folder
                                FOLDER="${BASE_OUTPUT_DIR}/${dataset}/${augmented}_${selection}${human_ratio_str}"

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
                                ARGS="--model-name ${model} --dataset-name ${dataset} --selection-method ${selection} --split-name ${split} --num-beams ${num_beams} --batch-size ${BATCH_SIZE} --augmented-data ${augmented} --human-ratio ${human_ratio}"

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

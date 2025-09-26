# 5. Inference

This directory contains the scripts for running inference with the trained sequence-to-sequence models for Named Entity Disambiguation (NED). The process involves generating predictions on the test set using the models trained in step 4.

## Files

-   `infer.py`: The core Python script that performs inference. It loads a trained model and a test set, then generates predictions. It supports both unconstrained beam search and constrained beam search using a pre-computed Trie of candidate entities to ensure that the generated output is a valid entity.
-   `run_all_inferences.sh`: A shell script designed to launch inference jobs for multiple model configurations. It iterates over different datasets, selection methods, and other parameters, submitting a separate Slurm job for each combination.
-   `run.slurm`: A Slurm script that defines the necessary computational resources (like GPUs and CPUs) and executes the `infer.py` script with the appropriate arguments.

## Usage

To run inference for all the model configurations, you can execute the `run_all_inferences.sh` script.

```bash
bash scripts/5_inference/run_all_inferences.sh
```

This will submit multiple jobs to the Slurm scheduler. You can monitor the progress of these jobs using standard Slurm commands like `squeue`.

### `infer.py` Arguments

The `infer.py` script accepts the following command-line arguments:

-   `--model-name`: (Required) The name of the base model to use (e.g., `google/mt5-large`).
-   `--num-beams`: The number of beams to use for beam search during generation. Default is `5`.
-   `--best`: If specified, the model from the best checkpoint will be used. Otherwise, the last checkpoint is used.
-   `--dataset-name`: The name of the dataset to use for inference (e.g., `MedMentions`). Default is `MedMentions`.
-   `--selection-method`: The candidate selection method used during training (e.g., `random`, `tfidf`, `embedding`). Default is `random`.
-   `--with-group`: If specified, it indicates that the model was trained with semantic group information.
-   `--augmented-data`: If specified, it indicates that the model was trained on augmented data.

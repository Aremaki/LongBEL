# Step 4a: Train Seq2Seq Model (Encoder-Decoder)

This directory contains the scripts for training Sequence-to-Sequence (Seq2Seq) models like mBART or T5 for the Entity Linking task. These models take the clinical text as input and generate the target entity name or identifier.

## Files

-   **`train.py`**: The main training script using Hugging Face Transformers. It handles data loading, model initialization, and the training loop.
-   **`run.slurm`**: A Slurm submission script for running the training on a cluster environment.
-   **`run_all_trainings.sh`**: A shell script to automate running multiple experiments with different hyperparameters or datasets.

## Usage

### Single Training Run

To train a single model, use `train.py`.

```bash
uv run scripts/4a_training_seq2seq/train.py \
    --model-name "facebook/mbart-large-50" \
    --dataset-name "MedMentions" \
    --use-augmented-data
```

**Arguments:**

-   `--model-name`: The pre-trained model checkpoint (e.g., `facebook/mbart-large-50`, `google/mt5-large`).
-   `--dataset-name`: The dataset to train on (e.g., `MedMentions`, `SPACCC`).
-   `--use-augmented-data`: Flag to include synthetic data in the training set.
-   `--batch-size`: Batch size for training.
-   `--learning-rate`: Learning rate for the optimizer.

### Batch Experiments

To run a suite of experiments:

```bash
uv run scripts/4a_training_seq2seq/run_all_trainings.sh
```

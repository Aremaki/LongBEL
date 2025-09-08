# Step 4: Model Training

This directory contains the scripts for training the Named Entity Disambiguation (NED) models.

## Files

-   `train.py`: The main Python script for training the sequence-to-sequence models.
-   `run.slurm`: A Slurm script for submitting a training job to a cluster.
-   `run_all_trainings.sh`: A bash script that automates the process of submitting multiple training jobs with different configurations.

## Usage

### `train.py`

This script trains a model with the specified parameters.

**Arguments:**

-   `--model-name`: (Required) The name or path of the pre-trained model to use (e.g., `google/mt5-large`).
-   `--lr`: (Optional) The learning rate for the optimizer. Default is `3e-05`.
-   `--dataset-name`: (Required) The name of the dataset to use for training (e.g., `MedMentions`).
-   `--augmented-data`: (Optional) A flag to indicate whether to use augmented data for training.
-   `--with-group`: (Optional) A flag to indicate whether to use data with group annotations.
-   `--selection-method`: (Optional) The method used for selecting concept synonyms. Choices are `embedding`, `tfidf`, and `levenshtein`. Default is `embedding`.

**Example:**

```bash
python scripts/4_training/train.py --model-name google/mt5-large --dataset-name MedMentions --augmented-data --with-group --selection-method embedding
```

### `run.slurm`

This script is configured to submit a training job to a Slurm-managed cluster. It sets up the environment, loads the necessary modules, and then executes `train.py` with the arguments passed to it.

The script is designed to be launched via `sbatch`, and it expects the python script arguments to be passed through the `SCRIPT_ARGS` environment variable.

### `run_all_trainings.sh`

This script automates running multiple training experiments. It iterates through different datasets, selection methods, and data augmentation options, submitting a separate Slurm job for each combination.

**To run all training configurations:**

```bash
bash scripts/4_training/run_all_trainings.sh
```

This will submit multiple jobs to the Slurm scheduler based on the configurations defined in the script.

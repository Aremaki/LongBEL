# Step 5: Inference (Generative Entity Linking)

This directory contains scripts for running inference using the trained generative models (both Seq2Seq and Decoder-only). It generates predictions on the test set, optionally using constrained beam search with a Trie for valid entity generation.

## Files

-   **`infer.py`**: The core inference script. It loads the trained model and test data, performs generation (beam search), and saves the predictions.
-   **`run.slurm`**: Slurm script for inference jobs.
-   **`run_all_inferences.sh`**: Automation script to run inference across multiple model checkpoints and configurations.

## Usage

### Single Inference Run

To run inference with a specific model and configuration:

```bash
uv run scripts/5_inference/infer.py \
    --model-name "google/mt5-large" \
    --dataset-name "MedMentions" \
    --num-beams 5 \
    --best
```

**Arguments:**

-   `--model-name`: The base model name used during training.
-   `--dataset-name`: The dataset to evaluate on.
-   `--num-beams`: Number of beams for beam search.
-   `--best`: Use the best performing checkpoint (based on validation) instead of the last one.
-   `--selection-method`: (Optional) If applicable, the method used for candidate selection training.

### Batch Inference

To run inference for all trained models:

```bash
uv run scripts/5_inference/run_all_inferences.sh
```

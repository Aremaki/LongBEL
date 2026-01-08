# Step 4b: Train Decoder-Only Model (LLM)

This directory contains the scripts for fine-tuning Decoder-only Large Language Models (LLMs) like Llama-3 or Mistral for Generative Entity Linking. Unlike the Seq2Seq approach, this methods fine-tunes a causal language model to generate the entity given the context.

## Files

-   **`train.py`**: The training script using `trl` (Transformer Reinforcement Learning) library's `SFTTrainer` for Supervised Fine-Tuning. It supports PEFT (LoRA/QLoRA) and other advanced training techniques.
-   **`run.slurm`**: Slurm script for cluster job submission.
-   **`run_all_trainings.sh`**: Automation script for multiple experiments.

## Usage

### Single Training Run

To fine-tune a model:

```bash
uv run scripts/4b_training_decoder/train.py \
    --model-name "meta-llama/Meta-Llama-3-8B" \
    --dataset-name "MedMentions" \
    --augmented-data
```

**Arguments:**

-   `--model-name`: The pre-trained LLM path or identifier (e.g., `meta-llama/Meta-Llama-3-8B`).
-   `--dataset-name`: Target dataset (e.g., `MedMentions`).
-   `--augmented-data`: Include synthetic data.
-   `--selection-method`: Method for candidate selection/negative sampling (e.g., `embedding`, `tfidf`).

### Batch Experiments

```bash
uv run scripts/4b_training_decoder/run_all_trainings.sh
```

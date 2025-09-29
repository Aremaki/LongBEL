# Step 6: Evaluate SynCABEL

This folder contains scripts for evaluating the SynCABEL model.

## `evaluate.py`

This script evaluates the trained SynCABEL model on the test sets of MedMentions and QUAERO. It computes various metrics to assess the model's performance and saves the results in `evaluation_results.csv`.

The script calculates the following recall metrics:
- **Recall**: Overall recall.
- **Recall Seen Mention**: Recall for mentions that were seen during training.
- **Recall Unseen Mention**: Recall for mentions that were not seen during training.
- **Recall Seen CUI**: Recall for CUIs that were seen during training.
- **Recall Unseen CUI**: Recall for CUIs that were not seen during training.
- **Recall Top 100 CUI**: Recall for the top 100 most frequent CUIs in the training data.

### Arguments

- `--datasets`: A list of datasets to evaluate on. Choices: `MedMentions`, `QUAERO`.
- `--model_names`: A list of model names to evaluate.
- `--selection_methods`: A list of selection methods used during training.
- `--num_beams`: A list of beam sizes to use for generation.
- `--with_group`: Whether the model was trained with group information.
- `--best`: Whether to use the best or last model checkpoint.
- `--augmented_data`: Whether the model was trained on augmented data.
- `--constraints`: Whether to use constraints during generation.

## `run.sh`

This is a convenience script to run the evaluation with a predefined set of parameters. It will generate the `evaluation_results.csv` file.

### Usage

```bash
bash scripts/6_evaluate_syncabel/run.sh
```

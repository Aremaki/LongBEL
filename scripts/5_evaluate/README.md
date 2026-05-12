# Step 6: Evaluation & Error Analysis

This directory contains scripts for evaluating the LongBEL model's performance, ranging from standard metrics (Recall) to advanced semantic analysis (LLM-as-a-judge).

## Scripts

Computes standard entity linking metrics (Recall@1) and stratified performance (Seen vs. Unseen mentions/concepts).

**Metrics:**
- **Recall**: Overall linking accuracy.
- **Recall Seen Mention**: Accuracy on entity mentions present in the training set.
- **Recall Unseen Mention**: Accuracy on mentions never seen during training (Generalization).
- **Recall Top 100 CUI**: Performance on the most frequent concepts.

**Usage:**
```bash
uv run scripts/6_evaluate/error_analysis.py
```
*(Note: This script currently expects `results/inference_outputs/...` structure)*

## Expected Directory Structure for Results
The scripts expect inference outputs to be located at:
`results/inference_outputs/<DATASET>/<AUGMENTATION_TYPE>_<SELECTION>/<MODEL_NAME>_best/pred_<SPLIT>_constraint_<BEAMS>_beams.tsv`

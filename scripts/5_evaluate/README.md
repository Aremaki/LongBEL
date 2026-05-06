# Step 6: Evaluation & Error Analysis

This directory contains scripts for evaluating the LongBEL model's performance, ranging from standard metrics (Recall) to advanced semantic analysis (LLM-as-a-judge).

## Scripts

### 1. `error_analysis.py` (Standard Recalls)
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

### 2. `evaluate_llm.py` (LLM-as-a-judge)
Uses a large language model (e.g., Llama-3-70B) to evaluate the **semantic correctness** of predictions that didn't effectively match the gold code. It classifies errors into:
- `CORRECT`: Semantically equivalent
- `BROAD`: Prediction is a broader concept
- `NARROW`: Prediction is a narrower concept
- `NO_RELATION`: Wrong prediction

**Usage:**
```bash
uv run scripts/6_evaluate/evaluate_llm.py \
    --datasets SPACCC \
    --model-name Meta-Llama-3-8B-Instruct
```

### 3. `evaluate_termino.py` (Ontology-based)
Evaluates predictions using ontology structure (e.g., SNOMED-CT hierarchy). It checks if a "wrong" prediction is actually a parent or child of the gold concept in the knowledge graph.

### 4. `mixed_ratio.py`
Analyzes performance across different ratios of synthetic vs. human data if you trained multiple mixed models.

## Expected Directory Structure for Results
The scripts expect inference outputs to be located at:
`results/inference_outputs/<DATASET>/<AUGMENTATION_TYPE>_<SELECTION>/<MODEL_NAME>_best/pred_<SPLIT>_constraint_<BEAMS>_beams.tsv`

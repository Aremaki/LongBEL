<div align="center">
    <img src="figures/overview.svg" alt="LongBEL">
    <p align="center">
<a href="https://doi.org/10.48550/arXiv.2601.19667"><img src="https://zenodo.org/badge/DOI/paper.svg" alt="DOI"></a>
<a href="https://github.com/astral-sh/uv" target="_blank">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="UV">
</a>
<a href="https://github.com/astral-sh/ruff" target="_blank">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" alt="Ruff">
</a>
<a href="https://anonymous.4open.science/r/LongBEL-31AD/LICENSE">
   <img alt="GitHub" src="https://img.shields.io/badge/license-MIT-blue">
</a>
<h3>
    <a href="https://huggingface.co/collections/AnonymousARR42/longbel">🤗 LongBEL Hugging Face Collection</a>
</h3>
</div>

## Introduction

**LongBEL** is a document-level framework for biomedical entity linking (BEL). Instead of normalizing each mention independently, LongBEL conditions each prediction on the document context and on previous normalizations produced in the same document.

This design helps the model enforce document-level consistency, especially for recurring biomedical concepts, abbreviations, and underspecified mentions whose correct normalization can be inferred from earlier mentions.

This repository contains the complete pipeline for:

- **Knowledge base preprocessing**: preparing UMLS and SNOMED CT candidate dictionaries.
- **Dataset construction**: converting biomedical entity linking datasets into LongBEL inputs with document context and prediction memory.
- **Robust memory training**: constructing realistic prediction memory using cross-validated predictions.
- **Fine-tuning**: training decoder-only and sequence-to-sequence generative BEL models.
- **Constrained decoding**: restricting generation to valid concepts from the target knowledge base.
- **Evaluation and analysis**: computing Recall@1, consistency metrics, cascading-error analysis, and qualitative examples.

## LongBEL Input Format

LongBEL represents each example using three sections:

```text
### Context
Document context containing the target [mention].

### Previous Normalizations
[previous mention]{semantic group} previous predicted concept
...

### Prediction
[target mention]{semantic group}
````

For example:

```text
### Context
Remitting seronegative symmetrical synovitis with pitting edema ([RS3PE]) is a rare condition...
The patient was diagnosed with [RS3PE].

### Previous Normalizations
[Remitting seronegative symmetrical synovitis with pitting edema]{Disorders} Remitting seronegative symmetrical synovitis with pitting edema

### Prediction
[RS3PE]{Disorders}
```

Expected output:

```text
Remitting seronegative symmetrical synovitis with pitting edema
```

## Direct Use

### Fine-tuned Models

We release fine-tuned LongBEL checkpoints for the main benchmarks used in the paper:

🤗 [AnonymousARR42/LongBEL_8B_MedMentions_st21pv](https://huggingface.co/AnonymousARR42/LongBEL_8B_MedMentions_st21pv) 
🤗 [AnonymousARR42/LongBEL_1B_MedMentions_st21pv](https://huggingface.co/AnonymousARR42/LongBEL_1B_MedMentions_st21pv)   
🤗 [AnonymousARR42/LongBEL_8B_QUAERO_EMEA](https://huggingface.co/AnonymousARR42/LongBEL_8B_QUAERO_EMEA)  
🤗 [AnonymousARR42/LongBEL_1B_QUAERO_EMEA](https://huggingface.co/AnonymousARR42/LongBEL_1B_QUAERO_EMEA)  
🤗 [AnonymousARR42/LongBEL_8B_SPACCC](https://huggingface.co/AnonymousARR42/LongBEL_SPACCC)  
🤗 [AnonymousARR42/LongBEL_1B_SPACCC](https://huggingface.co/AnonymousARR42/LongBEL_1B_SPACCC)  

Each checkpoint includes the model and the resources required for constrained biomedical entity linking.

### Loading

```python
import torch
from longbel.models import LongBEL

device = "cuda" if torch.cuda.is_available() else "cpu"

model = (
    LongBEL.from_pretrained("AnonymousARR42/LongBEL_8B_MedMentions_st21pv")
    .eval()
    .to(device)
)
```

### Inference

LongBEL expects inputs in the [BigBio KB format](https://huggingface.co/datasets/bigbio), the schema used by biomedical entity-linking datasets such as MedMentions and QUAERO on Hugging Face.

```python
from datasets import load_dataset

dataset = load_dataset(
    "bigbio/medmentions",
    "medmentions_st21pv_bigbio_kb",
    split="test",
)

examples = [dataset[i] for i in range(4)]

results = model.sample(
    data=examples,
    constrained=True,
    num_beams=2,
)

for example, beam_results in zip(examples, results):
    print(f"Document ID: {example['document_id']}")

    for result in beam_results:
        print(
            f"Predicted concept name: {result['pred_concept_name']}\n"
            f"Predicted code: {result['pred_concept_code']}\n"
            f"Beam score: {result['beam_score']:.3f}\n"
        )
```

### Datasets

We provide the processed BigBio-style datasets used with LongBEL on Hugging Face:

- 🤗 [AnonymousARR42/MedMentions](https://huggingface.co/datasets/AnonymousARR42/MedMentions)
- 🤗 [AnonymousARR42/EMEA](https://huggingface.co/datasets/AnonymousARR42/EMEA)
- 🤗 [AnonymousARR42/SPACCC](https://huggingface.co/datasets/AnonymousARR42/SPACCC)

These datasets are formatted for direct use with LongBEL inference and evaluation. The original benchmark sources are MedMentions, QUAERO-EMEA, and the Spanish SPACCC datasets.

## Customize Your Own LongBEL Model

### Requirements

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository:

```bash
git clone https://anonymous.4open.science/r/LongBEL-31AD
cd LongBEL
```

Create the environment:

```bash
uv venv --python 3.9
source .venv/bin/activate
uv sync
```

## Pipeline

### Step 1: Preprocess the Knowledge Base

Prepare the terminology resources used for candidate generation and constrained decoding.

#### UMLS-based datasets

For MedMentions and QUAERO:

```bash
uv run python scripts/1_preprocess_termino/run_extract_and_prepare_umls.py
```

#### SNOMED CT / SPACCC

For SPACCC-style gazetteers:

```bash
uv run bash scripts/1b_preprocess_SPACCC/run.sh
```

### Step 2: Prepare LongBEL Data

Convert the original entity linking datasets into LongBEL inputs. This step builds the context, previous normalization memory, and target concept strings.

```bash
uv run python scripts/2_prepare_data/run.py
```

The resulting examples follow this structure:

```text
### Context
...

### Previous Normalizations
...

### Prediction
...
```

### Step 3: Construct Robust Memory

LongBEL uses previous predictions as memory. To avoid training only with perfect gold memory, we construct realistic memory using cross-validation: models trained on several folds generate predictions for held-out folds, and these predictions are used as memory during final training.

```bash
uv run python scripts/2_prepare_data/build_robust_memory.py
```

### Step 4: Train LongBEL

```bash
uv run python scripts/3_training_decoder/train.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name MedMentions \
    --context-format hybrid_long
```

### Step 5: Run Inference

```bash
uv run python scripts/4_inference/infer.py \
    --model-name <path_to_checkpoint> \
    --dataset-name MedMentions \
    --context-format hybrid_long \
    --constrained
```

### Step 6: Evaluate

Compute standard entity linking metrics:

```bash
uv run python scripts/5_evaluate/error_analysis.py
```

Run document-level consistency and cascading-error analyses:

```bash
uv run python scripts/5_evaluate/consistency_analysis.py
uv run python scripts/5_evaluate/copy_wrong_memory_error.py
```

## Project Structure

```text
LongBEL/
├── scripts/
│   ├── 1_preprocess_termino/       # UMLS preprocessing
│   ├── 2_prepare_data/             # LongBEL data and memory construction
│   ├── 3_training_decoder/        # Decoder-only training
│   ├── 4_inference/                # Inference and constrained decoding
│   └── 5_evaluate/                 # Evaluation and analysis
├── longbel/
│   ├── guided_inference.py         # Trie-based constrained decoding
│   ├── models.py                   # Model wrappers
│   ├── parse_data.py               # LongBEL input formatting
│   ├── trie.py                     # Prefix trie for constrained decoding
│   └── utils.py                    # General utilities
├── data/
├── figures/
├── pyproject.toml
└── README.md
```

## Training Hyperparameters

The main decoder-only LongBEL models are trained with the following setup.

| Hyperparameter           |                                         Value |
| ------------------------ | --------------------------------------------: |
| Base model               |                         Llama-3.1-8B-Instruct |
| Epochs                   |                                            50 |
| Learning rate            |                                          3e-5 |
| Scheduler                |                                        Linear |
| Warmup ratio             |                                          0.03 |
| Precision                |                                          BF16 |
| Attention implementation |                             Flash Attention 2 |
| Optimizer                |                                         AdamW |
| Maximum sequence length  | Longest tokenized input in the training split |
| Batch size               |                `16384 // max_sequence_length` |
| Gradient accumulation    |                                             1 |

At inference time, we use beam search with 5 beams and BF16 precision.

## Scores

Entity linking performance is reported using Recall@1 with bootstrap confidence intervals. The best result is shown in **bold**, and the second-best result is <u>underlined</u>.

| Model | MM-ST21pv<br>(English) | QUAERO-EMEA<br>(French) | SympTEMIST<br>(Spanish) | DisTEMIST<br>(Spanish) | MedProcNER<br>(Spanish) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Context-Free BEL** ||||| |
| SciSpacy | 53.8 ± 1.0 | 37.1 ± 4.3 | 9.8 ± 1.3 | 21.1 ± 1.9 | 10.3 ± 1.2 |
| SapBERT | 65.6 ± 1.0 | 59.7 ± 3.8 | 34.2 ± 2.0 | 38.6 ± 2.6 | 30.4 ± 2.1 |
| CODER-all | 62.9 ± 1.1 | 66.9 ± 4.0 | 42.2 ± 2.2 | 47.0 ± 2.6 | 42.7 ± 2.1 |
| SapBERT-all | 64.6 ± 1.1 | 67.9 ± 3.9 | 49.8 ± 2.4 | 49.6 ± 2.6 | 45.1 ± 2.2 |
| BERGAMOT | 60.9 ± 1.1 | 63.8 ± 4.9 | 48.0 ± 2.7 | 48.9 ± 2.4 | 42.3 ± 2.2 |
| **Local-Context BEL** ||||| |
| ArboEL | 76.9 ± 0.9 | 63.0 ± 3.9 | 55.4 ± 2.5 | 54.7 ± 2.6 | 59.7 ± 2.6 |
| GENRE / mBART-large | 69.6 ± 1.0 | 69.3 ± 5.4 | 59.8 ± 2.7 | 58.7 ± 2.7 | 66.0 ± 2.3 |
| GENRE / Llama-1B | 73.1 ± 1.0 | 75.1 ± 3.6 | 60.5 ± 2.4 | 62.5 ± 2.3 | 67.4 ± 2.1 |
| GENRE / Llama-8B | 75.0 ± 0.9 | 73.8 ± 4.0 | 61.7 ± 2.5 | 63.2 ± 2.5 | 68.3 ± 2.2 |
| **Global-Context BEL (LongBEL)** ||||| |
| LongBEL-1B | 77.6 ± 0.9 | 74.5 ± 3.7 | 59.8 ± 2.5 | 61.9 ± 2.4 | 66.6 ± 2.1 |
| LongBEL-1B + Ensemble | 78.6 ± 0.8 | <u>77.2 ± 3.0</u> | 61.8 ± 2.5 | <u>64.3 ± 2.2</u> | 69.0 ± 2.0 |
| LongBEL-8B | <u>79.3 ± 0.8</u> | 74.9 ± 4.0 | <u>62.0 ± 2.6</u> | 63.6 ± 2.1 | <u>69.0 ± 2.1</u> |
| LongBEL-8B + Ensemble | **80.0 ± 0.8** | **77.6 ± 3.0** | **63.3 ± 2.5** | **65.8 ± 2.2** | **71.0 ± 2.0** |


## Computing Resources

Several steps require GPU resources. The provided Slurm scripts are configured for our cluster setup and may need to be adapted to your environment.

You may need to change:

* partition name,
* GPU type,
* number of GPUs,
* memory,
* wall time,
* paths to UMLS / SNOMED resources.

## Ethical Considerations

LongBEL is trained and evaluated on biomedical entity linking datasets and knowledge bases used under their respective licenses. No new patient data are collected by this repository.

The model is intended for research in biomedical entity linking. It should not be used as a standalone clinical decision-making system. Predictions should be validated before any downstream biomedical or clinical use.

## Acknowledgments

* **Constrained decoding**: inspired by autoregressive entity retrieval and adapted for biomedical entity linking.
* **Computing resources**: experiments were conducted using HPC resources from GENCI–IDRIS.
* **Collaborations**: this work was supported by LIMICS and collaborators from the Barcelona Supercomputing Center.

## License

MIT License. See [LICENSE](LICENSE) for details.
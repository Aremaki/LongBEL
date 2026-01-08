# SynCABEL: Synthetic Contextualized Augmentation for Biomedical Entity Linking

<div align="center">
    <img src="figures/overall.png" alt="SynCABEL">
    <p align="center">
<!-- <a href="https://doi.org/10.5281/zenodo.13838918"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13838918.svg" alt="DOI"></a> -->
<a href="https://github.com/astral-sh/uv" target="_blank">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"
    alt="UV">
</a>
<a href="https://github.com/astral-sh/ruff" target="_blank">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json"
    alt="Ruff">
</a>
<a href="https://github.com/dmis-lab/ANGEL/blob/main/LICENSE">
   <img alt="GitHub" src="https://img.shields.io/badge/license-MIT-blue">
</a>
</p>

<h3>
    <a href="https://huggingface.co/collections/Aremaki/syncabel">🤗 SynCABEL HuggingFace Collection</a>
</h3>
</div>

## Introduction

**SynCABEL** is a novel framework designed to enhance generative biomedical entity linking (BEL) by leveraging Large Language Models (LLMs) to generate synthetic, contextualized training data for every concept in a target knowledge base (KB).

This repository contains a complete pipeline for:
- **Synthetic Data Generation**: Scripts and prompts to build custom synthetic datasets covering your entire KB.
- **Data Preprocessing**: Tools to convert BigBio datasets and select optimal concept synonyms using TF-IDF.
- **Fine-tuning**: Training scripts for decoder-only and seq2seq models.
- **Advanced Evaluation**: A novel **LLM-as-a-judge** protocol that assesses semantic relationships (equivalent, broader, narrower) between predictions and gold concepts, going beyond simple exact code matching.

## Direct Use

### Generated Synthetic Datasets

We constructed synthetic datasets for three corpora: MedMentions-ST21pv (English), QUAERO (French) and SPACCC (Spanish).

🤗 [Aremaki/SynCABEL](https://huggingface.co/datasets/Aremaki/SynCABEL)

| Dataset | Language | # Generated Examples | # Concepts in KB | KB Source |
| :--- | :---: | :---: | :---: | :--- |
| **SynthMM** | English | ~766k | ~2.4M | UMLS 2017AA |
| **SynthQUAERO** | French | ~658k | ~3.0M | UMLS 2014AB |
| **SynthSPACCC** | Spanish | TBD | ~257k | SPACCC/UMLS |

### Fine-tuned Models

Checkpoints are available of our best performing model: **Llama-3-8B** fine-tuned on MM-ST21pv, QUAERO-EMEA, QUAERO-MEDLINE, SPACCC:

🤗 [Aremaki/SynCABEL_MedMentions_st21pv](https://huggingface.co/Aremaki/SynCABEL_MedMentions_st21pv) \
🤗 [Aremaki/SynCABEL_SPACCC](https://huggingface.co/Aremaki/SynCABEL_SPACCC) \
🤗 [Aremaki/SynCABEL_QUAERO_EMEA](https://huggingface.co/Aremaki/SynCABEL_QUAERO_EMEA) \
🤗 [Aremaki/SynCABEL_QUAERO_MEDLINE](https://huggingface.co/Aremaki/SynCABEL_QUAERO_MEDLINE)

#### Loading
```python
import torch
from transformers import AutoModelForCausalLM

# Load the model (requires trust_remote_code for custom architecture)
model = AutoModelForCausalLM.from_pretrained(
    "Aremaki/SynCABEL_MedMentions_st21pv",
    trust_remote_code=True,
    device_map="auto"
)
```

### Inference
```python
# The input must follow this format
sentences = [
    "[Ibuprofen]{Chemicals & Drugs} is a non-steroidal anti-inflammatory drug",
    "[Myocardial infarction]{Disorder} requires immediate intervention"
]

results = model.sample(
    sentences=sentences,
    constrained=True, # With or without guided inference
    num_beams=3,
)

for i, beam_results in enumerate(results):
    print(f"Input: {sentences[i]}")

    mention = beam_results[0]["mention"]
    print(f"Mention: {mention}")

    for j, result in enumerate(beam_results):
        print(
            f"Beam {j+1}"
            f"Predicted concept name:{result['pred_concept_name']}"
            f"Predicted code: {result['pred_concept_code']} "
            f"Beam score: {result['beam_score']:.3f})"
        )
        
```

## Customize your own

### Requirements

1. **Install uv package manager**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. **Clone the repository**:
```bash
git clone https://github.com/Aremaki/SynCABEL.git
cd SynCABEL
```
3. **Create virtual environment and install dependencies**:

```bash
uv venv --python 3.9
source .venv/bin/activate
uv sync
```

### Step by step pipeline

```bash
# Step 1: Preprocess UMLS data
uv run python scripts/1a_preprocess_UMLS/run_extract_and_prepare_umls.py

# Step 2: Generate synthetic data
# (See scripts/2_generate_synthetic_data/README.md for details)
uv run python scripts/2_generate_synthetic_data/prepare_concepts.py
uv run bash scripts/2_generate_synthetic_data/run_generate.sh

# Step 3: Prepare final training data
uv run python scripts/3a_prepare_data_seq2seq/run.py # For seq2seq
uv run python scripts/3b_prepare_data_encoder/run.py # For encoder

# Step 4: Train model
uv run python scripts/4b_training_decoder/train.py # Example for decoder

# Step 5: Run inference
uv run python scripts/5_inference/infer.py

# Step 6-8: Evaluate results
uv run python scripts/6_evaluate_syncabel/evaluate.py
uv run python scripts/7_evaluate_scispacy.py
uv run python scripts/8_evaluate_embedding.py
```

## Project Structure

```
SynCABEL/
├── scripts/                    # Training and evaluation pipeline
│   ├── 1a_preprocess_UMLS/    # UMLS data extraction & prep
│   ├── 1b_preprocess_SPACCC/  # SPACCC specific prep
│   ├── 2_generate_synthetic_data/  # LLM augmentation
│   ├── 3a_prepare_data_seq2seq/    # Data prep for seq2seq
│   ├── 3b_prepare_data_encoder/    # Data prep for encoders
│   ├── 4b_training_decoder/   # Training decoder models
│   ├── 5_inference/           # Inference scripts
│   ├── 6_evaluate_syncabel/   # Evaluation logic
│   ├── 7_evaluate_scispacy.py # Baseline eval
│   └── 8_evaluate_embedding.py # Embedding analyses
├── syncabel/                   # Main package
│   ├── utils.py               # Helper functions
│   ├── guided_inference.py    # Guided inference logic
│   ├── models.py              # Model definitions
│   ├── parse_data.py          # Data parsing utilities
│   └── trie.py                # Trie data structure
├── arboEL/                     # Submodule/dependency
├── data/                       # Datasets
├── notebooks/                  # Analysis notebooks
├── pyproject.toml             # Dependencies
└── README.md                  # This file
```

## Scores

Entity linking performance (Recall@1) on biomedical benchmarks. The best results are shown in **bold**, the second-best results are <u>underlined</u>, and the "Average" column reports the mean score across the four benchmarks.

| Model | MM-ST21PV<br>(english) | QUAERO-MEDLINE<br>(french) | QUAERO-EMEA<br>(french) | SPACCC<br>(spanish) | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SciSpacy | 53.8 | 40.5 | 37.1 | 13.2 | 36.2 |
| SapBERT | 51.1 | 50.6 | 49.8 | 33.9 | 46.4 |
| CODER-all | 56.6 | 58.7 | 58.1 | 43.7 | 54.3 |
| SapBERT-all | 64.6 | 74.7 | 67.9 | 47.9 | 63.8 |
| ArboEL | <u>74.5</u> | 70.9 | 62.8 | 49.0 | 64.2 |
| mBART-large | 65.5 | 61.5 | 58.6 | 57.7 | 60.8 |
| + Guided inference | 70.0 | 72.8 | 71.1 | 61.8 | 68.9 |
| **+ SynCABEL (Our method)** | 71.5 | 77.1 | <u>75.3</u> | 64.0 | 72.0 |
| Llama-3-8B | 69.0 | 66.4 | 65.5 | 59.9 | 65.2 |
| + Guided inference | 74.4 | <u>77.5</u> | 72.9 | <u>64.2</u> | <u>72.3</u> |
| **+ SynCABEL (Our method)** | **75.4** | **79.7** | **79.0** | **67.0** | **75.3** |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Issues and pull requests welcome!

# Citation
```
@unpublished{syncabel,
author = {Adam Remaki and Christel Gérardin and Eulàlia Farré-Maduell and Martin Krallinger and Xavier Tannier},
title = {SynCABEL: Synthetic Contextualized Augmentation for Biomedical Entity Linking},
note = {Manuscript submitted for publication},
year = {2026}
}
```
# Prepare data – BigBio ➜ Pickle Builder

This CLI prepares source/target sequence pickle files from selected BigBio biomedical entity linking datasets (MedMentions, QUAERO EMEA, QUAERO MEDLINE). It can also process synthetic training data you generated earlier (SynthMM / SynthQUAERO) as standalone datasets.

The script processes each entity in the input datasets to create a pair of sequences:
- **Source Sequence**: The sentence containing the entity, with the entity mention wrapped in special markers (e.g., `[` and `]`).
- **Target Sequence**: A string in the format `"<entity> is <annotation>"`, where `<annotation>` is the best available synonym for the entity's CUI, determined either by Levenshtein distance or embedding similarity.

For example, for the entity "Pseudomonas aeruginosa" in the sentence "DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection in cystic fibrosis", the script could generate:
- **Source**: "DCTN4 as a modifier of chronic [Pseudomonas aeruginosa] infection in cystic fibrosis"
- **Target**: "Pseudomonas aeruginosa is Pseudomonas aeruginosa" (if no better synonym is found)

## Quick Start
Run with default datasets (MedMentions, EMEA, MEDLINE) and embedding-based synonym selection:
```bash
python scripts/3_prepare_data/run.py
```

Limit to a subset of datasets:
```bash
python scripts/3_prepare_data/run.py --datasets MedMentions EMEA
```

Add custom entity markers:
```bash
python scripts/3_prepare_data/run.py --start-entity « --end-entity »
```

Switch to a different synonym selection method:
```bash
python scripts/3_prepare_data/run.py --selection-method levenshtein
# Or
python scripts/3_prepare_data/run.py --selection-method tfidf
```

## Annotation Selection

You can choose how the script selects the best synonym for the target sequence:
- **`embedding` (default)**: For each mention, this method finds the synonym with the highest cosine similarity. It precomputes embeddings for all mentions and synonyms in a dataset and caches the results in a `best_synonyms.parquet` file within the dataset's output directory. This is generally more accurate but computationally intensive on the first run.
- **`levenshtein`**: This method selects the synonym with the smallest Levenshtein (edit) distance to the mention text. It's faster but may be less semantically accurate.
- **`tfidf`**: This method uses TF-IDF vectorization to find the most similar synonym based on term frequency. It requires a pre-trained TF-IDF vectorizer model.

## CLI Arguments
| Option | Default | Description |
|--------|---------|-------------|
| `--datasets` | `['MedMentions', 'EMEA', 'MEDLINE']` | List of BigBio datasets to process. |
| `--start-entity` | `[` | Opening marker for entity mentions in the source text. |
| `--end-entity` | `]` | Closing marker for entity mentions in the source text. |
| `--selection-method` | `embedding` | How to pick the best annotation: `levenshtein`, `embedding`, or `tfidf`. |
| `--encoder-name` | `encoder/coder-all` | Sentence Transformer model for embedding-based selection. |
| `--tfidf-vectorizer-path` | `encoder/umls_tfidf_vectorizer.joblib` | Path to the pre-trained TF-IDF vectorizer model. |
| `--synth-mm-path` | `data/synthetic_data/SynthMM/SynthMM_bigbio.json` | Path to synthetic MedMentions-style data. |
| `--synth-quaero-path`| `data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio.json` | Path to synthetic QUAERO-style data. |
| `--umls-mm-parquet` | `data/UMLS_processed/MM/all_disambiguated.parquet` | Path to UMLS concepts for MedMentions. |
| `--umls-quaero-parquet`| `data/UMLS_processed/QUAERO/all_disambiguated.parquet` | Path to UMLS concepts for QUAERO. |
| `--out-root` | `data/final_data` | Root directory for all processed output. |

## Outputs
For each processed dataset (e.g., `MedMentions`), a folder is created under `--out-root` containing separate pickle files for each data split. The filenames now include the selection method used (e.g., `train_embedding_source.pkl`).
```
<out_root>/
├── MedMentions/
│   ├── train_embedding_source.pkl
│   ├── train_embedding_target.pkl
│   ├── validation_embedding_source.pkl
│   ├── validation_embedding_target.pkl
│   ├── test_embedding_source.pkl
│   ├── test_embedding_target.pkl
│   └── best_synonyms.parquet  (cache for embedding selection)
│
├── EMEA/
│   └── ...
│
└── SynthMM/
    ├── train_embedding_source.pkl
    └── train_embedding_target.pkl
```
Each `_source.pkl` file contains a Python list of source strings, and each `_target.pkl` contains a list of target strings.

## Synthetic Data
- If `synth-mm-path` points to an existing file, a `SynthMM` dataset will be created.
- If `synth-quaero-path` points to an existing file, a `SynthQUAERO` dataset will be created.
- These are treated as independent datasets with only a `train` split.

## Verifying Pickle Contents (Optional)
```python
import pickle
from pathlib import Path

p = Path('data/final_data/MedMentions/train_source.pkl')
with p.open('rb') as f:
    sources = pickle.load(f)
print(f"Found {len(sources)} source examples.")
if sources:
    print("First example:", sources[0])
```

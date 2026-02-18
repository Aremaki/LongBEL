# Step 3: Prepare Data for Generative Models

This CLI prepares source/target sentence pairs for training and evaluating generative entity linking models. It processes standard BigBio biomedical entity linking datasets (MedMentions, QUAERO EMEA, QUAERO MEDLINE, SPACCC) and can also process synthetic data generated in previous steps as long as it is in BigBio format.

For each entity mention, the script creates:
- **Source Sequence**: The sentence with the entity mention wrapped in markers (e.g., `DCTN4 as a modifier of chronic [Pseudomonas aeruginosa] infection...`).
- **Target Sequence**: A canonicalized form `"<Entity> is <Synonym>"`, where `<Synonym>` is the best matching term from the ontology (e.g., `Pseudomonas aeruginosa is Pseudomonas aeruginosa`).

## Pipeline Overview
1.  **Load Datasets**: Loads real datasets (MedMentions, EMEA, MEDLINE, SPACCC) and checks for synthetic data (SynthMM, SynthQUAERO, SynthSPACCC).
2.  **Select Synonyms**: Determines the "best synonym" for each entity mention to serve as the target label. This can be done via embedding similarity (most accurate), Levenshtein distance, TF-IDF, or title matching.
3.  **Format Data**: Wraps mentions in the source text and formats the target string.
4.  **Save Output**: Writes `_source.pkl`, `_target.pkl`, and `file_annotations.tsv` files to the output directory.

---

## Usage

### 1. Standard Run
Process the default datasets (MedMentions, EMEA, MEDLINE) using embedding-based synonym selection.

```bash
python scripts/3_prepare_data/run.py
```

Then upload them to the HuggingFace Repository:
```bash
huggingface-cli login
huggingface-cli upload Aremaki/MedMentions data/final_data/MedMentions/bigbio_dataset --repo-type dataset
huggingface-cli upload Aremaki/SPACCC data/final_data/SPACCC/bigbio_dataset --repo-type dataset
huggingface-cli upload Aremaki/EMEA data/final_data/EMEA/bigbio_dataset --repo-type dataset
huggingface-cli upload Aremaki/MEDLINE data/final_data/MEDLINE/bigbio_dataset --repo-type dataset
```


### 2. Custom Selection & Datasets
Process specific datasets and use a different method for synonym selection (e.g., Levenshtein distance).

```bash
python scripts/3_prepare_data/run.py \
  --datasets MedMentions EMEA \
  --selection-method levenshtein
```

### 3. Using Synthetic Data
If you have generated synthetic data in step 2, ensure the paths are correct. The script defaults to looking in `data/synthetic_data/`.

```bash
python scripts/3_prepare_data/run.py \
  --datasets MedMentions \
  --synth-mm-path data/synthetic_data/SynthMM/SynthMM_bigbio_def.json
```
---

## Key Arguments

| Option | Default | Description |
| :--- | :--- | :--- |
| `--datasets` | `['MedMentions', 'EMEA', 'MEDLINE']` | List of datasets to process. Supported: `MedMentions`, `EMEA`, `MEDLINE`, `SPACCC`. |
| `--selection-method` | `embedding` | Method to pick the target synonym: `embedding`, `levenshtein`, `tfidf`, or `title`. |
| `--encoder-name` | `encoder/coder-all` | Sentence Transformer model used for `embedding` selection. |
| `--out-root` | `data/final_data` | Root directory for output files. |
| `--start-entity` / `--end-entity` | `[` / `]` | Markers used to wrap the entity mention in the source text. |

**Annotation Selection Methods:**
*   **`embedding`** (Default): Finds the synonym with highest cosine similarity to the mention using a Sentence Transformer. Most accurate but slower on first run (caches results).
*   **`levenshtein`**: Selects synonym with smallest edit distance. Faster but less semantically aware.
*   **`tfidf`**: Uses TF-IDF vector similarity. Requires a pre-trained vectorizer.
*   **`title`**: Always uses the canonical title of the CUI. Fastest and deterministic.

---

## Outputs

Files are saved in `--out-root` (default `data/final_data/`), organized by dataset.

```text
data/final_data/
├── MedMentions/
│   ├── train_embedding_source.pkl       (List of source strings)
│   ├── train_embedding_target.pkl       (List of target strings)
│   ├── validation_embedding_source.pkl
│   ├── ...
│   └── best_synonyms.parquet            (Cache for embedding selection)
├── SynthMM/
│   ├── train_embedding_source.pkl       (Synthetic training data)
│   └── train_embedding_target.pkl
└── ...
```

### Verifying Output (Python)
You can inspect the generated pickle files to ensure they look correct:

```python
import pickle
from pathlib import Path

source_path = Path('data/final_data/MedMentions/train_embedding_source.pkl')
target_path = Path('data/final_data/MedMentions/train_embedding_target.pkl')

with source_path.open('rb') as f:
    sources = pickle.load(f)
with target_path.open('rb') as f:
    targets = pickle.load(f)

print(f"Example 1 Source: {sources[0]}")
print(f"Example 1 Target: {targets[0]}")
```

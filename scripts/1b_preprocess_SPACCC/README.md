# Step 1b: SPACCC Preprocessing

This directory contains the preprocessing pipeline for the **Spanish Clinical Case Corpus (SPACCC)**. This step is necessary to normalize the Spanish terminology and standardize the dataset format before it can be used for training or evaluation in the SynCABEL pipeline.

## Scripts

### 1. `prepare_terminology.py`
**Purpose**: Cleans and preprocesses the SPACCC terminology dictionary.
-   Reads `terminology.tsv` (raw SNOMED data).
-   Resolves entities with multiple SNOMED codes based on priority heuristics.
-   Outputs a disambiguated terminology file (`all_disambiguated.parquet`) and a corrected code mapping.

### 2. `prepare_corpus.py`
**Purpose**: Converts the dataset splits into the standard BigBio format.
-   Reads the train/test splits from `data/SPACCC/Normalization/` (TSV annotations + Text files).
-   Converts them into JSON Lines format (`SPACCC_train.json`, `SPACCC_test.json`, etc.).
-   **Note**: This script does *not* generate model-specific embeddings or pickle files; that is done in Step 3a.

### 3. `run.sh`
**Purpose**: Orchestrates the entire pipeline by running the above scripts in order.

## Usage

To run the full preprocessing pipeline for SPACCC:

```bash
# Run from the root of the repository
bash scripts/1b_preprocess_SPACCC/run.sh
```

Or run individual steps using `uv`:

```bash
# Step 1: Terminology
uv run scripts/1b_preprocess_SPACCC/prepare_terminology.py

# Step 2: Corpus Conversion
uv run scripts/1b_preprocess_SPACCC/prepare_corpus.py
```

## Outputs

After running this pipeline, you will find:

1.  **Processed Terminology**:
    -   `data/UMLS_processed/SPACCC/all_disambiguated.parquet`
    -   `data/corrected_code/SPACCC_adapted.csv`

2.  **Standardized Dataset (BigBio JSONs)**:
    -   `data/bigbio/SPACCC/SPACCC_train.json`
    -   `data/bigbio/SPACCC/SPACCC_test.json`


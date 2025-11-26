# SPACCC Preprocessing

This directory contains scripts to preprocess the SPACCC dataset, including terminology resolution and corpus preparation for model training.

## Scripts

- `prepare_terminology.py`: This script preprocesses the SPACCC terminology file (`terminology.tsv`). It resolves ambiguous entities where a single term might be associated with multiple CUIs (Concept Unique Identifiers). It creates a disambiguated terminology file and a mapping file for corrected CUIs.
- `prepare_corpus.py`: This script takes the preprocessed terminology and the SPACCC corpus (train/test sets) to generate model-ready training, development, and test sets in a pickled format. It formats the data into source and target files for sequence-to-sequence models.
- `run.sh`: This is a bash script to orchestrate the entire preprocessing pipeline. It runs `prepare_terminology.py` first, followed by `prepare_corpus.py`.

## Usage

To run the complete preprocessing pipeline, execute the `run.sh` script from the root of the repository:

```bash
bash scripts/1b_preprocess_SPACCC/run.sh
```

This will:
1.  Generate a disambiguated terminology file (`all_disambiguated.parquet`) in `data/UMLS_processed/SPACCC/`.
2.  Generate a CUI correction map (`SPACCC_adapted.csv`) in `data/corrected_code/`.
3.  Generate model-specific data files (e.g., `train_tfidf_source.pkl`, `test_tfidf_target.pkl`) in `data/final_data/SPACCC/`.

Make sure the required input files are in place as specified in the scripts, primarily within `data/SPACCC/`.


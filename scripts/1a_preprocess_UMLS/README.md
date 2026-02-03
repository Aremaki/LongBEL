# Step 1a: Preprocess UMLS

This directory contains the scripts to extract specific UMLS releases (RRF files from zip) and prepare dataset-specific synonym files for the LongBEL pipeline.

## What it does
- Extracts UMLS zip archives and converts RRF files to Parquet format.
- Builds dataset-tailored synonym tables used later in training (e.g., for MedMentions or QUAERO).

## Configuration

Default mapping of UMLS release zip to dataset:

-   `UMLS_2014AB.zip` → `QUAERO` (French corpus)
-   `UMLS_2017AA.zip` → `MM` (MedMentions)

## Expected Input Layout

Place the raw UMLS zip files under `data/UMLS_raw/` in the project root:

```
LongBEL/
  data/
    UMLS_raw/
      UMLS_2014AB.zip
      UMLS_2017AA.zip
```

Inside each zip, the script expects exactly one top-level folder named like the release, containing the RRF files (`MRCONSO.RRF`, `MRDEF.RRF`, `MRSTY.RRF`).

## Files

-   **`extract_umls_data.py`**: Extracts the RRF files from the zips and converts them to raw Parquet files (codes, semantic types, titles).
-   **`prepare_umls_data.py`**: Processes the raw Parquet tables to create the final disambiguated synonym lists (`all_disambiguated.parquet`).
-   **`run_extract_and_prepare_umls.py`**: A master runner script that automates the extraction and preparation for all defined releases.

## Usage

From the repository root, run the automation script:

```bash
uv run scripts/1a_preprocess_UMLS/run_extract_and_prepare_umls.py
```

This will process the available zips and output the artifacts to `data/UMLS_processed/<DATASET>/`.

### Advanced Usage

You can also run the individual steps using their Typer CLIs:

**Extract:**
```bash
uv run scripts/1a_preprocess_UMLS/extract_umls_data.py \
    --zip-path data/UMLS_raw/UMLS_2017AA.zip \
    --output-dir data/UMLS_processed/MM
```

**Prepare:**
```bash
uv run scripts/1a_preprocess_UMLS/prepare_umls_data.py \
    --dataset-name MM \
    --input-dir data/UMLS_processed/MM \
    --output-dir data/UMLS_processed/MM
```

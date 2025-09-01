# UMLS Extraction + Preparation (Automation)

This folder contains a small runner that extracts specific UMLS releases and prepares dataset-specific synonym files for the SynCABEL pipeline.

## What it does
- Extracts UMLS zip archives and converts RRF files to parquet.
- Builds dataset-tailored synonym tables used later in training.

Default mapping of zip → dataset:
- `UMLS_2014AB.zip` → `QUAERO`
- `UMLS_2017AA.zip` → `MM`

## Expected input layout
Place the raw UMLS zip files under `data/UMLS_raw/` with this structure:

```
SynCABEL/
  data/
    UMLS_raw/
      UMLS_2014AB.zip
      UMLS_2017AA.zip
```

Inside each zip, we expect exactly one top-level folder named like the release and containing the three files MRCONSO, MRDEF, MRSTY (RRF). For example:

```
UMLS_2014AB.zip
└── UMLS_2014AB/
    ├── MRCONSO.RRF
    ├── MRDEF.RRF
    └── MRSTY.RRF

UMLS_2017AA.zip
└── UMLS_2017AA/
    ├── MRCONSO.RRF
    ├── MRDEF.RRF
    └── MRSTY.RRF
```

## Outputs
For each dataset (`QUAERO`, `MM`):
- Extracted parquet files under `data/UMLS_processed/<DATASET>/`:
  - `codes.parquet`, `semantic.parquet`, `title_syn.parquet`
- Prepared synonym parquets under `data/UMLS_processed/<DATASET>/`:
  - `all_disambiguated.parquet`, `fr_disambiguated.parquet`

## Run
From the repository root:

```bash
python scripts/1_preprocess_UMLS/run_extract_and_prepare_umls.py
```

This will iterate the default releases list and produce the processed artifacts in `data/UMLS_processed/`.

## Notes
- You can edit the `RELEASES` list in `run_extract_and_prepare_umls.py` to add or remove releases or change the dataset mapping.
- The scripts rely on Typer CLIs under the hood (`extract_umls_data.py`, `prepare_umls_data.py`). See `--help` on those files for advanced usage.

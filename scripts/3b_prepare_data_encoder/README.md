# 3b - Prepare data for encoder models

Typer CLI to build dictionary pickles from UMLS and convert BigBio datasets (and synthetic JSONs) into BLINK-style JSONL files for encoder-based entity linking.

## What it does
- Builds per-ontology dictionaries from processed UMLS parquet files:
  - Writes `umls_info_encoder.pkl` alongside UMLS data
  - Writes per-dataset `dictionary.pickle` under the output root
- Converts datasets to JSONL with fields: `mention`, `mention_id`, `context_left`, `context_right`, `context_doc_id`, `type`, `label_id`, `label`, `label_title`.
  - HF datasets: MedMentions, QUAERO EMEA, QUAERO MEDLINE (train/validation/test if available)
  - Augmented datasets: Combines original validation/test with augmented training data (original + synthetic)
    - `MedMentions_augmented`: MedMentions val/test + (MedMentions train + SynthMM)
    - `EMEA_augmented`: EMEA val/test + (EMEA train + SynthQUAERO)  
    - `MEDLINE_augmented`: MEDLINE val/test + (MEDLINE train + SynthQUAERO)

## Inputs
- Processed UMLS folders (each must contain `all_disambiguated.parquet`):
  - MM: `data/UMLS_processed/MM`
  - QUAERO: `data/UMLS_processed/QUAERO`
- Semantic info parquet: located next to those folders as `data/UMLS_processed/semantic_info.parquet` (the script reads it via `umls_path.parent`).
- Optional synthetic BigBio JSON files:
  - `data/synthetic_data/SynthMM/SynthMM_bigbio.json`
  - `data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio.json`

## Outputs
Under `--out-root` (default: `data/final_data_encoder/`):
- `MedMentions/{train,validation,test}.jsonl` and `dictionary.pickle`
- `EMEA/{train,validation,test}.jsonl` and `dictionary.pickle`
- `MEDLINE/{train,validation,test}.jsonl` and `dictionary.pickle`
- `SynthMM/train.jsonl` and `dictionary.pickle`
- `SynthQUAERO/train.jsonl` and `dictionary.pickle`

Also, UMLS-side cache files:
- `data/UMLS_processed/MM/umls_info_encoder.pkl`
- `data/UMLS_processed/QUAERO/umls_info_encoder.pkl`

## Usage
Run the CLI with Python:

Default end-to-end run (build dictionaries and process all datasets):

```bash
python scripts/3b_prepare_data_encoder/run.py
```

Select specific datasets:

```bash
python scripts/3b_prepare_data_encoder/run.py \
  --datasets MedMentions EMEA MEDLINE
```

Customize paths:

```bash
python scripts/3b_prepare_data_encoder/run.py \
  --umls-mm-path data/UMLS_processed/MM \
  --umls-quaero-path data/UMLS_processed/QUAERO \
  --out-root data/final_data_encoder \
  --synth-mm-json data/synthetic_data/SynthMM/SynthMM_bigbio.json \
  --synth-quaero-json data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio.json
```

## Options
- `--datasets`: Any of `MedMentions`, `EMEA`, `MEDLINE`, `SynthMM`, `SynthQUAERO`.
- `--umls-mm-path`: Folder containing MM `all_disambiguated.parquet`.
- `--umls-quaero-path`: Folder containing QUAERO `all_disambiguated.parquet`.
- `--out-root`: Output root directory.
- `--synth-mm-json`: Path to SynthMM JSON (if processing `SynthMM`).
- `--synth-quaero-json`: Path to SynthQUAERO JSON (if processing `SynthQUAERO`).

## Notes
- The script uses Hugging Face datasets:
  - `bigbio/medmentions: medmentions_st21pv_bigbio_kb`
  - `bigbio/quaero: quaero_emea_bigbio_kb`, `quaero_medline_bigbio_kb`
- Entities without CUIs or CUIs not found in the prepared UMLS info are skipped (with a log message).
- Ensure `polars`, `typer`, and `datasets` are installed according to the project’s `pyproject.toml`.

## Troubleshooting
- Missing `semantic_info.parquet`:
  - Expected at `data/UMLS_processed/semantic_info.parquet`. Make sure the UMLS preprocessing step (scripts/1_preprocess_UMLS) has been run.
- Empty JSONL outputs:
  - Check logs for warnings about unknown groups or missing CUIs.
- HF dataset download/auth issues:
  - Ensure you have internet access and the `datasets` library can download `bigbio` corpora.

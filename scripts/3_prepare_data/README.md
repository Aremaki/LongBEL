# Prepare data – BigBio ➜ Model-Specific Pickle Builder

This CLI prepares source / target sequence pickle files for multiple NLG/seq2seq models from selected BigBio biomedical entity linking datasets (MedMentions, QUAERO EMEA, QUAERO MEDLINE). It can also augment the training set with synthetic sentences you generated earlier (SynthMM / SynthQUAERO).

## Quick Start
Minimal run with all default datasets & models. If required embeddings are missing, they will be generated automatically using CODER (GanjinZero/coder-large-v3). Synthetic JSONs are optional and skipped if absent.
```bash
python scripts/3_prepare_data/run.py run
```

Limit to a subset of datasets and models:
```bash
python scripts/3_prepare_data/run.py run \
  --datasets MedMentions EMEA \
  --models mt5-large bart-large \
  --out-root data/preprocessed_dataset
```

Add custom entity/tag markers (if your model relies on different tokens):
```bash
python scripts/3_prepare_data/run.py run \
  --start-entity « --end-entity » \
  --start-tag <start> --end-tag <end›
```

## Embedding-based annotation selection

Embedding selection is now the default. The script will try to load the following parquet files from `--embeddings-dir` (default `data/embeddings`):
```
umls_synonyms_MM.parquet (for MedMentions)
umls_synonyms_QUAERO.parquet (for EMEA/MEDLINE)
mentions_MedMentions.parquet
mentions_EMEA.parquet
mentions_MEDLINE.parquet
mentions_SynthMM.parquet (if SynthMM JSON exists)
mentions_SynthQUAERO.parquet (if SynthQUAERO JSON exists)
```
If any are missing, they will be generated automatically before processing.

If you prefer to precompute embeddings manually, you can still use the helper script:
```bash
python scripts/3_prepare_data/generate_embeddings.py run \
  --out-dir data/embeddings \
  --coder-model GanjinZero/coder-large-v3 \
  --include-datasets MedMentions EMEA MEDLINE \
  --include-synth true
```
This writes parquet files like:
```
data/embeddings/
  umls_synonyms_MM.parquet
  umls_synonyms_QUAERO.parquet
  mentions_MedMentions.parquet
  mentions_EMEA.parquet
  mentions_MEDLINE.parquet
  mentions_SynthMM.parquet
  mentions_SynthQUAERO.parquet
```

Run prepare with embedding selection explicitly (optional, as it's the default):
```bash
python scripts/3_prepare_data/run.py run \
  --selection-method embedding \
  --embeddings-dir data/embeddings
```
To switch back to edit-distance selection:
```bash
python scripts/3_prepare_data/run.py run --selection-method levenshtein
```

## CLI Arguments (with defaults)
| Option | Default | Description |
|--------|---------|-------------|
| `--datasets` | MedMentions EMEA MEDLINE | Subset to process. |
| `--models` | (all supported) | Provide a subset of supported models (see list below). |
| `--start-entity` | `[` | Opening marker inserted around entity surface forms. |
| `--end-entity` | `]` | Closing marker for entity surface forms. |
| `--start-tag` | `{` | Opening marker for concept / tag tokens. |
| `--end-tag` | `}` | Closing marker for concept / tag tokens. |
| `--selection-method` | `embedding` | How to pick best annotation among synonyms: `levenshtein` or `embedding`. |
| `--embeddings-dir` | `data/embeddings` | Directory containing or storing embeddings. Missing files will be generated. |
| `--coder-model` | `GanjinZero/coder-large-v3` | Encoder used when generating embeddings automatically. |
| `--synth-mm-path` | `data/synthetic_data/SynthMM/SynthMM_bigbio.json` | Synthetic MedMentions-style BigBio JSON. If missing: skipped. |
| `--synth-quaero-path` | `data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio.json` | Synthetic QUAERO-style BigBio JSON. If missing: skipped. |
| `--umls-mm-parquet` | `data/UMLS_processed/MM/all_disambiguated.parquet` | UMLS MedMentions concept parquet (for synonym mapping). |
| `--umls-quaero-parquet` | `data/UMLS_processed/QUAERO/all_disambiguated.parquet` | UMLS QUAERO concept parquet. |
| `--out-root` | `data/final_data` | Output root folder. |

## Supported Models (DEFAULT_MODELS)
```
mt5-large
bart-large
biobart-v2-large
bart-genre
mbart-large-50
```
If you pass `--models` you must list names drawn from this set.

## Outputs
For each requested dataset (e.g., `MedMentions`) a folder is created under `--out-root`:
```
<data_root>/MedMentions/
  train_source_<model>.pkl
  train_target_<model>.pkl
  validation_source_<model>.pkl   (if split exists)
  validation_target_<model>.pkl   (if split exists)
  test_source_<model>.pkl         (if split exists)
  test_target_<model>.pkl         (if split exists)
  synth_train_source_<model>.pkl  (only if synthetic JSON was loaded)
  synth_train_target_<model>.pkl  (only if synthetic JSON was loaded)
```
Each pickle contains a Python list of strings (sources or targets) already formatted with the specified markers and any model-specific adjustments applied by `process_bigbio_dataset`.

When `--selection-method embedding` is used, target strings are built with the synonym that has maximum cosine similarity to the mention among the synonyms of the same CUI; otherwise the closest edit-distance synonym is used.

## Synthetic Augmentation Logic
- If `SynthMM.json` exists: its examples are processed once per model and stored as `synth_train_*` pickles for MedMentions.
- If `SynthQUAERO.json` exists: used for EMEA + MEDLINE.
- Absence of either file triggers an info warning and silent skip.

## Custom Markers Rationale
Markers let the downstream model distinguish:
- `[ ... ]` spans: actual entity surface form occurrences.
- `{ ... }` tokens: normalization tags / CUI bracket tokens (exact semantics depend on `process_bigbio_dataset`).
Feel free to choose Unicode or rare tokens to minimize collisions with natural text.

## Example: Full Custom Run
```bash
python scripts/3_prepare_data/run.py run \
  --datasets MedMentions MEDLINE \
  --models mt5-large mbart-large-50 \
  --start-entity <e> --end-entity </e> \
  --start-tag <cui> --end-tag </cui> \
  --synth-mm-path data/bigbio_datasets/SynthMM.json \
  --synth-quaero-path data/bigbio_datasets/SynthQUAERO.json \
  --out-root data/preprocessed_dataset_custom
```

## Verifying Pickle Contents (Optional)
```python
import pickle, pathlib
p = pathlib.Path('data/preprocessed_dataset/MedMentions/train_source_mt5-large.pkl')
with p.open('rb') as f:
    sources = pickle.load(f)
print(len(sources), sources[0][:300])
```

## Tokenizer / Model Loading
The preprocessing now attempts to download tokenizers directly from the HuggingFace Hub using each `--models` value as a repo ID (e.g. `mt5-large`). If hub download fails (offline cluster, private model, throttling), it falls back to a local directory `models/<model_name>` if present. Ensure your environment has internet access or pre-download models via `transformers-cli` / `huggingface-cli` or by running a simple `from_pretrained` call beforehand.

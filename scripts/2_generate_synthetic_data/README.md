# Synthetic Data Generation CLI Guide

This folder provides a modular CLI pipeline to create synthetic biomedical context sentences for UMLS concepts and export them in BigBio JSON format.

## Overview Pipeline
1. (Optional) Prepare UMLS concept synonym & definition parquets (already produced in step 1).
2. Build per‑CUI user prompts (chunked) via `prepare_concepts.py`.
3. Generate synthetic sentences for each chunk with `generate.py`.
4. Convert generated parquet outputs to BigBio JSON with `convert_to_bigbio.py` (or let `generate.py` write JSON directly).
5. (Optional) Aggregate / convert remote or directory sources to BigBio JSON.

---
## 1. Prepare Concept User Prompts
Create chunked parquet files (`sample_{i}.parquet`) containing columns: `CUI`, `user_prompt`.

```bash
python scripts/2_generate_synthetic_data/prepare_concepts.py \
  --mm-path data/UMLS_processed/MM/all_disambiguated.parquet \
  --quaero-path data/UMLS_processed/QUAERO/all_disambiguated.parquet \
  --spaccc-path data/UMLS_processed/SPACCC/all_disambiguated.parquet \
  --spaccc-umls-path  data/UMLS_processed/SPACCC_UMLS/all_disambiguated.parquet \
  --spaccc-umls-def data/UMLS_processed/SPACCC_UMLS/umls_def.parquet \
  --mm-def data/UMLS_processed/MM/umls_def.parquet \
  --quaero-def data/UMLS_processed/QUAERO/umls_def.parquet \
  --chunk-size 2500
```
Options:
- `--shuffle / --no-shuffle` shuffle concepts before chunking.

Outputs: `data/user_prompts_MM/sample_0.parquet`, `sample_2500.parquet`, ...

---
## 2. Generate Synthetic Sentences
Generates a parquet (`CUI`, `llm_output`) per‑chunk parquets into separate subfolders.

```bash
bash scripts/2_generate_synthetic_data/run_generate.sh
```

Optional overrides when submitting:
```bash
BATCH_SIZE=2 MAX_NEW_TOKENS=768 MAX_RETRIES=4 bash scripts/2_generate_synthetic_data/generate.sh
```

Prompts used:
- MM: `scripts/2_generate_synthetic_data/prompts/system_prompt_mm.txt`
- QUAERO: `scripts/2_generate_synthetic_data/prompts/system_prompt_quaero.txt`

---
## 3. Convert Parquet(s) to BigBio JSON
`convert_to_bigbio.py` offers multiple commands.

### a. Directory of parquets
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py convert \
  --parquet data/synthetic_data/SynthMM/generated_def \
  --json-out data/synthetic_data/SynthMM/SynthMM_bigbio_def.json
```
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py convert \
  --parquet data/synthetic_data/SynthQUAERO/generated_def \
  --json-out data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio_def.json
```
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py convert \
  --parquet data/synthetic_data/SynthSPACCC/generated_no_def \
  --json-out data/synthetic_data/SynthSPACCC/SynthSPACCC_bigbio_no_def.json
```
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py convert \
  --parquet data/synthetic_data/SynthSPACCC_UMLS/generated_def \
  --json-out data/synthetic_data/SynthSPACCC_UMLS/SynthSPACCC_UMLS_bigbio_def.json
```
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py convert \
  --parquet data/synthetic_data/SynthSPACCC_UMLS/generated_no_def \
  --json-out data/synthetic_data/SynthSPACCC_UMLS/SynthSPACCC_UMLS_bigbio_no_def.json
```
Options: `--limit` (row cap), `--fail-pattern` (default FAIL).

### b. Directly from HuggingFace Hub dataset
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py from-hub \
  --repo-id XXXX/SynCABEL \
  --split SynthMedMentions \
  --json-out data/synthetic_data/SynthMM/SynthMM_bigbio_def.json
```
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py from-hub \
  --repo-id XXXX/SynCABEL \
  --split SynthQUAERO \
  --json-out data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio_def.json
```
Custom columns:
```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py from-hub \
  --repo-id XXXX/SynCABEL \
  --split train \
  --cui-col concept_id \
  --text-col generations \
  --fail-pattern FAIL \
  --limit 20000 \
  --json-out data/bigbio_datasets/Synth_custom.json
```
---
## BigBio Output Structure
Each JSON record includes:
- `id`, `document_id`
- `passages`: single passage with full sentence text
- `entities`: bracketed entities (type `LLM_generated`, normalized to UMLS CUI)
- Empty lists for `events`, `coreferences`, `relations` (placeholder fields)

---
## Tips
- Ensure GPU with enough memory for chosen model & batch size (reduce `--batch-size` if OOM).
- For reproducibility you can later extend scripts to accept a random seed (not yet added here).
- Inspect failures by filtering rows containing `FAIL` in `llm_output` before conversion if needed.

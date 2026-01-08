# Step 2: Synthetic Data Generation

This folder provides a modular CLI pipeline to create synthetic biomedical context sentences for UMLS concepts and export them in BigBio JSON format.

## Pipeline Overview
1. **Prepare User Prompts**: Create "user prompts" for each concept (chunked parquet files) using `prepare_concepts.py`.
2. **Generate Sentences**: Use an LLM to generate synthetic sentences for each prompt using `generate.py` (or `run_generate.sh` for SLURM).
3. **Convert to BigBio**: Convert the generated output to BigBio JSON format using `convert_to_bigbio.py`.

---

## 1. Prepare Concept User Prompts
Create chunked parquet files (e.g., `sample_{i}.parquet`) containing columns `CUI` and `user_prompt` for the LLM.

### Usage
You can run generation for multiple datasets (MM, QUAERO, SPACCC) in one go by providing the necessary arguments for each.

```bash
python scripts/2_generate_synthetic_data/prepare_concepts.py \
  --chunk-size 2500 \
  --mm-train-path data/final_data/MedMentions/train_tfidf_annotations.tsv \
  --mm-path data/UMLS_processed/MM/all_disambiguated.parquet \
  --mm-def data/UMLS_processed/MM/umls_def.parquet \
  --medline-train-path data/final_data/MEDLINE/train_tfidf_annotations.tsv \
  --emea-train-path data/final_data/EMEA/train_tfidf_annotations.tsv \
  --quaero-path data/UMLS_processed/QUAERO/all_disambiguated.parquet \
  --quaero-def data/UMLS_processed/QUAERO/umls_def.parquet \
  --spaccc-train-path data/final_data/SPACCC/train_tfidf_annotations.tsv \
  --spaccc-path data/UMLS_processed/SPACCC/all_disambiguated.parquet \
  --spaccc-def data/UMLS_processed/SPACCC/umls_def.parquet
```

**Common Options:**
- `--chunk-size`: Number of prompts per output parquet file (default: 2500).
- `--shuffle / --no-shuffle`: Shuffle concepts before chunking (default: True).

**Outputs:**
- `data/synthetic_data/SynthMM/user_prompts_def/sample_*.parquet`
- `data/synthetic_data/SynthMM/user_prompts_no_def/sample_*.parquet`
- (And similar folders for QUAERO and SPACCC)

---

## 2. Generate Synthetic Sentences
This step runs the LLM to generate sentences based on the prompts created in Step 1.

### Option A: Run on SLURM Cluster (Recommended for large scale)
Use the provided `run_generate.sh` script to submit jobs for all chunks across all datasets.

```bash
bash scripts/2_generate_synthetic_data/run_generate.sh
```
*Note: This script submits `sbatch` jobs using `generate.slurm`. Ensure your SLURM environment is configured correctly.*

### Option B: Run Locally (Single Chunk)
You can run `generate.py` directly for a specific chunk.

```bash
python scripts/2_generate_synthetic_data/generate.py \
  --chunk 0 \
  --user-prompts-dir data/synthetic_data/SynthMM/user_prompts_def \
  --out-dir data/synthetic_data/SynthMM/generated_def \
  --model-path models/Llama-3.3-70B-Instruct \
  --system-prompt-path scripts/2_generate_synthetic_data/prompts/system_prompt_mm.txt \
  --batch-size 8 \
  --max-new-tokens 512
```

**Key Arguments:**
- `--chunk`: The chunk index (integer) to process.
- `--user-prompts-dir`: Directory containing the `sample_{chunk}.parquet` input.
- `--out-dir`: Directory where the result `sample_{chunk}.parquet` will be saved.
- `--system-prompt-path`: Path to the system prompt text file (choose `system_prompt_mm.txt`, `system_prompt_quaero.txt`, etc.).

---

## 3. Convert Parquet(s) to BigBio JSON
Convert the raw LLM output parquets into the BigBio JSON format.

### Method A: Convert local parquet files
You can convert a single file or a directory of parquets.

```bash
# Example for SynthMM (definitions)
python scripts/2_generate_synthetic_data/convert_to_bigbio.py convert \
  --parquet data/synthetic_data/SynthMM/generated_def \
  --json-out data/synthetic_data/SynthMM/SynthMM_bigbio_def.json
```

Repeat for other datasets/variants as needed (e.g., SynthQUAERO, SynthSPACCC).

### Method B: Convert from HuggingFace Hub
If the dataset is already on the Hub, you can pull and convert it directly.

```bash
python scripts/2_generate_synthetic_data/convert_to_bigbio.py from-hub \
  --repo-id <your-repo-id> \
  --split SynthMedMentions \
  --json-out data/synthetic_data/SynthMM/SynthMM_bigbio_def.json
```
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

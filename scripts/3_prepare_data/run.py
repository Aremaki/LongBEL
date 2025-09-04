"""Prepare model-specific source/target datasets from BigBio corpora (Typer CLI).

For each selected dataset (MedMentions, EMEA, MEDLINE) and model name, creates
pickle files with token-inserted source/target sequences. Optionally includes
synthetic training data (SynthMM / SynthQUAERO JSON) if available.
"""

import json
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from datasets import load_dataset

from syncabel.embeddings import (
    TextEncoder,
    save_embeddings_parquet,
)
from syncabel.embeddings import (
    load_embeddings_parquet as _read_emb_map,
)
from syncabel.parse_data_v2 import process_bigbio_dataset

app = typer.Typer(
    help="Preprocess BigBio datasets into model-specific train/dev/test pickles."
)

DEFAULT_MODELS = [
    "google/mt5-large",
    "facebook/bart-large",
    "GanjinZero/biobart-v2-large",
    "facebook/genre-linking-blink",
    "facebook/mbart-large-50",
]


# --- Embedding helpers -----------------------------------------------------


def _yield_mentions_from_bigbio(ds) -> list[str]:
    mentions: set[str] = set()
    for page in ds:
        for ent in page.get("entities", []):
            if ent.get("text"):
                mentions.add(ent["text"][0])
    return list(mentions)


def _yield_mentions_from_synth(json_path: Path) -> list[str]:
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    mentions: set[str] = set()
    for page in data:
        for ent in page.get("entities", []):
            if ent.get("text"):
                mentions.add(ent["text"][0])
    return list(mentions)


def _ensure_embeddings(
    datasets: list[str],
    out_dir: Path,
    umls_parquet_mm: Path,
    umls_parquet_quaero: Path,
    synth_mm_path: Path,
    synth_quaero_path: Path,
    coder_model: str,
) -> None:
    """Ensure all required embedding parquet files exist; generate missing ones.

    Required files per dataset:
    - Synonyms: umls_synonyms_MM.parquet if MedMentions selected,
                umls_synonyms_QUAERO.parquet if EMEA or MEDLINE selected.
    - Mentions: mentions_{DATASET}.parquet for each selected dataset.
    - Synthetic mentions (optional): mentions_SynthMM.parquet if SynthMM JSON exists,
                                    mentions_SynthQUAERO.parquet if SynthQUAERO JSON exists.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which synonym spaces are needed
    need_mm_syn = "MedMentions" in datasets
    need_quaero_syn = any(d in datasets for d in ["EMEA", "MEDLINE"])

    required_files: list[Path] = []
    if need_mm_syn:
        required_files.append(out_dir / "umls_synonyms_MM.parquet")
    if need_quaero_syn:
        required_files.append(out_dir / "umls_synonyms_QUAERO.parquet")

    # Mentions for datasets
    for ds_name in datasets:
        required_files.append(out_dir / f"mentions_{ds_name}.parquet")

    # Synthetic mention embeddings if JSON files exist
    if synth_mm_path.exists():
        required_files.append(out_dir / "mentions_SynthMM.parquet")
    if synth_quaero_path.exists():
        required_files.append(out_dir / "mentions_SynthQUAERO.parquet")

    missing = [p for p in required_files if not p.exists()]
    if not missing:
        return

    typer.echo("🔎 Some embedding files are missing. Generating them now…")
    encoder = TextEncoder(model_name=coder_model)
    typer.echo("CODER is loaded.")
    # Synonym embeddings
    if need_mm_syn and not (out_dir / "umls_synonyms_MM.parquet").exists():
        if umls_parquet_mm.exists():
            df_mm = pl.read_parquet(umls_parquet_mm)
            syns_mm = df_mm["Entity"].drop_nulls().unique().to_list()
            if syns_mm:
                typer.echo("Encoding UMLS MM synonyms...")
                embs = encoder.encode(syns_mm)
                save_embeddings_parquet(
                    syns_mm, embs, out_dir / "umls_synonyms_MM.parquet"
                )
                typer.echo("UMLS MM synonyms saved.")
        else:
            typer.echo(f"⚠️ Missing UMLS MM parquet: {umls_parquet_mm}")

    if need_quaero_syn and not (out_dir / "umls_synonyms_QUAERO.parquet").exists():
        if umls_parquet_quaero.exists():
            df_q = pl.read_parquet(umls_parquet_quaero)
            syns_q = df_q["Entity"].drop_nulls().unique().to_list()
            if syns_q:
                typer.echo("Encoding UMLS QUAERO synonyms...")
                embs = encoder.encode(syns_q)
                save_embeddings_parquet(
                    syns_q, embs, out_dir / "umls_synonyms_QUAERO.parquet"
                )
                typer.echo("UMLS QUAERO synonyms saved.")
        else:
            typer.echo(f"⚠️ Missing UMLS QUAERO parquet: {umls_parquet_quaero}")

    # Mentions from datasets
    ds_map = {
        "MedMentions": ("bigbio/medmentions", "medmentions_st21pv_bigbio_kb"),
        "EMEA": ("bigbio/quaero", "quaero_emea_bigbio_kb"),
        "MEDLINE": ("bigbio/quaero", "quaero_medline_bigbio_kb"),
    }
    for ds_name in datasets:
        out_path = out_dir / f"mentions_{ds_name}.parquet"
        if out_path.exists() or ds_name not in ds_map:
            continue
        hf_id, cfg = ds_map[ds_name]
        try:
            ds = load_dataset(hf_id, name=cfg)
        except Exception as e:  # pragma: no cover - network/IO variability
            typer.echo(f"⚠️ Could not load dataset {ds_name} ({hf_id}:{cfg}): {e}")
            continue
        all_mentions: list[str] = []
        for split in [k for k in ["train", "validation", "test"] if k in ds]:
            all_mentions += _yield_mentions_from_bigbio(ds[split])
        if all_mentions:
            embs = encoder.encode(all_mentions)
            save_embeddings_parquet(
                all_mentions,
                embs,
                out_path,
                extra_cols={"dataset": [ds_name] * len(all_mentions)},
            )

    # Mentions from synthetic JSONs
    synth_inputs = [
        ("SynthMM", synth_mm_path),
        ("SynthQUAERO", synth_quaero_path),
    ]
    for name, path in synth_inputs:
        out_path = out_dir / f"mentions_{name}.parquet"
        if not path.exists() or out_path.exists():
            continue
        mentions = _yield_mentions_from_synth(path)
        if mentions:
            embs = encoder.encode(mentions)
            save_embeddings_parquet(
                mentions,
                embs,
                out_path,
                extra_cols={"dataset": [name] * len(mentions)},
            )


def _load_syn_embeddings(emb_dir: Path, dataset_name: str):
    # Choose UMLS synonym space based on dataset family
    # MedMentions -> MM, EMEA/MEDLINE -> QUAERO
    tag = "MM" if dataset_name == "MedMentions" else "QUAERO"
    path = emb_dir / f"umls_synonyms_{tag}.parquet"
    if path.exists():
        return _read_emb_map(path)
    return None


def _load_mention_embeddings(emb_dir: Path, name: str):
    path = emb_dir / f"mentions_{name}.parquet"
    if path.exists():
        return _read_emb_map(path)
    return None


def _load_json_if_exists(path: Path):
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _build_mappings(umls_parquet: Path):
    df = pl.read_parquet(umls_parquet)
    syn_df = df.with_columns(Syn=pl.col("Entity").str.split(" of type ").list.get(0))
    cui_to_syn = dict(
        syn_df.group_by("CUI").agg([pl.col("Entity").unique()]).iter_rows()
    )
    return syn_df, cui_to_syn


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _dump(obj, path: Path):
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=-1)


def _iter_selected(options: Optional[Iterable[str]], all_list: list[str]) -> list[str]:
    if not options:
        return all_list
    return [m for m in options if m in all_list]


def _process_single_dataset(
    name: str,
    hf_id: str,
    hf_config: str,
    synth_data,
    syn_df,
    cui_to_syn,
    model_names: list[str],
    start_entity: str,
    end_entity: str,
    start_tag: str,
    end_tag: str,
    out_root: Path,
    selection_method: str,
    emb_dir: Path,
):
    typer.echo(f"→ Loading dataset {hf_id}:{hf_config} ...")
    ds = load_dataset(hf_id, name=hf_config)
    data_folder = out_root / name
    _ensure_dir(data_folder)

    # Determine sentence tokenizer language: MedMentions (English), else French for QUAERO variants.
    language = "english" if name == "MedMentions" else "french"

    # Load embeddings once per dataset
    syn_embeds = _load_syn_embeddings(
        out_root.parent / "embeddings" if not emb_dir else emb_dir, name
    )
    mention_embeds = _load_mention_embeddings(
        out_root.parent / "embeddings" if not emb_dir else emb_dir, name
    )

    for model_name in model_names:
        typer.echo(f"  • Processing model {model_name}")
        if synth_data is not None:
            synth_src, synth_tgt = process_bigbio_dataset(
                synth_data,
                start_entity,
                end_entity,
                start_tag,
                end_tag,
                natural=True,
                CUI_to_Syn=cui_to_syn,
                Syn_to_annotation=syn_df,
                model_name=model_name,
                language=language,
                selection_method=selection_method,
                syn_embeddings=syn_embeds,
                mention_embeddings=_load_mention_embeddings(
                    out_root.parent / "embeddings" if not emb_dir else emb_dir,
                    "SynthMM" if name == "MedMentions" else "SynthQUAERO",
                ),
            )
        else:
            synth_src, synth_tgt = [], []

        # Build splits dict only for existing keys to avoid static type checker complaints
        splits = {"train": ds["train"]}
        if "validation" in ds:
            splits["validation"] = ds["validation"]
        if "test" in ds:
            splits["test"] = ds["test"]
        processed = {}
        for split_name, split_data in splits.items():
            if not split_data:
                continue
            src, tgt = process_bigbio_dataset(
                split_data,
                start_entity,
                end_entity,
                start_tag,
                end_tag,
                natural=True,
                CUI_to_Syn=cui_to_syn,
                Syn_to_annotation=syn_df,
                model_name=model_name,
                language=language,
                selection_method=selection_method,
                syn_embeddings=syn_embeds,
                mention_embeddings=mention_embeds,
            )
            processed[split_name] = (src, tgt)

        # Write outputs
        if synth_src:
            _dump(synth_src, data_folder / f"synth_train_source_{model_name}.pkl")
            _dump(synth_tgt, data_folder / f"synth_train_target_{model_name}.pkl")
        for split_name, (src, tgt) in processed.items():
            _dump(src, data_folder / f"{split_name}_source_{model_name}.pkl")
            _dump(tgt, data_folder / f"{split_name}_target_{model_name}.pkl")


@app.command()
def run(
    datasets: list[str] = typer.Option(
        ["MedMentions", "EMEA", "MEDLINE"],
        help="Datasets to process (subset of MedMentions, EMEA, MEDLINE)",
    ),
    models: list[str] = typer.Option(
        None, help="Subset of model names; defaults to all supported if omitted"
    ),
    start_entity: str = typer.Option("[", help="Start entity marker"),
    end_entity: str = typer.Option("]", help="End entity marker"),
    start_tag: str = typer.Option("{", help="Start tag marker"),
    end_tag: str = typer.Option("}", help="End tag marker"),
    selection_method: str = typer.Option(
        "embedding", help="Annotation selection: 'levenshtein' or 'embedding'"
    ),
    embeddings_dir: Path = typer.Option(
        Path("data/embeddings"), help="Directory with precomputed embeddings"
    ),
    coder_model: str = typer.Option(
        "GanjinZero/coder-all", help="Text encoder model for embeddings"
    ),
    synth_mm_path: Path = typer.Option(
        Path("data/synthetic_data/SynthMM/SynthMM_bigbio.json"),
        help="Synthetic MM JSON",
    ),
    synth_quaero_path: Path = typer.Option(
        Path("data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio.json"),
        help="Synthetic QUAERO JSON",
    ),
    umls_mm_parquet: Path = typer.Option(
        Path("data/UMLS_processed/MM/all_disambiguated.parquet"), help="UMLS MM parquet"
    ),
    umls_quaero_parquet: Path = typer.Option(
        Path("data/UMLS_processed/QUAERO/all_disambiguated.parquet"),
        help="UMLS QUAERO parquet",
    ),
    out_root: Path = typer.Option(
        Path("data/final_data"), help="Root output directory"
    ),
) -> None:
    """Run preprocessing pipeline for selected datasets and models."""
    model_list = _iter_selected(models, DEFAULT_MODELS)
    typer.echo(f"Models: {model_list}")

    # Ensure required embeddings exist (generate missing ones automatically)
    _ensure_embeddings(
        datasets,
        embeddings_dir,
        umls_mm_parquet,
        umls_quaero_parquet,
        synth_mm_path,
        synth_quaero_path,
        coder_model,
    )

    # Load UMLS mapping resources
    syn_mm_df, cui_to_syn_mm = _build_mappings(umls_mm_parquet)
    syn_quaero_df, cui_to_syn_quaero = _build_mappings(umls_quaero_parquet)

    # Synthetic data (optional)
    synth_mm = _load_json_if_exists(synth_mm_path)
    synth_quaero = _load_json_if_exists(synth_quaero_path)
    if synth_mm is None:
        typer.echo(
            "⚠️ SynthMM not found; skipping synthetic augmentation for MedMentions."
        )
    if synth_quaero is None:
        typer.echo(
            "⚠️ SynthQUAERO not found; skipping synthetic augmentation for QUAERO-based datasets."
        )

    # Dispatch per dataset
    if "MedMentions" in datasets:
        _process_single_dataset(
            "MedMentions",
            "bigbio/medmentions",
            "medmentions_st21pv_bigbio_kb",
            synth_mm,
            syn_mm_df,
            cui_to_syn_mm,
            model_list,
            start_entity,
            end_entity,
            start_tag,
            end_tag,
            out_root,
            selection_method,
            embeddings_dir,
        )
    if "EMEA" in datasets:
        _process_single_dataset(
            "EMEA",
            "bigbio/quaero",
            "quaero_emea_bigbio_kb",
            synth_quaero,
            syn_quaero_df,
            cui_to_syn_quaero,
            model_list,
            start_entity,
            end_entity,
            start_tag,
            end_tag,
            out_root,
            selection_method,
            embeddings_dir,
        )
    if "MEDLINE" in datasets:
        _process_single_dataset(
            "MEDLINE",
            "bigbio/quaero",
            "quaero_medline_bigbio_kb",
            synth_quaero,
            syn_quaero_df,
            cui_to_syn_quaero,
            model_list,
            start_entity,
            end_entity,
            start_tag,
            end_tag,
            out_root,
            selection_method,
            embeddings_dir,
        )
    typer.echo("✅ Preprocessing complete.")


if __name__ == "__main__":  # pragma: no cover
    app()

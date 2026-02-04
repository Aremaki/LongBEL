"""Prepare model-specific source/target datasets from BigBio corpora (Typer CLI).

For each selected dataset (MedMentions, EMEA, MEDLINE) and model name, creates
pickle files with token-inserted source/target sequences. Optionally includes
synthetic training data (SynthMM / SynthQUAERO JSON) if available.
"""

import json
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, cast

import polars as pl
import typer
from datasets import load_dataset

from longbel.parse_data import compute_best_synonym_df, process_bigbio_dataset

app = typer.Typer(
    help="Preprocess BigBio datasets into model-specific train/dev/test pickles."
)

# --- Embedding helpers -----------------------------------------------------


def _load_json_if_exists(path: Path):
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _build_mappings(umls_parquet: Path):
    df = pl.read_parquet(umls_parquet)
    cui_to_title = dict(df.group_by("CUI").agg([pl.col("Title").first()]).iter_rows())
    cui_to_syn = dict(df.group_by("CUI").agg([pl.col("Entity").unique()]).iter_rows())
    cui_to_groups = dict(df.group_by("CUI").agg([pl.col("GROUP").unique()]).iter_rows())
    return cui_to_syn, cui_to_title, cui_to_groups


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _dump(obj, path: Path):
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=-1)


def _iter_selected(options: Optional[Iterable[str]], all_list: list[str]) -> list[str]:
    if not options:
        return all_list
    return [m for m in options if m in all_list]


def _compute_or_load_best_syn(
    pages: Iterable[dict],
    CUI_to_Syn: dict,
    encoder_name: str,
    cache_path: Path,
    batch_size: int = 4096,
    corrected_code: Optional[dict] = None,
):
    """Return best_syn DataFrame and mapping, using a parquet cache if present."""
    if cache_path.exists():
        typer.echo(f"  • Loading cached best synonyms: {cache_path}")
        best_syn_df = pl.read_parquet(cache_path)
    else:
        typer.echo("  • Precomputing best synonyms (batched embeddings)…")
        best_syn_df = compute_best_synonym_df(
            cast(Iterable[dict], list(pages)),
            CUI_to_Syn=CUI_to_Syn,
            encoder_name=encoder_name,
            batch_size=batch_size,
            corrected_code=corrected_code,
        )
        best_syn_df.write_parquet(cache_path)
    # Use polars DataFrame -> list of dicts
    best_syn_map = {
        (row["CUI"], row["entity"]): row["best_synonym"]
        for row in best_syn_df.to_dicts()
    }
    return best_syn_df, best_syn_map


def _process_spaccc_bigbio_dataset(
    name: str,
    bigbio_dir: Path,
    cui_to_title,
    cui_to_syn,
    cui_to_groups,
    semantic_info,
    encoder_name: str,
    tfidf_vectorizer_path: Path,
    start_entity: str,
    end_entity: str,
    start_group: str,
    end_group: str,
    out_root: Path,
    selection_method: str,
    corrected_code_path: Optional[Path] = None,
):
    typer.echo(f"→ Loading dataset {name} from {bigbio_dir} ...")
    data_folder = out_root / name
    _ensure_dir(data_folder)

    language = "spanish"

    corrected_code = None
    if corrected_code_path and corrected_code_path.exists():
        typer.echo(f"  • Using corrected code mapping from {corrected_code_path}...")
        corrected_code = {
            str(row[0]): str(row[1])
            for row in pl.read_csv(corrected_code_path).iter_rows()
        }

    # Load splits from JSONs
    # Expecting SPACCC_train.json, SPACCC_test.json, etc.
    splits_files = {
        "train": str(bigbio_dir / f"{name}_train.json"),
        "test": str(bigbio_dir / f"{name}_test.json"),
        "test_simple": str(bigbio_dir / f"{name}_test_simple.json"),
        "test_ner": str(bigbio_dir / f"{name}_test_ner.json"),
    }
    # Filter only existing files
    splits_files = {k: v for k, v in splits_files.items() if Path(v).exists()}

    if not splits_files:
        typer.echo(f"⚠️ No JSON files found in {bigbio_dir} for {name}.")
        return

    ds = load_dataset("json", data_files=splits_files)

    # Precompute best synonyms on this dataset's splits only
    def _iter_pages_all():
        for split_key in splits_files.keys():
            if split_key in ds:
                yield from ds[split_key]  # type: ignore

    best_syn_map = None
    if selection_method == "embedding":
        best_syn_path = data_folder / "best_synonyms.parquet"
        _, best_syn_map = _compute_or_load_best_syn(
            cast(Iterable[dict], list(_iter_pages_all())),
            CUI_to_Syn=cui_to_syn,
            encoder_name=encoder_name,
            cache_path=best_syn_path,
            corrected_code=corrected_code,
        )

    typer.echo(f"Processing dataset {name} ...")
    processed = {}
    for split_name in splits_files.keys():
        if split_name not in ds:
            continue
        split_data = ds[split_name]  # type: ignore
        # ner_mode for test_ner? Original script had ner_mode="ner" in split_name
        ner_mode = "ner" in split_name

        src, tgt, tsv_data = process_bigbio_dataset(
            split_data,  # type: ignore
            start_entity,
            end_entity,
            start_group,
            end_group,
            CUI_to_Title=cui_to_title,
            CUI_to_Syn=cui_to_syn,
            CUI_to_GROUP=cui_to_groups,
            semantic_info=semantic_info,
            encoder_name=encoder_name,
            tfidf_vectorizer_path=tfidf_vectorizer_path,
            corrected_code=corrected_code,
            language=language,
            selection_method=selection_method,
            best_syn_map=best_syn_map,
            ner_mode=ner_mode,
        )
        processed[split_name] = (src, tgt, tsv_data)

    # Write outputs
    for split_name, (src, tgt, tsv_data) in processed.items():
        _dump(
            src,
            data_folder / f"{split_name}_{selection_method}_source.pkl",
        )
        _dump(
            tgt,
            data_folder / f"{split_name}_{selection_method}_target.pkl",
        )
        pl.DataFrame(tsv_data).write_csv(
            file=data_folder / f"{split_name}_{selection_method}_annotations.tsv",
            separator="\t",
            include_header=True,
        )


def _process_hf_dataset(
    name: str,
    hf_id: str,
    hf_config: str,
    cui_to_title,
    cui_to_syn,
    cui_to_groups,
    semantic_info,
    encoder_name: str,
    tfidf_vectorizer_path: Path,
    start_entity: str,
    end_entity: str,
    start_group: str,
    end_group: str,
    out_root: Path,
    selection_method: str,
):
    typer.echo(f"→ Loading dataset {hf_id}:{hf_config} ...")
    ds = load_dataset(hf_id, name=hf_config)
    data_folder = out_root / name
    _ensure_dir(data_folder)

    # Determine sentence tokenizer language: MedMentions (English), else French for QUAERO variants.
    language = "english" if name == "MedMentions" else "french"

    # Precompute best synonyms on this dataset's splits only, cache per dataset
    def _iter_pages_all():
        for split_key in ("train", "validation", "test"):
            if split_key in ds:
                yield from ds[split_key]  # type: ignore

    # Optional: corrected CUI mapping for QUAERO (from manual review)
    corrected_code = None
    if name in ("EMEA", "MEDLINE"):
        corrected_code_path = (
            Path("data") / "corrected_code" / "QUAERO_2014_adapted.csv"
        )
        if corrected_code_path.exists():
            typer.echo("Using corrected CUI mapping...")
            corrected_code = dict(pl.read_csv(corrected_code_path).iter_rows())

    best_syn_map = None
    if selection_method == "embedding":
        best_syn_path = data_folder / "best_synonyms.parquet"
        _, best_syn_map = _compute_or_load_best_syn(
            cast(Iterable[dict], list(_iter_pages_all())),
            CUI_to_Syn=cui_to_syn,
            encoder_name=encoder_name,
            cache_path=best_syn_path,
            corrected_code=corrected_code,
        )
    typer.echo(f"Processing dataset {name} ...")
    # Build splits dict only for existing keys
    splits = {"train": ds["train"]}  # type: ignore
    if "validation" in ds:
        splits["validation"] = ds["validation"]  # type: ignore
    if "test" in ds:
        splits["test"] = ds["test"]  # type: ignore
    processed = {}
    for split_name, split_data in splits.items():
        if not split_data:
            continue
        src, tgt, tsv_data = process_bigbio_dataset(
            split_data,
            start_entity,
            end_entity,
            start_group,
            end_group,
            CUI_to_Title=cui_to_title,
            CUI_to_Syn=cui_to_syn,
            CUI_to_GROUP=cui_to_groups,
            semantic_info=semantic_info,
            encoder_name=encoder_name,
            tfidf_vectorizer_path=tfidf_vectorizer_path,
            corrected_code=corrected_code,
            language=language,
            selection_method=selection_method,
            best_syn_map=best_syn_map,
        )
        processed[split_name] = (src, tgt, tsv_data)

    # Write outputs
    for split_name, (src, tgt, tsv_data) in processed.items():
        _dump(
            src,
            data_folder / f"{split_name}_{selection_method}_source.pkl",
        )
        _dump(
            tgt,
            data_folder / f"{split_name}_{selection_method}_target.pkl",
        )
        pl.DataFrame(tsv_data).write_csv(
            file=data_folder / f"{split_name}_{selection_method}_annotations.tsv",
            separator="\t",
            include_header=True,
        )


def _process_synth_dataset(
    name: str,
    synth_pages: Optional[list[dict]],
    cui_to_title,
    cui_to_syn,
    cui_to_groups,
    semantic_info,
    encoder_name: str,
    tfidf_vectorizer_path: Path,
    start_entity: str,
    end_entity: str,
    start_group: str,
    end_group: str,
    out_root: Path,
    selection_method: str,
):
    """Process a synthetic dataset as an independent dataset (train split)."""
    if not synth_pages:
        typer.echo(f"⚠️ {name}: no synthetic data found; skipping.")
        return
    data_folder = out_root / name
    _ensure_dir(data_folder)

    # Language: assume English for SynthMM, French for SynthQUAERO, Spanish for SynthSPACCC
    language = (
        "english"
        if "MM" in name or "MedMentions" in name
        else ("spanish" if "SPACCC" in name else "french")
    )

    best_syn_map = None
    if selection_method == "embedding":
        best_syn_path = data_folder / "best_synonyms.parquet"
        _, best_syn_map = _compute_or_load_best_syn(
            cast(Iterable[dict], synth_pages),
            CUI_to_Syn=cui_to_syn,
            encoder_name=encoder_name,
            cache_path=best_syn_path,
        )

    typer.echo(f"  • Processing synthetic dataset {name} ...")
    src, tgt, _ = process_bigbio_dataset(
        synth_pages,
        start_entity,
        end_entity,
        start_group,
        end_group,
        CUI_to_Title=cui_to_title,
        CUI_to_Syn=cui_to_syn,
        CUI_to_GROUP=cui_to_groups,
        semantic_info=semantic_info,
        encoder_name=encoder_name,
        tfidf_vectorizer_path=tfidf_vectorizer_path,
        language=language,
        selection_method=selection_method,
        best_syn_map=best_syn_map,
    )
    # Treat as train split for the synthetic dataset
    _dump(src, data_folder / f"train_{selection_method}_source.pkl")
    _dump(tgt, data_folder / f"train_{selection_method}_target.pkl")


@app.command()
def run(
    datasets: list[str] = typer.Option(
        ["MedMentions", "EMEA", "MEDLINE"],
        help="Datasets to process (subset of MedMentions, EMEA, MEDLINE)",
    ),
    start_entity: str = typer.Option("[", help="Start entity marker"),
    end_entity: str = typer.Option("]", help="End entity marker"),
    start_group: str = typer.Option("{", help="Start group marker"),
    end_group: str = typer.Option("}", help="End group marker"),
    selection_method: str = typer.Option(
        "embedding",
        help="Annotation selection: 'levenshtein' or 'embedding' or 'tfidf' or 'title'",
    ),
    encoder_name: str = typer.Option(
        "encoder/coder-all", help="Text encoder model for embeddings"
    ),
    tfidf_vectorizer_path: Path = typer.Option(
        Path("models/encoder/umls_tfidf_vectorizer.joblib"),
        help="TF-IDF vectorizer model path",
    ),
    synth_mm_path: Path = typer.Option(
        Path("data/synthetic_data/SynthMM/SynthMM_bigbio_def.json"),
        help="Synthetic MM JSON",
    ),
    synth_quaero_path: Path = typer.Option(
        Path("data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio_def.json"),
        help="Synthetic QUAERO JSON",
    ),
    synth_spaccc_def_path: Path = typer.Option(
        Path("data/synthetic_data/SynthSPACCC/SynthSPACCC_bigbio_def.json"),
        help="SynthSPACCC definitions JSON",
    ),
    synth_spaccc_no_def_path: Path = typer.Option(
        Path("data/synthetic_data/SynthSPACCC/SynthSPACCC_bigbio_no_def.json"),
        help="SynthSPACCC no definitions JSON",
    ),
    synth_spaccc_filtered_path: Path = typer.Option(
        Path("data/synthetic_data/SynthSPACCC/SynthSPACCC_bigbio_filtered.json"),
        help="SynthSPACCC filtered JSON",
    ),
    umls_mm_parquet: Path = typer.Option(
        Path("data/termino_processed/MM/all_disambiguated.parquet"),
        help="UMLS MM parquet",
    ),
    umls_quaero_parquet: Path = typer.Option(
        Path("data/termino_processed/QUAERO/all_disambiguated.parquet"),
        help="UMLS QUAERO parquet",
    ),
    umls_spaccc_parquet: Path = typer.Option(
        Path("data/termino_processed/SPACCC/all_disambiguated.parquet"),
        help="UMLS SPACCC parquet",
    ),
    out_root: Path = typer.Option(
        Path("data/final_data"), help="Root output directory"
    ),
    semantic_info_mm_parquet: Path = typer.Option(
        Path("data/termino_processed/MM/semantic_info.parquet"),
        help="UMLS semantic info parquet",
    ),
    semantic_info_quaero_parquet: Path = typer.Option(
        Path("data/termino_processed/QUAERO/semantic_info.parquet"),
        help="UMLS semantic info parquet",
    ),
    semantic_info_spaccc_parquet: Path = typer.Option(
        Path("data/termino_processed/SPACCC/semantic_info.parquet"),
        help="UMLS semantic info parquet",
    ),
    spaccc_bigbio_dir: Path = typer.Option(
        Path("data/bigbio/SPACCC"),
        help="Directory containing SPACCC BigBio JSONs",
    ),
    corrected_code_spaccc_path: Path = typer.Option(
        Path("data/corrected_code/SPACCC_adapted.csv"),
        help="Corrected SNOMED mapping file for SPACCC",
    ),
) -> None:
    """Run preprocessing pipeline for selected datasets and models."""
    # Load UMLS mapping resources
    semantic_info_mm = pl.read_parquet(semantic_info_mm_parquet)
    semantic_info_quaero = pl.read_parquet(semantic_info_quaero_parquet)
    semantic_info_spaccc = None
    if semantic_info_spaccc_parquet.exists():
        semantic_info_spaccc = pl.read_parquet(semantic_info_spaccc_parquet)

    cui_to_syn_mm, cui_to_title_mm, cui_to_groups_mm = _build_mappings(umls_mm_parquet)
    cui_to_syn_quaero, cui_to_title_quaero, cui_to_groups_quaero = _build_mappings(
        umls_quaero_parquet
    )
    cui_to_syn_spaccc, cui_to_title_spaccc, cui_to_groups_spaccc = (None, None, None)
    if umls_spaccc_parquet.exists():
        cui_to_syn_spaccc, cui_to_title_spaccc, cui_to_groups_spaccc = _build_mappings(
            umls_spaccc_parquet
        )

    # Synthetic data (optional)
    synth_mm = _load_json_if_exists(synth_mm_path)
    synth_quaero = _load_json_if_exists(synth_quaero_path)
    synth_spaccc_def = _load_json_if_exists(synth_spaccc_def_path)
    synth_spaccc_no_def = _load_json_if_exists(synth_spaccc_no_def_path)
    synth_spaccc_filtered = _load_json_if_exists(synth_spaccc_filtered_path)

    if synth_mm is None:
        typer.echo(
            "⚠️ SynthMM not found; skipping synthetic augmentation for MedMentions."
        )
    if synth_quaero is None:
        typer.echo(
            "⚠️ SynthQUAERO not found; skipping synthetic augmentation for QUAERO-based datasets."
        )

    # Dispatch per dataset
    processed_synth: set[str] = set()
    if "MedMentions" in datasets:
        _process_hf_dataset(
            "MedMentions",
            "bigbio/medmentions",
            "medmentions_st21pv_bigbio_kb",
            cui_to_title_mm,
            cui_to_syn_mm,
            cui_to_groups_mm,
            semantic_info_mm,
            encoder_name,
            tfidf_vectorizer_path,
            start_entity,
            end_entity,
            start_group,
            end_group,
            out_root,
            selection_method,
        )
        # Synthetic MM as its own dataset
        if synth_mm is not None and "SynthMM" not in processed_synth:
            _process_synth_dataset(
                "SynthMM",
                synth_mm,
                cui_to_title_mm,
                cui_to_syn_mm,
                cui_to_groups_mm,
                semantic_info_mm,
                encoder_name,
                tfidf_vectorizer_path,
                start_entity,
                end_entity,
                start_group,
                end_group,
                out_root,
                selection_method,
            )
            processed_synth.add("SynthMM")
    if "EMEA" in datasets:
        _process_hf_dataset(
            "EMEA",
            "bigbio/quaero",
            "quaero_emea_bigbio_kb",
            cui_to_title_quaero,
            cui_to_syn_quaero,
            cui_to_groups_quaero,
            semantic_info_quaero,
            encoder_name,
            tfidf_vectorizer_path,
            start_entity,
            end_entity,
            start_group,
            end_group,
            out_root,
            selection_method,
        )
        if synth_quaero is not None and "SynthQUAERO" not in processed_synth:
            _process_synth_dataset(
                "SynthQUAERO",
                synth_quaero,
                cui_to_title_quaero,
                cui_to_syn_quaero,
                cui_to_groups_quaero,
                semantic_info_quaero,
                encoder_name,
                tfidf_vectorizer_path,
                start_entity,
                end_entity,
                start_group,
                end_group,
                out_root,
                selection_method,
            )
            processed_synth.add("SynthQUAERO")
    if "MEDLINE" in datasets:
        _process_hf_dataset(
            "MEDLINE",
            "bigbio/quaero",
            "quaero_medline_bigbio_kb",
            cui_to_title_quaero,
            cui_to_syn_quaero,
            cui_to_groups_quaero,
            semantic_info_quaero,
            encoder_name,
            tfidf_vectorizer_path,
            start_entity,
            end_entity,
            start_group,
            end_group,
            out_root,
            selection_method,
        )
        if synth_quaero is not None and "SynthQUAERO" not in processed_synth:
            _process_synth_dataset(
                "SynthQUAERO",
                synth_quaero,
                cui_to_title_quaero,
                cui_to_syn_quaero,
                cui_to_groups_quaero,
                semantic_info_quaero,
                encoder_name,
                tfidf_vectorizer_path,
                start_entity,
                end_entity,
                start_group,
                end_group,
                out_root,
                selection_method,
            )
            processed_synth.add("SynthQUAERO")
    if "SPACCC" in datasets and cui_to_syn_spaccc:
        _process_spaccc_bigbio_dataset(
            "SPACCC",
            spaccc_bigbio_dir,
            cui_to_title_spaccc,
            cui_to_syn_spaccc,
            cui_to_groups_spaccc,
            semantic_info_spaccc,
            encoder_name,
            tfidf_vectorizer_path,
            start_entity,
            end_entity,
            start_group,
            end_group,
            out_root,
            selection_method,
            corrected_code_path=corrected_code_spaccc_path,
        )

        # Synthetic SPACCC
        if (
            synth_spaccc_filtered is not None
            and "SynthSPACCC_Filtered" not in processed_synth
        ):
            _process_synth_dataset(
                "SynthSPACCC_Filtered",
                synth_spaccc_filtered,
                cui_to_title_spaccc,
                cui_to_syn_spaccc,
                cui_to_groups_spaccc,
                semantic_info_spaccc,
                encoder_name,
                tfidf_vectorizer_path,
                start_entity,
                end_entity,
                start_group,
                end_group,
                out_root,
                selection_method,
            )
            processed_synth.add("SynthSPACCC_Filtered")

        if synth_spaccc_def is not None and "SynthSPACCC_Def" not in processed_synth:
            _process_synth_dataset(
                "SynthSPACCC_Def",
                synth_spaccc_def,
                cui_to_title_spaccc,
                cui_to_syn_spaccc,
                cui_to_groups_spaccc,
                semantic_info_spaccc,
                encoder_name,
                tfidf_vectorizer_path,
                start_entity,
                end_entity,
                start_group,
                end_group,
                out_root,
                selection_method,
            )
            processed_synth.add("SynthSPACCC_Def")

        if (
            synth_spaccc_no_def is not None
            and "SynthSPACCC_No_Def" not in processed_synth
        ):
            _process_synth_dataset(
                "SynthSPACCC_No_Def",
                synth_spaccc_no_def,
                cui_to_title_spaccc,
                cui_to_syn_spaccc,
                cui_to_groups_spaccc,
                semantic_info_spaccc,
                encoder_name,
                tfidf_vectorizer_path,
                start_entity,
                end_entity,
                start_group,
                end_group,
                out_root,
                selection_method,
            )
            processed_synth.add("SynthSPACCC_No_Def")
    typer.echo("✅ Preprocessing complete.")


if __name__ == "__main__":  # pragma: no cover
    app()

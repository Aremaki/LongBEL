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


def _load_json_if_exists(path: Path):
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _build_mappings(umls_parquet: Path, lang: str):
    df = pl.read_parquet(umls_parquet)
    if "CUI" in df.columns:
        code_col = "CUI"
    elif "SNOMED_code" in df.columns:
        code_col = "SNOMED_code"
    else:
        raise ValueError(
            "Termino parquet must contain either 'CUI' or 'SNOMED_code' column."
        )

    code_to_title = dict(
        df.group_by(code_col).agg([pl.col("Title").first()]).iter_rows()
    )
    code_to_syn_all = dict(
        df.group_by(code_col).agg([pl.col("Entity").unique()]).iter_rows()
    )

    df_lang = df.filter(pl.col("lang") == lang)
    if df_lang.height > 0:
        code_to_syn_lang = dict(
            df_lang.group_by(code_col).agg([pl.col("Entity").unique()]).iter_rows()
        )
        code_to_syn = {
            code: code_to_syn_lang.get(code, code_to_syn_all[code])
            for code in code_to_syn_all
        }
    else:
        code_to_syn = code_to_syn_all

    code_to_group = dict(
        df.group_by(code_col).agg([pl.col("GROUP").unique()]).iter_rows()
    )
    return code_to_syn, code_to_title, code_to_group


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
    code_to_syn: dict,
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
            code_to_syn=code_to_syn,
            encoder_name=encoder_name,
            batch_size=batch_size,
            corrected_code=corrected_code,
        )
        best_syn_df.write_parquet(cache_path)
    # Use polars DataFrame -> list of dicts
    best_syn_map = {
        (row["code"], row["entity"]): row["best_synonym"]
        for row in best_syn_df.to_dicts()
    }
    return best_syn_df, best_syn_map


def _process_hf_dataset(
    name: str,
    hf_id: str,
    lang: str,
    code_to_title,
    code_to_syn,
    code_to_group,
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
    data_dir: Optional[str] = None,
    long_format: bool = False,
):
    typer.echo(f"→ Loading dataset {hf_id}:{data_dir} ...")
    ds = load_dataset(hf_id, data_dir=data_dir)
    data_folder = out_root / name
    _ensure_dir(data_folder)

    # Precompute best synonyms on this dataset's splits only, cache per dataset
    def _iter_pages_all():
        for split_key in ("train", "validation", "test"):
            if split_key in ds:
                yield from ds[split_key]  # type: ignore

    # Optional: corrected code mapping for QUAERO (from manual review)
    corrected_code = None
    if corrected_code_path is not None and corrected_code_path.exists():
        typer.echo(f"  • Using corrected code mapping from {corrected_code_path}...")
        corrected_code = {
            str(row[0]): str(row[1])
            for row in pl.read_csv(corrected_code_path).iter_rows()
        }

    best_syn_map = None
    if selection_method == "embedding":
        best_syn_path = data_folder / "best_synonyms.parquet"
        _, best_syn_map = _compute_or_load_best_syn(
            cast(Iterable[dict], list(_iter_pages_all())),
            code_to_syn=code_to_syn,
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
        src, tgt, tsv_data, bigbio_split = process_bigbio_dataset(
            split_data,
            start_entity,
            end_entity,
            start_group,
            end_group,
            code_to_title=code_to_title,
            code_to_syn=code_to_syn,
            code_to_group=code_to_group,
            semantic_info=semantic_info,
            encoder_name=encoder_name,
            tfidf_vectorizer_path=tfidf_vectorizer_path,
            corrected_code=corrected_code,
            lang=lang,
            selection_method=selection_method,
            best_syn_map=best_syn_map,
            long_format=long_format,
        )
        processed[split_name] = (src, tgt, tsv_data)
        if long_format:
            processed_data_folder = data_folder / "bigbio_dataset/processed_data"
            _ensure_dir(processed_data_folder)
            # Convert HF dataset to Parquet
            bigbio_split.to_parquet(  # type: ignore
                processed_data_folder / f"{split_name}-00000-of-00001.parquet"
            )

    # Write outputs
    for split_name, (src, tgt, tsv_data) in processed.items():
        _dump(
            src,
            data_folder
            / f"{split_name}_{selection_method}_source{'_long' if long_format else ''}.pkl",
        )
        _dump(
            tgt,
            data_folder
            / f"{split_name}_{selection_method}_target{'_long' if long_format else ''}.pkl",
        )
        pl.DataFrame(tsv_data).write_csv(
            file=data_folder
            / f"{split_name}_{selection_method}_annotations{'_long' if long_format else ''}.tsv",
            separator="\t",
            include_header=True,
        )


def _process_synth_dataset(
    name: str,
    synth_pages: Optional[list[dict]],
    lang: str,
    code_to_title,
    code_to_syn,
    code_to_group,
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

    best_syn_map = None
    if selection_method == "embedding":
        best_syn_path = data_folder / "best_synonyms.parquet"
        _, best_syn_map = _compute_or_load_best_syn(
            cast(Iterable[dict], synth_pages),
            code_to_syn=code_to_syn,
            encoder_name=encoder_name,
            cache_path=best_syn_path,
        )

    typer.echo(f"  • Processing synthetic dataset {name} ...")
    src, tgt, _, _ = process_bigbio_dataset(
        synth_pages,
        start_entity,
        end_entity,
        start_group,
        end_group,
        code_to_title=code_to_title,
        code_to_syn=code_to_syn,
        code_to_group=code_to_group,
        semantic_info=semantic_info,
        encoder_name=encoder_name,
        tfidf_vectorizer_path=tfidf_vectorizer_path,
        lang=lang,
        selection_method=selection_method,
        best_syn_map=best_syn_map,
    )
    # Treat as train split for the synthetic dataset
    _dump(src, data_folder / f"train_{selection_method}_source.pkl")
    _dump(tgt, data_folder / f"train_{selection_method}_target.pkl")


@app.command()
def run(
    datasets: list[str] = typer.Option(
        ["MedMentions", "EMEA", "MEDLINE", "SPACCC"],
        help="Datasets to process (subset of MedMentions, EMEA, MEDLINE, SPACCC)",
    ),
    start_entity: str = typer.Option("[", help="Start entity marker"),
    end_entity: str = typer.Option("]", help="End entity marker"),
    start_group: str = typer.Option("{", help="Start group marker"),
    end_group: str = typer.Option("}", help="End group marker"),
    selection_method: str = typer.Option(
        "tfidf",
        help="Annotation selection: 'levenshtein' or 'embedding' or 'tfidf' or 'title'",
    ),
    encoder_name: str = typer.Option(
        "encoder/coder-all", help="Text encoder model for embeddings"
    ),
    tfidf_vectorizer_path: Path = typer.Option(
        Path("models/tfidf_vectorizer.joblib"),
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
        help="SNOMED SPACCC parquet",
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
        help="SNOMED semantic info parquet",
    ),
    corrected_code_quaero_path: Path = typer.Option(
        Path("data/corrected_code/QUAERO_2014_adapted.csv"),
        help="Corrected code mapping file for QUAERO",
    ),
    corrected_code_spaccc_path: Path = typer.Option(
        Path("data/corrected_code/SPACCC_adapted.csv"),
        help="Corrected SNOMED mapping file for SPACCC",
    ),
    long_format: bool = typer.Option(
        False, help="Use long format parsing (with context sentences)"
    ),
) -> None:
    """Run preprocessing pipeline for selected datasets and models."""

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
            "⚠️ SynthQUAERO not found; skipping synthetic augmentation for EMEA and MEDLINE."
        )
    if (
        synth_spaccc_def is None
        and synth_spaccc_no_def is None
        and synth_spaccc_filtered is None
    ):
        typer.echo(
            "⚠️ SynthSPACCC not found; skipping synthetic augmentation for SPACCC."
        )

    # Dispatch per dataset
    if (
        "MedMentions" in datasets
        and umls_mm_parquet.exists()
        and semantic_info_mm_parquet.exists()
    ):
        lang = "en"
        semantic_info_mm = pl.read_parquet(semantic_info_mm_parquet)
        code_to_syn_mm, code_to_title_mm, code_to_group_mm = _build_mappings(
            umls_mm_parquet, lang
        )
        _process_hf_dataset(
            "MedMentions",
            "Aremaki/MedMentions",
            lang,
            code_to_title_mm,
            code_to_syn_mm,
            code_to_group_mm,
            semantic_info_mm,
            encoder_name,
            tfidf_vectorizer_path,
            start_entity,
            end_entity,
            start_group,
            end_group,
            out_root,
            selection_method,
            data_dir="original",
            long_format=long_format,
        )
        # Synthetic MM as its own dataset
        if synth_mm is not None:
            _process_synth_dataset(
                "SynthMM",
                synth_mm,
                lang,
                code_to_title_mm,
                code_to_syn_mm,
                code_to_group_mm,
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
    if "EMEA" in datasets or "MEDLINE" in datasets:
        lang = "fr"
        if umls_quaero_parquet.exists() and semantic_info_quaero_parquet.exists():
            code_to_syn_quaero, code_to_title_quaero, code_to_group_quaero = (
                _build_mappings(umls_quaero_parquet, lang)
            )
            semantic_info_quaero = pl.read_parquet(semantic_info_quaero_parquet)
            if "EMEA" in datasets:
                _process_hf_dataset(
                    "EMEA",
                    "Aremaki/EMEA",
                    lang,
                    code_to_title_quaero,
                    code_to_syn_quaero,
                    code_to_group_quaero,
                    semantic_info_quaero,
                    encoder_name,
                    tfidf_vectorizer_path,
                    start_entity,
                    end_entity,
                    start_group,
                    end_group,
                    out_root,
                    selection_method,
                    corrected_code_path=corrected_code_quaero_path,
                    data_dir="original",
                    long_format=long_format,
                )

            if "MEDLINE" in datasets:
                _process_hf_dataset(
                    "MEDLINE",
                    "Aremaki/MEDLINE",
                    lang,
                    code_to_title_quaero,
                    code_to_syn_quaero,
                    code_to_group_quaero,
                    semantic_info_quaero,
                    encoder_name,
                    tfidf_vectorizer_path,
                    start_entity,
                    end_entity,
                    start_group,
                    end_group,
                    out_root,
                    selection_method,
                    corrected_code_path=corrected_code_quaero_path,
                    data_dir="original",
                    long_format=long_format,
                )
            if synth_quaero is not None:
                _process_synth_dataset(
                    "SynthQUAERO",
                    synth_quaero,
                    lang,
                    code_to_title_quaero,
                    code_to_syn_quaero,
                    code_to_group_quaero,
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
    if (
        "SPACCC" in datasets
        and umls_spaccc_parquet.exists()
        and semantic_info_spaccc_parquet.exists()
    ):
        lang = "es"
        semantic_info_spaccc = pl.read_parquet(semantic_info_spaccc_parquet)
        code_to_syn_spaccc, code_to_title_spaccc, code_to_group_spaccc = (
            _build_mappings(umls_spaccc_parquet, lang)
        )
        _process_hf_dataset(
            "SPACCC",
            "Aremaki/SPACCC",
            lang,
            code_to_title_spaccc,
            code_to_syn_spaccc,
            code_to_group_spaccc,
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
            data_dir="original",
            long_format=long_format,
        )

        # Synthetic SPACCC
        if synth_spaccc_filtered is not None:
            _process_synth_dataset(
                "SynthSPACCC_Filtered",
                synth_spaccc_filtered,
                lang,
                code_to_title_spaccc,
                code_to_syn_spaccc,
                code_to_group_spaccc,
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
        if synth_spaccc_def is not None:
            _process_synth_dataset(
                "SynthSPACCC_Def",
                synth_spaccc_def,
                lang,
                code_to_title_spaccc,
                code_to_syn_spaccc,
                code_to_group_spaccc,
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
        if synth_spaccc_no_def is not None:
            _process_synth_dataset(
                "SynthSPACCC_No_Def",
                synth_spaccc_no_def,
                lang,
                code_to_title_spaccc,
                code_to_syn_spaccc,
                code_to_group_spaccc,
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
    typer.echo("✅ Preprocessing complete.")


if __name__ == "__main__":  # pragma: no cover
    app()

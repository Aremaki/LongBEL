"""Prepare model-specific source/target datasets from SPACCC corpora (Typer CLI)."""

import json
import logging
import pickle
from pathlib import Path

import polars as pl
import typer

from syncabel.parse_data import process_bigbio_dataset
from syncabel.utils import load_tsv_as_bigbio

app = typer.Typer(
    help="Preprocess SPACCC dataset into model-specific train/dev/test pickles."
)


def _build_mappings(umls_parquet: Path):
    df = pl.read_parquet(umls_parquet)
    # The 'Title' column might not exist in the SPACCC processed parquet.
    # Let's use 'Entity' as a fallback.
    if "Title" not in df.columns:
        df = df.with_columns(pl.col("Entity").alias("Title"))
    code_to_title = dict(
        df.group_by("SNOMED_code").agg([pl.col("Title").first()]).iter_rows()
    )
    code_to_syn = dict(
        df.group_by("SNOMED_code").agg([pl.col("Entity").unique()]).iter_rows()
    )
    code_to_groups = dict(
        df.group_by("SNOMED_code").agg([pl.col("GROUP").unique()]).iter_rows()
    )
    return code_to_syn, code_to_title, code_to_groups


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _dump(obj, path: Path):
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=-1)


def _load_json_if_exists(path: Path):
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _process_synth_dataset(
    name: str,
    code_to_title,
    code_to_syn,
    code_to_groups,
    semantic_info,
    tfidf_vectorizer_path: Path,
    start_entity: str,
    end_entity: str,
    start_group: str,
    end_group: str,
    out_root: Path,
    synth_data_dir: Path,
    corrected_code_path: Path,
):
    """Process a synthetic dataset already in BigBio format."""
    logging.info(f"→ Loading synthetic dataset {name} ...")
    data_folder = out_root / name
    _ensure_dir(data_folder)

    language = "spanish"
    selection_method = "tfidf"

    corrected_code = None

    logging.info(f"Processing dataset {name} ...")
    dataset = _load_json_if_exists(synth_data_dir)
    if dataset is None:
        logging.error(f"Error loading dataset from {synth_data_dir}: File not found.")
        return
    src, tgt, _ = process_bigbio_dataset(
        dataset,
        start_entity,
        end_entity,
        start_group,
        end_group,
        CUI_to_Title=code_to_title,
        CUI_to_Syn=code_to_syn,
        CUI_to_GROUP=code_to_groups,
        semantic_info=semantic_info,
        tfidf_vectorizer_path=tfidf_vectorizer_path,
        corrected_code=corrected_code,
        language=language,
        selection_method=selection_method,
        best_syn_map=None,  # Not used for tfidf
        ner_mode=False,
    )

    # Write outputs
    logging.info("Writing output")
    _dump(
        src,
        data_folder / f"train_{selection_method}_source.pkl",
    )
    _dump(
        tgt,
        data_folder / f"train_{selection_method}_target.pkl",
    )


def _process_spaccc_dataset(
    name: str,
    code_to_title,
    code_to_syn,
    code_to_groups,
    semantic_info,
    tfidf_vectorizer_path: Path,
    start_entity: str,
    end_entity: str,
    start_group: str,
    end_group: str,
    out_root: Path,
    spaccc_data_dir: Path,
    corrected_code_path: Path,
):
    logging.info(f"→ Loading dataset {name} ...")
    data_folder = out_root / name
    _ensure_dir(data_folder)

    language = "spanish"

    corrected_code = None
    if corrected_code_path.exists():
        logging.info(f"  • Using corrected code mapping from {corrected_code_path}...")
        corrected_code = {
            str(row[0]): str(row[1])
            for row in pl.read_csv(corrected_code_path).iter_rows()
        }

    logging.info(f"Processing dataset {name} ...")

    splits = {}
    train_annotations_path = spaccc_data_dir / "train.tsv"
    test_annotations_path = spaccc_data_dir / "test.tsv"
    test_simple_annotations_path = spaccc_data_dir / "test_simple.tsv"
    test_ner_annotations_path = spaccc_data_dir / "test_ner.tsv"
    train_raw_files_folder = spaccc_data_dir.parent / "raw_txt" / "train"
    test_raw_files_folder = spaccc_data_dir.parent / "raw_txt" / "test"
    test_ner_raw_files_folder = spaccc_data_dir.parent / "raw_txt" / "test"
    test_simple_raw_files_folder = spaccc_data_dir.parent / "raw_txt" / "test"

    if train_annotations_path.exists():
        logging.info(f"  • Loading train split from {train_annotations_path}")
        splits["train"] = load_tsv_as_bigbio(
            train_annotations_path, train_raw_files_folder
        )
    if test_annotations_path.exists():
        logging.info(f"  • Loading test split from {test_annotations_path}")
        splits["test"] = load_tsv_as_bigbio(
            test_annotations_path, test_raw_files_folder
        )
    if test_ner_annotations_path.exists():
        logging.info(f"  • Loading test_ner split from {test_ner_annotations_path}")
        splits["test_ner"] = load_tsv_as_bigbio(
            test_ner_annotations_path, test_ner_raw_files_folder
        )
    if test_simple_annotations_path.exists():
        logging.info(
            f"  • Loading test_simple split from {test_simple_annotations_path}"
        )
        splits["test_simple"] = load_tsv_as_bigbio(
            test_simple_annotations_path, test_simple_raw_files_folder
        )
    processed = {}
    for split_name, split_data in splits.items():
        if not split_data or len(split_data) == 0:
            logging.info(f"  • Skipping empty split: {split_name}")
            continue
        logging.info(f"  • Processing split: {split_name}")
        src, tgt, tsv_data = process_bigbio_dataset(
            split_data,
            start_entity,
            end_entity,
            start_group,
            end_group,
            CUI_to_Title=code_to_title,
            CUI_to_Syn=code_to_syn,
            CUI_to_GROUP=code_to_groups,
            semantic_info=semantic_info,
            tfidf_vectorizer_path=tfidf_vectorizer_path,
            corrected_code=corrected_code,
            language=language,
            selection_method="tfidf",
            best_syn_map=None,  # Not used for tfidf
            ner_mode="ner" in split_name,
        )
        processed[split_name] = (src, tgt, tsv_data)

    # Write outputs
    selection_method = "tfidf"
    for split_name, (src, tgt, tsv_data) in processed.items():
        logging.info(f"  • Writing output for split: {split_name}")
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


@app.command()
def run(
    start_entity: str = typer.Option("[", help="Start entity marker"),
    end_entity: str = typer.Option("]", help="End entity marker"),
    start_group: str = typer.Option("{", help="Start group marker"),
    end_group: str = typer.Option("}", help="End group marker"),
    tfidf_vectorizer_path: Path = typer.Option(
        Path("models/encoder/umls_tfidf_vectorizer.joblib"),
        help="TF-IDF vectorizer model path",
    ),
    terminology_parquet: Path = typer.Option(
        Path("data/UMLS_processed/SPACCC/all_disambiguated.parquet"),
        help="UMLS SPACCC parquet",
    ),
    out_root: Path = typer.Option(
        Path("data/final_data"), help="Root output directory"
    ),
    semantic_info_parquet: Path = typer.Option(
        Path("data/UMLS_processed/SPACCC/semantic_info.parquet"),
        help="UMLS semantic info parquet. Will be created if it doesn't exist.",
    ),
    spaccc_data_dir: Path = typer.Option(
        Path("data/SPACCC/Normalization"), help="SPACCC data directory"
    ),
    corrected_code_path: Path = typer.Option(
        Path("data/corrected_code/SPACCC_adapted.csv"),
        help="Corrected SNOMED mapping file",
    ),
    synth_spaccc_def_data_dir: Path = typer.Option(
        Path("data/synthetic_data/SynthSPACCC/SynthSPACCC_bigbio_def.json"),
        help="SynthSPACCC definitions directory (BigBio format)",
    ),
    synth_spaccc_no_def_data_dir: Path = typer.Option(
        Path("data/synthetic_data/SynthSPACCC/SynthSPACCC_bigbio_no_def.json"),
        help="SynthSPACCC no definitions directory (BigBio format)",
    ),
    synth_spaccc_filtered_data_dir: Path = typer.Option(
        Path("data/synthetic_data/SynthSPACCC/SynthSPACCC_bigbio_filtered.json"),
        help="SynthSPACCC filtered directory (BigBio format)",
    ),
) -> None:
    """Run preprocessing pipeline for SPACCC dataset."""
    # Load UMLS mapping resources
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Building UMLS mappings...")
    code_to_syn, code_to_title, code_to_groups = _build_mappings(terminology_parquet)

    # Create semantic_info if it doesn't exist
    if not semantic_info_parquet.exists():
        logging.info(
            f"Semantic info file not found. Creating at {semantic_info_parquet}"
        )
        df = pl.read_parquet(terminology_parquet)
        semantic_info = df.group_by("SNOMED_code").agg([
            pl.col("GROUP").first(),
            pl.col("CATEGORY").first().alias("SEM_CODE"),
            pl.col("CATEGORY").first(),
        ])
        semantic_info.write_parquet(semantic_info_parquet)

    semantic_info = pl.read_parquet(semantic_info_parquet)

    _process_spaccc_dataset(
        "SPACCC",
        code_to_title,
        code_to_syn,
        code_to_groups,
        semantic_info,
        tfidf_vectorizer_path,
        start_entity,
        end_entity,
        start_group,
        end_group,
        out_root,
        spaccc_data_dir,
        corrected_code_path,
    )

    if synth_spaccc_filtered_data_dir.exists():
        _process_synth_dataset(
            "SynthSPACCC_Filtered",
            code_to_title,
            code_to_syn,
            code_to_groups,
            semantic_info,
            tfidf_vectorizer_path,
            start_entity,
            end_entity,
            start_group,
            end_group,
            out_root,
            synth_spaccc_filtered_data_dir,
            corrected_code_path,
        )
    if synth_spaccc_def_data_dir.exists():
        _process_synth_dataset(
            "SynthSPACCC_Def",
            code_to_title,
            code_to_syn,
            code_to_groups,
            semantic_info,
            tfidf_vectorizer_path,
            start_entity,
            end_entity,
            start_group,
            end_group,
            out_root,
            synth_spaccc_def_data_dir,
            corrected_code_path,
        )

    if synth_spaccc_no_def_data_dir.exists():
        _process_synth_dataset(
            "SynthSPACCC_No_Def",
            code_to_title,
            code_to_syn,
            code_to_groups,
            semantic_info,
            tfidf_vectorizer_path,
            start_entity,
            end_entity,
            start_group,
            end_group,
            out_root,
            synth_spaccc_no_def_data_dir,
            corrected_code_path,
        )

    logging.info("✅ Preprocessing complete.")


if __name__ == "__main__":  # pragma: no cover
    app()

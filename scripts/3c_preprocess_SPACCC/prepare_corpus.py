"""Prepare model-specific source/target datasets from SPACCC corpora (Typer CLI)."""

import json
import pickle
from pathlib import Path

import polars as pl
import typer
from datasets import Dataset

from syncabel.parse_data import process_bigbio_dataset

app = typer.Typer(
    help="Preprocess SPACCC dataset into model-specific train/dev/test pickles."
)


def _build_mappings(umls_parquet: Path):
    df = pl.read_parquet(umls_parquet)
    # The 'Title' column might not exist in the SPACCC processed parquet.
    # Let's use 'Entity' as a fallback.
    if "Title" not in df.columns:
        df = df.with_columns(pl.col("Entity").alias("Title"))
    cui_to_title = dict(df.group_by("CUI").agg([pl.col("Title").first()]).iter_rows())
    cui_to_syn = dict(df.group_by("CUI").agg([pl.col("Entity").unique()]).iter_rows())
    cui_to_groups = dict(df.group_by("CUI").agg([pl.col("GROUP").unique()]).iter_rows())
    return cui_to_syn, cui_to_title, cui_to_groups


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


def _load_spaccc_as_bigbio(annotation_file: Path, raw_files_folder: Path) -> Dataset:
    """
    Load a SPACCC TSV annotation file and corresponding raw text files, and format them as a Hugging Face Dataset in BigBio format.

    The `annotation_file` should be a TSV file with columns: doc_id, entity_type, start_span, end_span, entity_text, cui, cui_type.
    The `raw_files_folder` should be a directory containing raw text files named as <doc_id>.txt for each document referenced in the TSV.
    Each passage will include the full raw text and annotated entities with their offsets.
    """
    # Read annotations and group by document
    try:
        # Read the file first without forcing column names
        annotations_df = pl.read_csv(
            annotation_file,
            separator="\t",
            has_header=True,
        )

        # Define the expected column names (7 total)
        expected_cols = [
            "doc_id",
            "entity_type",
            "start_span",
            "end_span",
            "entity_text",
            "cui",
            "cui_type",
        ]

        # If fewer than expected columns exist, add missing ones as null columns
        for col in expected_cols[len(annotations_df.columns) :]:
            annotations_df = annotations_df.with_columns(pl.lit(None).alias(col))

        # Rename all columns to expected names
        annotations_df = annotations_df.rename(
            dict(zip(annotations_df.columns, expected_cols))
        )
    except pl.ShapeError:
        # Handle empty file
        typer.echo(f"Warning: Annotation file {annotation_file} is empty.")
        return Dataset.from_dict({
            "id": [],
            "document_id": [],
            "text": [],
            "entities": [],
        })

    pages = []
    id = 0
    for doc_id, group in annotations_df.group_by("doc_id"):
        # Load raw text for the document
        raw_text_path = raw_files_folder / f"{doc_id[0]}.txt"
        if raw_text_path.exists():
            with raw_text_path.open("r", encoding="utf-8") as raw_file:
                text = raw_file.read()
        else:
            # Skip documents with no corresponding text file
            typer.echo(f"Warning: Raw text file {raw_text_path} not found.")
            continue

        page = {
            "id": doc_id[0],
            "document_id": doc_id[0],
        }

        entities = []
        for i, record in enumerate(group.to_dicts()):
            entity = {
                "id": f"{doc_id[0]}_T{i}",
                "text": [record["entity_text"]],
                "offsets": [[int(record["start_span"]), int(record["end_span"])]],
                "type": record["entity_type"],
                "normalized": [{"db_name": "SNOMED_CT", "db_id": record["cui"]}],
            }
            entities.append(entity)
        page["entities"] = entities
        page["passages"] = [
            {
                "id": f"{doc_id[0]}_passage",
                "type": "clinical_case",
                "text": [text],
                "offsets": [[0, len(text)]],
            }
        ]
        pages.append(page)
        id += 1

    if not pages:
        return Dataset.from_list([
            {
                "id": None,
                "document_id": None,
                "passages": [],
                "entities": [],
            }
        ])

    # Convert to Hugging Face Dataset
    typer.echo(f"Loaded {len(pages)} pages from {annotation_file}")
    return Dataset.from_list(pages)


def _process_synth_dataset(
    name: str,
    cui_to_title,
    cui_to_syn,
    cui_to_groups,
    semantic_info,
    tfidf_vectorizer_path: Path,
    start_entity: str,
    end_entity: str,
    start_group: str,
    end_group: str,
    out_root: Path,
    synth_data_dir: Path,
    corrected_cui_path: Path,
):
    """Process a synthetic dataset already in BigBio format."""
    typer.echo(f"→ Loading synthetic dataset {name} ...")
    data_folder = out_root / name
    _ensure_dir(data_folder)

    language = "spanish"
    selection_method = "tfidf"

    corrected_cui = None

    typer.echo(f"Processing dataset {name} ...")
    dataset = _load_json_if_exists(synth_data_dir)
    if dataset is None:
        typer.echo(f"Error loading dataset from {synth_data_dir}: File not found.")
        return
    src, src_with_group, tgt, _ = process_bigbio_dataset(
        dataset,
        start_entity,
        end_entity,
        start_group,
        end_group,
        CUI_to_Title=cui_to_title,
        CUI_to_Syn=cui_to_syn,
        CUI_to_GROUP=cui_to_groups,
        semantic_info=semantic_info,
        tfidf_vectorizer_path=tfidf_vectorizer_path,
        corrected_cui=corrected_cui,
        language=language,
        selection_method=selection_method,
        best_syn_map=None,  # Not used for tfidf
        ner_mode=False,
    )

    # Write outputs
    typer.echo("Writing output")
    _dump(
        src,
        data_folder / f"train_{selection_method}_source.pkl",
    )
    _dump(
        src_with_group,
        data_folder / f"train_{selection_method}_source_with_group.pkl",
    )
    _dump(
        tgt,
        data_folder / f"train_{selection_method}_target.pkl",
    )


def _process_spaccc_dataset(
    name: str,
    cui_to_title,
    cui_to_syn,
    cui_to_groups,
    semantic_info,
    tfidf_vectorizer_path: Path,
    start_entity: str,
    end_entity: str,
    start_group: str,
    end_group: str,
    out_root: Path,
    spaccc_data_dir: Path,
    corrected_cui_path: Path,
):
    typer.echo(f"→ Loading dataset {name} ...")
    data_folder = out_root / name
    _ensure_dir(data_folder)

    language = "spanish"

    corrected_cui = None
    if corrected_cui_path.exists():
        typer.echo(f"  • Using corrected CUI mapping from {corrected_cui_path}...")
        corrected_cui = {
            str(row[0]): str(row[1])
            for row in pl.read_csv(corrected_cui_path).iter_rows()
        }

    typer.echo(f"Processing dataset {name} ...")

    splits = {}
    train_annotations_path = spaccc_data_dir / "train.tsv"
    test_annotations_path = spaccc_data_dir / "test.tsv"
    test_ner_annotations_path = spaccc_data_dir / "test_ner.tsv"
    train_raw_files_folder = spaccc_data_dir / "raw_txt" / "train"
    test_raw_files_folder = spaccc_data_dir / "raw_txt" / "test"
    test_ner_raw_files_folder = spaccc_data_dir / "raw_txt" / "test"

    if train_annotations_path.exists():
        typer.echo(f"  • Loading train split from {train_annotations_path}")
        splits["train"] = _load_spaccc_as_bigbio(
            train_annotations_path, train_raw_files_folder
        )
    if test_annotations_path.exists():
        typer.echo(f"  • Loading test split from {test_annotations_path}")
        splits["test"] = _load_spaccc_as_bigbio(
            test_annotations_path, test_raw_files_folder
        )
    if test_ner_annotations_path.exists():
        typer.echo(f"  • Loading test_ner split from {test_ner_annotations_path}")
        splits["test_ner"] = _load_spaccc_as_bigbio(
            test_ner_annotations_path, test_ner_raw_files_folder
        )

    processed = {}
    for split_name, split_data in splits.items():
        if not split_data or len(split_data) == 0:
            typer.echo(f"  • Skipping empty split: {split_name}")
            continue
        typer.echo(f"  • Processing split: {split_name}")
        src, src_with_group, tgt, tsv_data = process_bigbio_dataset(
            split_data,
            start_entity,
            end_entity,
            start_group,
            end_group,
            CUI_to_Title=cui_to_title,
            CUI_to_Syn=cui_to_syn,
            CUI_to_GROUP=cui_to_groups,
            semantic_info=semantic_info,
            tfidf_vectorizer_path=tfidf_vectorizer_path,
            corrected_cui=corrected_cui,
            language=language,
            selection_method="tfidf",
            best_syn_map=None,  # Not used for tfidf
            ner_mode="ner" in split_name,
        )
        processed[split_name] = (src, src_with_group, tgt, tsv_data)

    # Write outputs
    selection_method = "tfidf"
    for split_name, (src, src_with_group, tgt, tsv_data) in processed.items():
        typer.echo(f"  • Writing output for split: {split_name}")
        _dump(
            src,
            data_folder / f"{split_name}_{selection_method}_source.pkl",
        )
        _dump(
            src_with_group,
            data_folder / f"{split_name}_{selection_method}_source_with_group.pkl",
        )
        _dump(
            tgt,
            data_folder / f"{split_name}_{selection_method}_target.pkl",
        )
        pl.DataFrame(tsv_data).write_csv(
            data_folder / f"{split_name}_{selection_method}_annotations.tsv"
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
    umls_parquet: Path = typer.Option(
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
        Path("data/SPACCC"), help="SPACCC data directory"
    ),
    corrected_cui_path: Path = typer.Option(
        Path("data/corrected_cui/SPACCC_adapted.csv"),
        help="Corrected CUI mapping file",
    ),
    synth_spaccc_data_dir: Path = typer.Option(
        Path("data/synthetic_data/SynthSPACCC/SynthSPACCC_bigbio_no_def.json"),
        help="SynthSPACCC data directory (BigBio format)",
    ),
) -> None:
    """Run preprocessing pipeline for SPACCC dataset."""
    # Load UMLS mapping resources
    typer.echo("Building UMLS mappings...")
    cui_to_syn, cui_to_title, cui_to_groups = _build_mappings(umls_parquet)

    # Create semantic_info if it doesn't exist
    if not semantic_info_parquet.exists():
        typer.echo(f"Semantic info file not found. Creating at {semantic_info_parquet}")
        df = pl.read_parquet(umls_parquet)
        semantic_info = df.group_by("CUI").agg([
            pl.col("GROUP").first(),
            pl.col("CATEGORY").first().alias("SEM_CODE"),
            pl.col("CATEGORY").first(),
        ])
        semantic_info.write_parquet(semantic_info_parquet)

    semantic_info = pl.read_parquet(semantic_info_parquet)

    _process_spaccc_dataset(
        "SPACCC",
        cui_to_title,
        cui_to_syn,
        cui_to_groups,
        semantic_info,
        tfidf_vectorizer_path,
        start_entity,
        end_entity,
        start_group,
        end_group,
        out_root,
        spaccc_data_dir,
        corrected_cui_path,
    )

    if synth_spaccc_data_dir.exists():
        _process_synth_dataset(
            "SynthSPACCC",
            cui_to_title,
            cui_to_syn,
            cui_to_groups,
            semantic_info,
            tfidf_vectorizer_path,
            start_entity,
            end_entity,
            start_group,
            end_group,
            out_root,
            synth_spaccc_data_dir,
            corrected_cui_path,
        )

    typer.echo("✅ Preprocessing complete.")


if __name__ == "__main__":  # pragma: no cover
    app()

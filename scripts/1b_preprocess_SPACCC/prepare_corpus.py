"""Prepare model-specific source/target datasets from SPACCC corpora (Typer CLI)."""

import logging
from pathlib import Path

import typer

from longbel.utils import load_tsv_as_bigbio

app = typer.Typer(help="Preprocess SPACCC dataset into BigBio JSON format.")


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _process_spaccc_dataset(
    out_root: Path,
    spaccc_data_dir: Path,
):
    logging.info("→ Loading dataset SPACCC ...")
    data_folder = out_root / "SPACCC"
    _ensure_dir(data_folder)

    splits = {}
    train_annotations_path = spaccc_data_dir / "train.tsv"
    test_annotations_path = spaccc_data_dir / "test.tsv"
    test_simple_annotations_path = spaccc_data_dir / "test_simple.tsv"
    test_ner_annotations_path = spaccc_data_dir / "test_ner.tsv"

    # Raw files are expected to be in parallel directories
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

    for split_name, split_data in splits.items():
        if not split_data or len(split_data) == 0:
            logging.info(f"  • Skipping empty split: {split_name}")
            continue
        logging.info(f"  • Saving split: {split_name}")

        # Save as JSON (BigBio format compatible).
        # to_json saves as JSON Lines by default (lines=True) or we specifies it.
        # Usually BigBio datasets are loaded via JSON.
        out_file = data_folder / f"SPACCC_{split_name}.json"
        split_data.to_json(out_file)
        logging.info(f"    Saved to {out_file}")


@app.command()
def run(
    out_root: Path = typer.Option(
        Path("data/bigbio/"), help="Root output directory for BigBio JSONs"
    ),
    spaccc_data_dir: Path = typer.Option(
        Path("data/SPACCC/Normalization"), help="SPACCC data directory"
    ),
) -> None:
    """Run preprocessing pipeline for SPACCC dataset (Convert to BigBio)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    _process_spaccc_dataset(
        out_root,
        spaccc_data_dir,
    )

    logging.info("✅ SPACCC BigBio conversion complete.")


if __name__ == "__main__":
    app()

import json
import logging
import pickle
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, cast

import polars as pl
import typer
from datasets import load_dataset

from syncabel.utils import load_tsv_as_bigbio


def prepare_dictionary_from_umls(umls_path: Path):
    """Prepare dictionary pickle files from UMLS data for encoder training/eval."""
    umls_df = pl.read_parquet(umls_path / "all_disambiguated.parquet")
    # Rename SNOMED_code into CUI for consistency if needed
    if "SNOMED_code" in umls_df.columns:
        umls_df = umls_df.rename({"SNOMED_code": "CUI"})

    umls_df = (
        umls_df.group_by(["CUI", "Title", "GROUP"])
        .agg(pl.col("Entity").unique())
        .sort("GROUP", "CUI")
    )
    umls_df = umls_df.with_columns(
        description=pl.col("Title")
        + " ( "
        + pl.col("GROUP")
        + " : "
        + pl.col("Entity").list.join(" ; ")
        + " )"
    )
    umls_df = umls_df.drop("Entity")

    # Nested dict: { type: { cui: {"title": ..., "description": ...} } }
    records = defaultdict(dict)

    for row in umls_df.to_dicts():
        records[row["GROUP"]][row["CUI"]] = {
            "title": row["Title"],
            "description": row["description"],
        }

    # Convert back to normal dict if desired
    records = dict(records)

    # Save nested dictionary in pickle format
    with open(umls_path / "umls_info_encoder.pkl", "wb") as f:
        pickle.dump(records, f)

    logging.info(f"UMLS info saved to {umls_path / 'umls_info_encoder.pkl'}")

    rename_map = {
        "CUI": "cui",
        "Title": "title",
        "description": "description",
        "GROUP": "type",
    }

    # Convert to list of dicts with renamed keys
    records = umls_df.rename(rename_map).to_dicts()

    return records


def _transform_pages(
    pages: Iterable[dict],
    umls_info: dict,
    semantic_info: pl.DataFrame,
    cui_to_groups: dict,
    corrected_code: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Transform a sequence of BigBio-style pages into BLINK-style mention dicts."""
    # Precompute mappings for efficient GROUP lookup
    cat_to_group = {
        row["CATEGORY"]: row["GROUP"]
        for row in semantic_info.select(["CATEGORY", "GROUP"]).to_dicts()
    }
    sem_to_group = {
        row["SEM_CODE"]: row["GROUP"]
        for row in semantic_info.select(["SEM_CODE", "GROUP"]).to_dicts()
    }
    blink_mentions: list[dict] = []
    for page in pages:
        document_id = page["document_id"]  # type: ignore
        all_text = " ".join([
            passage["text"][0] for passage in page.get("passages", [])
        ])  # type: ignore
        entity_id = 1
        for entity in page.get("entities", []):  # type: ignore
            if not entity.get("normalized"):
                # No normalized id -> skip (no annotation)
                ent_text = (
                    " ".join(entity.get("text", [])) if entity.get("text") else ""
                )
                logging.warning(f"Entity '{ent_text}' has no CUI; skipping.")
                continue
            # Extract CUI
            cui = entity["normalized"][0]["db_id"]  # type: ignore
            if corrected_code and cui in corrected_code:
                cui = corrected_code[cui]
                logging.info(
                    f"Corrected CUI {entity['normalized'][0]['db_id']} -> {cui} for entity '{' '.join(entity.get('text', []))}'"
                )
            # Determine group
            entity_type = entity.get("type")  # type: ignore
            groups = cui_to_groups.get(cui, [])
            if len(groups) == 1:
                group = groups[0]
            else:
                if entity_type in cat_to_group.values():
                    group = entity_type
                elif entity_type in cat_to_group.keys():
                    group = cat_to_group[entity_type]
                elif entity_type in sem_to_group.keys():
                    group = sem_to_group[entity_type]
                else:
                    group = "Unknown"
                    logging.info(f"No group found for entity type {entity_type}.")
                if group not in groups and groups:
                    group = groups[0]
            if group == "Unknown":
                logging.info(
                    f"Group is 'Unknown' for CUI {cui} and entity type {entity_type}. skipping."
                )
                continue
            if group not in umls_info.keys():
                ent_text = (
                    " ".join(entity.get("text", [])) if entity.get("text") else ""
                )
                logging.warning(
                    f"Group '{group}' not found in UMLS info; skipping entity '{ent_text}'."
                )
                continue
            if cui not in umls_info[group].keys():
                ent_text = (
                    " ".join(entity.get("text", [])) if entity.get("text") else ""
                )
                logging.warning(
                    f"CUI '{cui}' not found in UMLS info under group '{group}'; skipping entity '{ent_text}'."
                )
                continue
            label = umls_info[group][cui]["description"]
            label_title = umls_info[group][cui]["title"]
            # Context windows
            offsets = entity.get("offsets", [])  # type: ignore
            start_index = offsets[0][0] if offsets and offsets[0] else 0
            end_index = offsets[-1][1] if offsets and offsets[-1] else 0
            context_left = all_text[:start_index].strip()
            context_right = all_text[end_index:].strip()
            mention = " ".join(entity.get("text", [])).strip()
            transformed_mention = {
                "mention": mention,
                "mention_id": f"{document_id}.{entity_id}",
                "context_left": context_left,
                "context_right": context_right,
                "context_doc_id": document_id,
                "type": group,
                "label_id": cui,
                "label": label,
                "label_title": label_title,
            }
            entity_id += 1
            blink_mentions.append(transformed_mention)
    return blink_mentions


def process_bigbio_dataset(
    hf_id: Optional[str],
    hf_config: Optional[str],
    input_path: Optional[Path],
    output_path: Path,
    umls_path: Path,
    corrected_code: Optional[dict[str, str]] = None,
):
    dataset = {}
    if hf_id is not None and hf_config is not None:
        dataset = load_dataset(hf_id, hf_config)
    elif input_path is not None:
        logging.info(f"Converting dataset from folder: {input_path} to BigBio...")
        train_annotations_path = input_path / "train.tsv"
        test_annotations_path = input_path / "test.tsv"
        train_raw_files_folder = input_path.parent / "raw_txt" / "train"
        test_raw_files_folder = input_path.parent / "raw_txt" / "test"

        if train_annotations_path.exists():
            logging.info(f"  • Loading train split from {train_annotations_path}")
            dataset["train"] = load_tsv_as_bigbio(
                train_annotations_path, train_raw_files_folder
            )
        if test_annotations_path.exists():
            logging.info(f"  • Loading test split from {test_annotations_path}")
            dataset["test"] = load_tsv_as_bigbio(
                test_annotations_path, test_raw_files_folder
            )
    umls_df = pl.read_parquet(umls_path / "all_disambiguated.parquet")
    # Rename SNOMED_code into CUI for consistency if needed
    if "SNOMED_code" in umls_df.columns:
        umls_df = umls_df.rename({"SNOMED_code": "CUI"})
    cui_to_groups = dict(
        umls_df.group_by("CUI").agg([pl.col("GROUP").unique()]).iter_rows()
    )
    umls_info = pickle.load(open(umls_path / "umls_info_encoder.pkl", "rb"))
    semantic_info = pl.read_parquet(umls_path / "semantic_info.parquet")
    for split in ["validation", "test", "train"]:
        if split not in dataset:
            continue
        logging.info(f"Processing split: {split}")
        pages = dataset[split]
        blink_mentions = _transform_pages(
            cast(Iterable[dict], pages),
            umls_info,
            semantic_info,
            cui_to_groups,
            corrected_code,
        )
        # write all of the transformed mentions
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Writing {len(blink_mentions)} processed mentions to file...")
        if split == "validation":
            split_name = "valid"
        else:
            split_name = split
        with open(output_path / f"{split_name}.jsonl", "w") as f:
            f.write("\n".join([json.dumps(m) for m in blink_mentions]))
        logging.info(
            f"Finished writing {split} mentions to {output_path / f'{split_name}.jsonl'}."
        )


# --- Typer CLI -------------------------------------------------------------

app = typer.Typer(
    help="Prepare encoder-style data and dictionaries from BigBio corpora."
)


@app.command()
def run(
    datasets: list[str] = typer.Option(
        ["MedMentions", "EMEA", "MEDLINE", "SPACCC"],
        help="Datasets to process. Include both original and synthetic to create augmented versions.",
    ),
    umls_mm_path: Path = typer.Option(
        Path("data/UMLS_processed/MM"), help="Path to UMLS MM directory"
    ),
    umls_quaero_path: Path = typer.Option(
        Path("data/UMLS_processed/QUAERO"), help="Path to UMLS QUAERO directory"
    ),
    umls_spaccc_path: Path = typer.Option(
        Path("data/UMLS_processed/SPACCC"), help="Path to UMLS SPACCC directory"
    ),
    out_root: Path = typer.Option(
        Path("arboEL/data/final_data_encoder"), help="Root output directory"
    ),
):
    """Run dictionary prep and dataset processing for encoder training/eval."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Prepare dictionaries once for MM, QUAERO, SPACCC
    typer.echo("→ Preparing dictionaries (MM, QUAERO, SPACCC)…")
    dictionary_mm = prepare_dictionary_from_umls(umls_mm_path)
    medmentions_path = out_root / "MedMentions"
    medmentions_path.mkdir(parents=True, exist_ok=True)
    with open(medmentions_path / "dictionary.pickle", "wb") as f:
        pickle.dump(dictionary_mm, f)
    typer.echo(f"MM dictionary saved to {medmentions_path / 'dictionary.pickle'}")
    dictionary_quaero = prepare_dictionary_from_umls(umls_quaero_path)
    emea_path = out_root / "EMEA"
    emea_path.mkdir(parents=True, exist_ok=True)
    with open(emea_path / "dictionary.pickle", "wb") as f:
        pickle.dump(dictionary_quaero, f)
    typer.echo(f"EMEA dictionary saved to {emea_path / 'dictionary.pickle'}")
    medline_path = out_root / "MEDLINE"
    medline_path.mkdir(parents=True, exist_ok=True)
    with open(medline_path / "dictionary.pickle", "wb") as f:
        pickle.dump(dictionary_quaero, f)
    typer.echo(f"MEDLINE dictionary saved to {medline_path / 'dictionary.pickle'}")
    dictionary_spaccc = prepare_dictionary_from_umls(umls_spaccc_path)
    spaccc_path = out_root / "SPACCC"
    spaccc_path.mkdir(parents=True, exist_ok=True)
    with open(spaccc_path / "dictionary.pickle", "wb") as f:
        pickle.dump(dictionary_spaccc, f)
    typer.echo(f"SPACCC dictionary saved to {spaccc_path / 'dictionary.pickle'}")

    # Optional: corrected CUI mapping for QUAERO (from manual review)
    quaero_corrected_code = None
    if "EMEA" in datasets or "MEDLINE" in datasets:
        quaero_corrected_code_path = (
            Path("data") / "corrected_code" / "QUAERO_2014_adapted.csv"
        )
        if quaero_corrected_code_path.exists():
            typer.echo("Using corrected CUI mapping...")
            quaero_corrected_code = {
                str(row[0]): str(row[1])
                for row in pl.read_csv(quaero_corrected_code_path).iter_rows()
            }

    # Optional: corrected SNOMED mapping for SPACCC
    spaccc_corrected_code = None
    if "SPACCC" in datasets:
        spaccc_corrected_code_path = (
            Path("data") / "corrected_code" / "SPACCC_adapted.csv"
        )
        if spaccc_corrected_code_path.exists():
            typer.echo("Using corrected SNOMED mapping...")
            spaccc_corrected_code = {
                str(row[0]): str(row[1])
                for row in pl.read_csv(spaccc_corrected_code_path).iter_rows()
            }

    # Process HF datasets
    if "MedMentions" in datasets:
        typer.echo("→ Processing MedMentions (HF)…")
        process_bigbio_dataset(
            "bigbio/medmentions",
            "medmentions_st21pv_bigbio_kb",
            None,
            medmentions_path,
            umls_mm_path,
        )
    if "EMEA" in datasets:
        typer.echo("→ Processing QUAERO EMEA (HF)…")
        process_bigbio_dataset(
            "bigbio/quaero",
            "quaero_emea_bigbio_kb",
            None,
            emea_path,
            umls_quaero_path,
            quaero_corrected_code,
        )
    if "MEDLINE" in datasets:
        typer.echo("→ Processing QUAERO MEDLINE (HF)…")
        process_bigbio_dataset(
            "bigbio/quaero",
            "quaero_medline_bigbio_kb",
            None,
            medline_path,
            umls_quaero_path,
            quaero_corrected_code,
        )

    if "SPACCC" in datasets:
        typer.echo("→ Processing SPACCC (local)…")
        process_bigbio_dataset(
            None,
            None,
            Path("data/SPACCC/Normalization/"),
            spaccc_path,
            umls_spaccc_path,
            spaccc_corrected_code,
        )

    typer.echo("✅ Encoder data preparation complete.")


if __name__ == "__main__":  # pragma: no cover
    app()

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


def prepare_dictionary_from_umls(umls_path: Path):
    """Prepare dictionary pickle files from UMLS data for encoder training/eval."""
    umls_df = pl.read_parquet(umls_path / "all_disambiguated.parquet")
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
    corrected_cui: Optional[dict[str, str]] = None,
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
            if corrected_cui and cui in corrected_cui:
                cui = corrected_cui[cui]
                logging.info(
                    f"Corrected CUI {entity['normalized'][0]['db_id']} -> {cui} for entity '{' '.join(entity.get('text', []))}'"
                )
            # Determine group
            entity_type = entity.get("type")  # type: ignore
            groups = cui_to_groups.get(cui, [])
            if len(groups) == 1:
                group = groups[0]
            else:
                if entity_type in cat_to_group.keys():
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
    hf_id: str,
    hf_config: str,
    output_path: Path,
    umls_path: Path,
    corrected_cui: Optional[dict[str, str]] = None,
):
    dataset = load_dataset(hf_id, hf_config)
    umls_df = pl.read_parquet(umls_path / "all_disambiguated.parquet")
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
            corrected_cui,
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


def create_augmented_dataset(
    name: str,
    original_path: Path,
    synth_json_path: Path,
    umls_path: Path,
    dictionary: list[dict],
    out_root: Path,
    corrected_cui: Optional[dict[str, str]] = None,
):
    """Create an augmented dataset combining original val/test with original+synthetic train."""
    augmented_path = out_root / name
    augmented_path.mkdir(parents=True, exist_ok=True)

    # Save dictionary for augmented dataset
    with open(augmented_path / "dictionary.pickle", "wb") as f:
        pickle.dump(dictionary, f)
    logging.info(f"{name} dictionary saved to {augmented_path / 'dictionary.pickle'}")

    # Copy validation and test splits from original (if they exist)
    for split in ["validation", "test"]:
        if split == "validation":
            split_name = "valid"
        else:
            split_name = split
        original_split_file = original_path / f"{split_name}.jsonl"
        if original_split_file.exists():
            augmented_split_file = augmented_path / f"{split_name}.jsonl"
            augmented_split_file.write_text(original_split_file.read_text())
            logging.info(f"Copied {split} split to {augmented_split_file}")

    # Create augmented train split: original train + synthetic train
    original_train = original_path / "train.jsonl"
    synthetic_mentions = []

    # Process synthetic data if available
    if synth_json_path.exists():
        pages = json.loads(synth_json_path.read_text(encoding="utf-8"))
        umls_df = pl.read_parquet(umls_path / "all_disambiguated.parquet")
        cui_to_groups = dict(
            umls_df.group_by("CUI").agg([pl.col("GROUP").unique()]).iter_rows()
        )
        umls_info = pickle.load(open(umls_path / "umls_info_encoder.pkl", "rb"))
        semantic_info = pl.read_parquet(umls_path / "semantic_info.parquet")
        synthetic_mentions = _transform_pages(
            pages,
            umls_info,
            semantic_info,
            cui_to_groups,
            corrected_cui,
        )
        logging.info(
            f"Processed {len(synthetic_mentions)} synthetic mentions from {synth_json_path.name}"
        )

    # Combine original train + synthetic
    augmented_train_file = augmented_path / "train.jsonl"
    with open(augmented_train_file, "w") as f:
        # Write original training data first
        if original_train.exists():
            f.write(original_train.read_text().rstrip())
            if synthetic_mentions:
                f.write("\n")  # Add newline separator

        # Write synthetic data
        if synthetic_mentions:
            f.write("\n".join([json.dumps(m) for m in synthetic_mentions]))

    # Count total mentions in augmented train
    total_train_lines = sum(1 for _ in open(augmented_train_file))
    logging.info(
        f"Created augmented train split with {total_train_lines} total mentions in {augmented_train_file}"
    )


# --- Typer CLI -------------------------------------------------------------

app = typer.Typer(
    help="Prepare encoder-style data and dictionaries from BigBio corpora."
)


@app.command()
def run(
    datasets: list[str] = typer.Option(
        ["MedMentions", "EMEA", "MEDLINE", "SynthMM", "SynthQUAERO"],
        help="Datasets to process. Include both original and synthetic to create augmented versions.",
    ),
    umls_mm_path: Path = typer.Option(
        Path("data/UMLS_processed/MM"), help="Path to UMLS MM directory"
    ),
    umls_quaero_path: Path = typer.Option(
        Path("data/UMLS_processed/QUAERO"), help="Path to UMLS QUAERO directory"
    ),
    out_root: Path = typer.Option(
        Path("arboEL/data/final_data_encoder"), help="Root output directory"
    ),
    synth_mm_json: Path = typer.Option(
        Path("data/synthetic_data/SynthMM/SynthMM_bigbio_def.json"),
        help="Synthetic MedMentions JSON path",
    ),
    synth_quaero_json: Path = typer.Option(
        Path("data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio_def.json"),
        help="Synthetic QUAERO JSON path",
    ),
):
    """Run dictionary prep and dataset processing for encoder training/eval."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Prepare dictionaries once for MM and QUAERO
    typer.echo("→ Preparing dictionaries (MM, QUAERO)…")
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

    # Optional: corrected CUI mapping for QUAERO (from manual review)
    corrected_cui = None
    if "EMEA" in datasets or "MEDLINE" in datasets:
        corrected_cui_path = Path("data") / "corrected_cui" / "QUAERO_2014_adapted.csv"
        if corrected_cui_path.exists():
            typer.echo("Using corrected CUI mapping...")
            corrected_cui = dict(pl.read_csv(corrected_cui_path).iter_rows())

    # Process HF datasets
    if "MedMentions" in datasets:
        typer.echo("→ Processing MedMentions (HF)…")
        process_bigbio_dataset(
            "bigbio/medmentions",
            "medmentions_st21pv_bigbio_kb",
            medmentions_path,
            umls_mm_path,
        )
    if "EMEA" in datasets:
        typer.echo("→ Processing QUAERO EMEA (HF)…")
        process_bigbio_dataset(
            "bigbio/quaero",
            "quaero_emea_bigbio_kb",
            emea_path,
            umls_quaero_path,
            corrected_cui,
        )
    if "MEDLINE" in datasets:
        typer.echo("→ Processing QUAERO MEDLINE (HF)…")
        process_bigbio_dataset(
            "bigbio/quaero",
            "quaero_medline_bigbio_kb",
            medline_path,
            umls_quaero_path,
            corrected_cui,
        )

    # Process augmented datasets (original + synthetic)
    if "SynthMM" in datasets and "MedMentions" in datasets:
        typer.echo("→ Creating MedMentions_augmented (MedMentions + SynthMM)…")
        create_augmented_dataset(
            "MedMentions_augmented",
            medmentions_path,
            synth_mm_json,
            umls_mm_path,
            dictionary_mm,
            out_root,
        )
    if "SynthQUAERO" in datasets and "EMEA" in datasets:
        typer.echo("→ Creating EMEA_augmented (EMEA + SynthQUAERO)…")
        create_augmented_dataset(
            "EMEA_augmented",
            emea_path,
            synth_quaero_json,
            umls_quaero_path,
            dictionary_quaero,
            out_root,
            corrected_cui,
        )
    if "SynthQUAERO" in datasets and "MEDLINE" in datasets:
        typer.echo("→ Creating MEDLINE_augmented (MEDLINE + SynthQUAERO)…")
        create_augmented_dataset(
            "MEDLINE_augmented",
            medline_path,
            synth_quaero_json,
            umls_quaero_path,
            dictionary_quaero,
            out_root,
            corrected_cui,
        )

    typer.echo("✅ Encoder data preparation complete.")


if __name__ == "__main__":  # pragma: no cover
    app()

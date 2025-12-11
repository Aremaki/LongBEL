# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import polars as pl
from datasets import Dataset


def chunk_it(seq, num):
    assert num > 0
    chunk_len = len(seq) // num
    chunks = [seq[i * chunk_len : i * chunk_len + chunk_len] for i in range(num)]

    diff = len(seq) - chunk_len * num
    for i in range(diff):
        chunks[i].append(seq[chunk_len * num + i])

    return chunks


def load_tsv_as_bigbio(annotation_file: Path, raw_files_folder: Path) -> Dataset:
    """
    Load a SPACCC TSV annotation file and corresponding raw text files, and format them as a Hugging Face Dataset in BigBio format.

    The `annotation_file` should be a TSV file with columns: doc_id, entity_type, start_span, end_span, entity_text, snomed_code, label.
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
            schema_overrides={
                "code": str,
                "filename": str,  # force as string
            },  # type: ignore
        )

        # Define the expected column names (7 total)
        expected_cols = [
            "doc_id",
            "entity_type",
            "start_span",
            "end_span",
            "entity_text",
            "code",
            "semantic_type",
        ]

        # If fewer than expected columns exist, add missing ones as null columns
        for col in expected_cols[len(annotations_df.columns) :]:
            annotations_df = annotations_df.with_columns(pl.lit(None).alias(col))

        # Rename all columns to expected names
        annotations_df = annotations_df.rename(
            dict(zip(annotations_df.columns, expected_cols))
        )
    except pl.ShapeError:  # type: ignore
        # Handle empty file
        logging.warning(f"Warning: Annotation file {annotation_file} is empty.")
        return Dataset.from_dict({
            "id": [],
            "document_id": [],
            "text": [],
            "entities": [],
        })

    # Sort to make grouping deterministic
    annotations_df = annotations_df.unique().sort(
        by=["doc_id", "entity_type", "start_span", "end_span"],
        descending=[False, False, False, True],
    )

    pages = []
    id = 0
    for doc_id, group in annotations_df.group_by("doc_id", maintain_order=True):
        # Load raw text for the document
        raw_text_path = raw_files_folder / f"{doc_id[0]}.txt"
        if raw_text_path.exists():
            with raw_text_path.open("r", encoding="utf-8") as raw_file:
                text = raw_file.read()
        else:
            # Skip documents with no corresponding text file
            logging.warning(f"Warning: Raw text file {raw_text_path} not found.")
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
                "normalized": [{"db_name": "SNOMED_CT", "db_id": record["code"]}],
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
    logging.info(f"Loaded {len(pages)} pages from {annotation_file}")
    return Dataset.from_list(pages)

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

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


def convert_tsv_as_bigbio(
    tsv_data: list[dict],
    raw_data: dict[str, list[tuple[str, str]]],
    db_name: Optional[str] = "SNOMED_CT",
) -> Dataset:
    """
    Load a SPACCC TSV annotation file and corresponding raw text files, and format them as a Hugging Face Dataset in BigBio format.

    The `annotation_file` should be a TSV file with columns: doc_id, entity_type, start_span, end_span, entity_text, snomed_code, label.
    The `raw_files_folder` should be a directory containing raw text files named as <doc_id>.txt for each document referenced in the TSV.
    Each passage will include the full raw text and annotated entities with their offsets.
    """
    annotations_df = pl.DataFrame(tsv_data)

    pages = []
    id = 0
    for doc_id, group in annotations_df.group_by("doc_id", maintain_order=True):
        page = {
            "id": doc_id[0],
            "document_id": doc_id[0],
        }

        entities = []
        for i, record in enumerate(group.to_dicts()):
            entity = {
                "id": f"{doc_id[0]}_T{i}",
                "text": [record["mention"]],
                "offsets": [[int(record["start_span"]), int(record["end_span"])]],
                "type": record["semantic_group"],
                "normalized": [
                    {
                        "db_name": db_name,
                        "db_id": record["gold_concept_code"],
                        "db_match": record["gold_concept_name"],
                    }
                ],
            }
            entities.append(entity)
        page["entities"] = entities
        page["passages"] = []
        offset = 0
        for i, (passage, passage_type) in enumerate(raw_data[doc_id[0]]):
            page["passages"].append({
                "id": f"{doc_id[0]}_passage_{i}",
                "type": passage_type,
                "text": [passage],
                "offsets": [[offset, offset + len(passage)]],
            })
            offset += len(passage) + 1
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
    logging.info(f"Loaded {len(pages)} pages")
    return Dataset.from_list(pages)


def add_headers_to_prompt(source: str, target: str, context_format: str):
    if context_format == "long":
        prompt = f"### Context\n{source.rstrip()}\n\n"
        completion = f"### Predictions\n{target}"
    elif context_format in ["short", "hybrid_medium"]:
        target_split = target.split("}")
        if len(target_split) == 2:
            prefix = target_split[0] + "}"
            completion = target_split[1]
        else:
            raise ValueError(f"Unexpected target format: {target}")
        # Add Instruction prefix to source
        prompt = f"### Context\n{source.rstrip()}\n\n### Prediction\n{prefix}"
    elif context_format in ["hybrid_short", "hybrid_long"]:
        split_target = target.split("\n")
        # remove empty string
        split_target = [s for s in split_target if s]
        if len(split_target) >= 2:
            previous_tgt = "\n".join(split_target[:-1]) + "\n"
            current_tgt = split_target[-1]
        elif len(split_target) == 1:
            previous_tgt = "None"
            current_tgt = split_target[0]
        else:
            raise ValueError(f"Unexpected target format: {target}")
        current_tgt_split = current_tgt.split("}")
        if len(current_tgt_split) == 2:
            current_tgt_prefix = current_tgt_split[0] + "}"
            completion = current_tgt_split[1]
        else:
            raise ValueError(f"Unexpected current target format: {current_tgt}")
        # Add Instruction prefix to source
        prompt = f"### Context\n{source.rstrip()}\n\n### Previous Normalizations\n{previous_tgt.rstrip()}\n\n### Prediction\n{current_tgt_prefix}"
    else:
        raise ValueError(f"Unknown context_format: {context_format}")
    return prompt, completion
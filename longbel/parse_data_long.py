import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import joblib
import nltk
import numpy as np
import polars as pl
from tqdm import tqdm

from longbel.embeddings import TextEncoder, best_by_cosine


def cal_similarity_tfidf(a: list, b: str, vectorizer):
    features_a = vectorizer.transform(a)
    features_b = vectorizer.transform([b])
    features_T = features_a.T
    sim = np.array(features_b.dot(features_T).todense())[0]
    return sim.argmax(), np.max(np.array(sim))


def clean_natural(text):
    return (
        text.replace("\xa0", " ")
        .replace("{", "(")
        .replace("}", ")")
        .replace("[", "(")
        .replace("]", ")")
    )


def parse_text(
    data,
    start_entity,
    end_entity,
    start_group,
    end_group,
    CUI_to_Title,
    CUI_to_Syn,
    CUI_to_GROUP,
    cat_to_group,
    sem_to_group,
    transition_verb,
    corrected_code=None,
    selection_method: str = "levenshtein",
    encoder: Optional[TextEncoder] = None,
    tfidf_vectorizer=None,
    best_syn_map: Optional[dict[tuple[str, str], str]] = None,
):
    """Create simple (source, target) pairs per entity.

    For each entity in the BigBio page, returns one pair where:
      - source: the sentence text that contains the entity mention
      - target: "<entity> is <annotation>" where <annotation> is the best synonym
        if available (or the normalized id otherwise).
    """
    target_texts: list[str] = []
    tsv_lines: list[dict[str, str]] = []
    source_texts: list[str] = []
    entity_id = 1
    for passage in data.get("passages", []):
        target_text = ""
        passage_text = passage["text"][0]
        start_offset_passage = passage["offsets"][0][0]
        end_offset_passage = passage["offsets"][0][1]

        passage_text = clean_natural(passage_text)

        # Iterate over entities and emit one pair per entity found in this passage
        all_spans = []
        for entity in data.get("entities", []):
            global_start = entity["offsets"][0][0]
            # Keep only entities whose start falls inside this passage
            if not (start_offset_passage <= global_start < end_offset_passage):
                continue
            entity_text = " ".join(entity["text"])
            entity_text = clean_natural(entity_text)
            if not entity.get("normalized"):
                logging.warning(
                    f"Entity '{' '.join(entity['text'])}' has no CUI; skipping."
                )
                # No normalized id -> skip (no annotation)
                continue

            normalized_ids = entity["normalized"][0]["db_id"]
            if not normalized_ids:
                logging.warning(
                    f"Entity '{entity_text}' has empty CUI; skipping entity."
                )
                continue
            normalized_ids = normalized_ids.split("+")  # Handle multiple CUIs
            annotations = []
            group_annotations = []
            for i, normalized_id in enumerate(normalized_ids):
                if corrected_code and normalized_id in corrected_code:
                    normalized_id = corrected_code[normalized_id]
                    normalized_ids[i] = normalized_id
                    logging.info(
                        f"Corrected CUI {entity['normalized'][0]['db_id']} -> {normalized_id} for entity '{entity_text}'"
                    )

                possible_syns = None
                annotation = None
                if selection_method == "title":
                    if CUI_to_Title is not None:
                        annotation = CUI_to_Title.get(normalized_id)
                    else:
                        logging.warning(
                            f"Title selection requested but no title mapping available for CUI {normalized_id} (entity '{entity_text}');"
                        )
                        continue
                # Prefer precomputed best synonyms if provided
                elif selection_method == "embedding":
                    if best_syn_map is not None:
                        pre_key = (normalized_id, entity_text)
                        if pre_key in best_syn_map:
                            annotation = best_syn_map[pre_key]
                    # Otherwise select from possible synonyms
                    if annotation is None and CUI_to_Syn is not None:
                        possible_syns = CUI_to_Syn.get(normalized_id)
                        if possible_syns and encoder is not None:
                            logging.warning(
                                f"No precomputed best synonym map provided; Selecting best synonym by embedding for CUI {normalized_id} (entity '{entity_text}')"
                            )
                            best_syn, _ = best_by_cosine(
                                encoder=encoder,
                                mention=entity_text,
                                candidates=list(possible_syns),
                            )  # type: ignore
                            annotation = best_syn
                elif selection_method == "tfidf":
                    if CUI_to_Syn is not None:
                        possible_syns = CUI_to_Syn.get(normalized_id)
                    if possible_syns and tfidf_vectorizer is not None:
                        best_idx, best_score = cal_similarity_tfidf(
                            possible_syns, entity_text, tfidf_vectorizer
                        )
                        annotation = possible_syns[best_idx]
                    else:
                        logging.warning(
                            f"TF-IDF selection requested but no synonyms or vectorizer available for CUI {normalized_id} (entity '{entity_text}');"
                        )
                        continue
                elif selection_method == "levenshtein":
                    if CUI_to_Syn is not None:
                        possible_syns = CUI_to_Syn.get(normalized_id)
                    if possible_syns:
                        # Default to Levenshtein matching (previous behavior)
                        text = entity_text
                        dists = [nltk.edit_distance(text, syn) for syn in possible_syns]
                        best_syn = possible_syns[int(np.argmin(dists))]
                        annotation = best_syn
                if annotation is None:
                    # If no synonyms mapping, skip entity
                    logging.warning(
                        f"No synonyms found for CUI {normalized_id} (entity '{entity_text}'); skipping entity."
                    )
                    continue

                if isinstance(annotation, str):
                    annotations.append(clean_natural(annotation))

                # Define CUI group
                entity_type = entity.get("type")
                groups = CUI_to_GROUP.get(normalized_id, [])
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
                        f"Group is 'Unknown' for CUI {normalized_id} and entity type {entity_type}. skipping."
                    )
                    continue

                # Append only if different
                if group not in group_annotations:
                    group_annotations.append(group)
                    # Warning if multiple groups found
                    if len(group_annotations) > 1:
                        logging.warning(
                            f"Multiple groups {group_annotations} found for CUI {normalized_id} (entity '{entity_text}')"
                        )

            # Merge annotations in a string with | separator
            if not annotations:
                continue
            group_annotation = "<SEP>".join(group_annotations)
            annotation = "<SEP>".join(annotations)

            # Get all offsets, convert to relative, and filter for this sentence
            entity_spans = []
            for off in entity["offsets"]:
                global_start_off, global_end_off = off
                if not (start_offset_passage <= global_start_off < end_offset_passage):
                    continue

                rel_start_off = global_start_off - start_offset_passage
                rel_end_off = global_end_off - start_offset_passage
                entity_spans.append((rel_start_off, rel_end_off))
            entity_spans.sort(key=lambda x: x[0], reverse=True)
            all_spans.append(entity_spans)

            # Emit the pair
            doc_id = data.get("document_id", "")
            tsv_line = {
                "filename": doc_id,
                "mention_id": f"{doc_id}.{entity_id}",
                "label": group_annotation,
                "start_span": entity["offsets"][0][0],
                "end_span": entity["offsets"][-1][1],
                "span": entity_text,
                "code": "+".join(normalized_ids),
                "semantic_rel": "EXACT" if len(normalized_ids) == 1 else "COMPOSITE",
                "annotation": annotation,
            }
            entity_id += 1
            tsv_lines.append(tsv_line)
            target_text += f"[{entity_text}] {transition_verb} {annotation}\n"

        # Sort spans in reverse to mark from the end, preventing offset shifts
        all_spans.sort(key=lambda x: x[0][0], reverse=True)
        group_annotation = "GROUP"
        for entity_span in all_spans:
            for i, (start_in_sent, end_in_sent) in enumerate(entity_span):
                if i == 0:
                    passage_text = (
                        passage_text[:start_in_sent]
                        + start_entity
                        + passage_text[start_in_sent:end_in_sent]
                        + end_entity
                        + start_group
                        + group_annotation
                        + end_group
                        + passage_text[end_in_sent:]
                    )
                else:
                    passage_text = (
                        passage_text[:start_in_sent]
                        + start_entity
                        + passage_text[start_in_sent:end_in_sent]
                        + end_entity
                        + passage_text[end_in_sent:]
                    )
        source_texts.append(passage_text)
        target_texts.append(target_text)

    return source_texts, target_texts, tsv_lines


def process_bigbio_dataset(
    bigbio_dataset,
    start_entity,
    end_entity,
    start_group,
    end_group,
    CUI_to_Title,
    CUI_to_Syn,
    CUI_to_GROUP,
    semantic_info: pl.DataFrame,
    encoder_name=None,
    tfidf_vectorizer_path: Optional[Path] = None,
    corrected_code=None,
    language: str = "english",
    selection_method: str = "levenshtein",
    best_syn_map: Optional[dict[tuple[str, str], str]] = None,
):
    """Process a BigBio KB dataset into source/target sequences.

    Parameters
    ----------
    language: str
        Punkt language model to use (e.g. 'english', 'french').
    """
    # Build quick lookup of category/group and sem_code/group
    cat_to_group = {
        row["CATEGORY"]: row["GROUP"]
        for row in semantic_info.select(["CATEGORY", "GROUP"]).to_dicts()
    }
    sem_to_group = {
        row["SEM_CODE"]: row["GROUP"]
        for row in semantic_info.select(["SEM_CODE", "GROUP"]).to_dicts()
    }
    # transition verb depend on language (french, english, spanish)
    if language == "french":
        transition_verb = "est"
    elif language == "spanish":
        transition_verb = "es"
    else:
        transition_verb = "is"
    target_data = []
    source_data = []
    tsv_data = []
    if selection_method == "embedding" and encoder_name and best_syn_map is None:
        encoder = TextEncoder(model_name=encoder_name)
        print(f"Using embedding-based selection with encoder '{encoder_name}'.")
    else:
        encoder = None
    if selection_method == "tfidf" and tfidf_vectorizer_path:
        try:
            tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
            print(
                f"Using TF-IDF-based selection with vectorizer at '{tfidf_vectorizer_path}'."
            )
        except Exception as e:
            print(
                f"⚠️ Failed to load TF-IDF vectorizer from '{tfidf_vectorizer_path}': {e}"
            )
            tfidf_vectorizer = None
    else:
        tfidf_vectorizer = None

    for page in tqdm(bigbio_dataset, total=len(bigbio_dataset)):
        source_texts, target_texts, tsv_lines = parse_text(
            page,
            start_entity,
            end_entity,
            start_group,
            end_group,
            CUI_to_Title,
            CUI_to_Syn,
            CUI_to_GROUP,
            cat_to_group,
            sem_to_group,
            transition_verb,
            corrected_code,
            selection_method,
            encoder,
            tfidf_vectorizer,
            best_syn_map,
        )
        # Each entity yields one pair; extend the global lists accordingly.
        target_data.extend(target_texts)
        source_data.extend(source_texts)
        tsv_data.extend(tsv_lines)

    return source_data, target_data, tsv_data


def compute_best_synonym_df(
    bigbio_dataset: Iterable[dict],
    CUI_to_Syn: dict[str, Iterable[str]],
    encoder_name: str = "encoder/coder-all",
    batch_size: int = 4096,
    corrected_code: Optional[dict[str, str]] = None,
) -> "pl.DataFrame":
    """Precompute best synonyms per unique (CUI, entity) using batched embeddings.

    Returns a DataFrame with columns: [CUI, entity, best_synonym].

    Notes
    -----
    - Deduplicates pairs by (CUI, entity) for efficiency.
    - Uses cosine similarity via best_by_cosine with precomputed vectors.
    """
    # Gather unique (CUI, entity) pairs and the set of CUIs present in dataset
    unique_pairs: set[tuple[str, str]] = set()
    present_cuis: set[str] = set()
    for page in bigbio_dataset:
        for ent in page.get("entities", []):
            if not ent.get("normalized"):
                continue
            cui = ent["normalized"][0]["db_id"]
            if corrected_code and cui in corrected_code:
                cui = corrected_code[cui]
            mention = clean_natural(" ".join(ent["text"]))
            unique_pairs.add((cui, mention))
            present_cuis.add(cui)

    # Build per-CUI synonym lists (ensure non-empty)
    cui_to_syns: dict[str, list[str]] = {}
    for cui in present_cuis:
        syns = list(CUI_to_Syn.get(cui, []))
        cui_to_syns[cui] = [clean_natural(s) for s in syns]

    # Initialize encoder
    encoder = TextEncoder(model_name=encoder_name)

    # Prepare batched encoding for all unique mentions
    pairs_list = sorted(unique_pairs)  # deterministic order
    mentions = [m for _, m in pairs_list]
    mention_vecs = encoder.encode(mentions, batch_size=batch_size, tqdm_bar=True)

    # Precompute embeddings for all CUIs' synonyms at once for efficiency
    all_syns = []
    cui_syn_indices = {}
    idx = 0
    for cui, syns in cui_to_syns.items():
        cui_syn_indices[cui] = (idx, idx + len(syns))
        all_syns.extend(syns)
        idx += len(syns)
    all_syn_vecs = encoder.encode(all_syns, batch_size=batch_size, tqdm_bar=True)
    cui_to_cvecs: dict[str, np.ndarray] = {
        cui: all_syn_vecs[start:end] for cui, (start, end) in cui_syn_indices.items()
    }

    # Compute best synonym for each pair using precomputed vectors
    rows = []
    for (cui, mention), m_vec in tqdm(
        zip(pairs_list, mention_vecs), total=len(pairs_list), desc="Match best syn"
    ):
        syns = cui_to_syns[cui]
        syn_vecs = cui_to_cvecs[cui]
        best_syn, best_score = best_by_cosine(
            encoder=encoder,
            mention=mention,
            candidates=syns,
            mention_vec=m_vec,
            candidates_vecs=syn_vecs,
        )
        rows.append({
            "CUI": cui,
            "entity": mention,
            "best_synonym": best_syn,
            "score": best_score,
        })

    return pl.DataFrame(rows)

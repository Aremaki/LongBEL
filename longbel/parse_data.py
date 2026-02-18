import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import joblib
import nltk
import nltk.data
import numpy as np
import polars as pl
from tqdm import tqdm

from longbel.embeddings import TextEncoder, best_by_cosine
from longbel.utils import convert_tsv_as_bigbio


def _process_chunk(chunk, offset, text, sent_spans, nlp, entity_spans=None):
    """Helper to process a chunk of text for sentence tokenization."""
    if not chunk:
        return

    chunk_entity_spans = []
    if entity_spans:
        for estart, eend in entity_spans:
            if estart >= offset and eend <= offset + len(chunk):
                chunk_entity_spans.append((estart - offset, eend - offset))

    last_break = 0
    spans_from_nlp = nlp.span_tokenize(chunk)
    for _, end in spans_from_nlp:
        should_break = True
        if chunk_entity_spans:
            for estart, eend in chunk_entity_spans:
                if estart < end < eend:
                    should_break = False
                    break
        if should_break:
            abs_start = offset + last_break
            abs_end = offset + end
            while abs_start < abs_end and text[abs_start].isspace():
                abs_start += 1
            while abs_end > abs_start and text[abs_end - 1].isspace():
                abs_end -= 1
            if abs_start < abs_end:
                sent_spans.append((abs_start, abs_end))
            last_break = end

    if last_break < len(chunk):
        abs_start = offset + last_break
        abs_end = offset + len(chunk)
        while abs_start < abs_end and text[abs_start].isspace():
            abs_start += 1
        while abs_end > abs_start and text[abs_end - 1].isspace():
            abs_end -= 1
        if abs_start < abs_end:
            sent_spans.append((abs_start, abs_end))


def span_tokenize_with_trailing_newlines(text, nlp, entity_spans=None):
    """
    Tokenize text into sentence spans, treating punctuation and line breaks as sentence boundaries.

    Newlines are NOT included in the resulting spans, and leading/trailing
    whitespace is trimmed so each span covers the sentence content only
    (no space before or after).

    Args:
        text (str): The input passage.
        nlp (PunktSentenceTokenizer): Pre-trained NLTK sentence tokenizer.
        entity_spans (List[Tuple[int, int]], optional): List of (start, end) character
            offsets of entities in the text. If provided, sentence splitting will be
            avoided within these spans.
    """
    sent_spans = []
    last_break = 0

    # Find all potential break points (newlines)
    break_points = [m.start() for m in re.finditer(r"\n+", text)]
    break_points.append(len(text))

    for point in break_points:
        is_in_entity = False
        if entity_spans:
            for estart, eend in entity_spans:
                if estart <= point < eend:
                    is_in_entity = True
                    break

        if not is_in_entity:
            chunk = text[last_break:point]
            offset = last_break
            _process_chunk(chunk, offset, text, sent_spans, nlp, entity_spans)
            last_break = point

    # Process the final chunk if it wasn't handled
    if last_break < len(text):
        chunk = text[last_break:]
        offset = last_break
        _process_chunk(chunk, offset, text, sent_spans, nlp, entity_spans)

    return sent_spans


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


def _insert_entity_markers(
    text: str, spans: list[tuple[int, int]], start_entity: str, end_entity: str
) -> str:
    """Insert entity markers into text using original offsets, handling nested spans.

    The insertion is done in a single pass using start/end events, so offsets
    remain valid even when spans are nested or adjacent.
    """
    if not spans:
        return text

    text_len = len(text)
    starts: dict[int, list[tuple[int, int]]] = {}
    ends: dict[int, list[tuple[int, int]]] = {}

    for start, end in spans:
        if 0 <= start < end <= text_len:
            starts.setdefault(start, []).append((start, end))
            ends.setdefault(end, []).append((start, end))

    if not starts and not ends:
        return text

    positions = sorted(set(starts) | set(ends))
    out: list[str] = []
    last_idx = 0

    for idx in positions:
        if last_idx < idx:
            out.append(text[last_idx:idx])

        # Close inner spans first when multiple end at the same position
        for _ in sorted(ends.get(idx, []), key=lambda x: x[0], reverse=True):
            out.append(end_entity)

        # Open outer spans first when multiple start at the same position
        for _ in sorted(starts.get(idx, []), key=lambda x: x[1], reverse=True):
            out.append(start_entity)

        last_idx = idx

    if last_idx < text_len:
        out.append(text[last_idx:])

    return "".join(out)


def parse_text(
    data,
    start_entity,
    end_entity,
    start_group,
    end_group,
    nlp,
    code_to_title,
    code_to_syn,
    code_to_group,
    cat_to_group,
    sem_to_group,
    transition_verb,
    corrected_code=None,
    selection_method: str = "levenshtein",
    encoder: Optional[TextEncoder] = None,
    tfidf_vectorizer=None,
    best_syn_map: Optional[dict[tuple[str, str], str]] = None,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Create simple (source, target) pairs per entity.

    For each entity in the BigBio page, returns one pair where:
      - source: the sentence text that contains the entity mention
      - target: "<entity> is <annotation>" where <annotation> is the best synonym
        if available (or the normalized id otherwise).
    """
    source_sentences: list[str] = []
    target_sentences: list[str] = []
    tsv_lines: list[dict[str, str]] = []

    # Build a fast lookup of sentence spans per passage
    entity_id = 1
    for passage in data.get("passages", []):
        passage_text = passage["text"][0]
        start_offset_passage = passage["offsets"][0][0]
        end_offset_passage = passage["offsets"][0][1]

        passage_text = clean_natural(passage_text)

        # Collect entity spans to avoid sentence splitting inside them
        entity_spans = []
        for entity in data.get("entities", []):
            for start, end in entity.get("offsets", []):
                if start_offset_passage <= start < end_offset_passage:
                    rel_start = start - start_offset_passage
                    rel_end = end - start_offset_passage
                    entity_spans.append((rel_start, rel_end))

        # Compute sentence spans within the passage text
        sent_spans = span_tokenize_with_trailing_newlines(
            passage_text, nlp, entity_spans=entity_spans
        )

        # Pre-extract sentences with text for quick access
        sentences = [
            (s_start, s_end, passage_text[s_start:s_end])
            for s_start, s_end in sent_spans
        ]

        # Iterate over entities and emit one pair per entity found in this passage
        for entity in data.get("entities", []):
            global_start = entity["offsets"][0][0]
            # Keep only entities whose start falls inside this passage
            if not (start_offset_passage <= global_start < end_offset_passage):
                continue
            rel_start = global_start - start_offset_passage
            entity_text = " ".join(entity["text"])
            entity_text = clean_natural(entity_text)
            if not entity.get("normalized"):
                logging.warning(
                    f"Entity '{' '.join(entity['text'])}' has no code; skipping."
                )
                # No normalized id -> skip (no annotation)
                continue

            normalized_ids = entity["normalized"][0]["db_id"]
            if not normalized_ids:
                logging.warning(
                    f"Entity '{entity_text}' has empty code; skipping entity."
                )
                continue
            normalized_ids = normalized_ids.split("+")  # Handle multiple codes
            annotations = []
            group_annotations = []
            for i, normalized_id in enumerate(normalized_ids):
                if corrected_code and normalized_id in corrected_code:
                    normalized_id = corrected_code[normalized_id]
                    normalized_ids[i] = normalized_id
                    logging.info(
                        f"Corrected code {entity['normalized'][0]['db_id']} -> {normalized_id} for entity '{entity_text}'"
                    )

                possible_syns = None
                annotation = None
                if selection_method == "title":
                    if code_to_title is not None:
                        annotation = code_to_title.get(normalized_id)
                    else:
                        logging.warning(
                            f"Title selection requested but no title mapping available for code {normalized_id} (entity '{entity_text}');"
                        )
                        continue
                # Prefer precomputed best synonyms if provided
                elif selection_method == "embedding":
                    if best_syn_map is not None:
                        pre_key = (normalized_id, entity_text)
                        if pre_key in best_syn_map:
                            annotation = best_syn_map[pre_key]
                    # Otherwise select from possible synonyms
                    if annotation is None and code_to_syn is not None:
                        possible_syns = code_to_syn.get(normalized_id)
                        if possible_syns and encoder is not None:
                            logging.warning(
                                f"No precomputed best synonym map provided; Selecting best synonym by embedding for code {normalized_id} (entity '{entity_text}')"
                            )
                            best_syn, _ = best_by_cosine(
                                encoder=encoder,
                                mention=entity_text,
                                candidates=list(possible_syns),
                            )  # type: ignore
                            annotation = best_syn
                elif selection_method == "tfidf":
                    if code_to_syn is not None:
                        possible_syns = code_to_syn.get(normalized_id)
                    if possible_syns and tfidf_vectorizer is not None:
                        best_idx, best_score = cal_similarity_tfidf(
                            possible_syns, entity_text, tfidf_vectorizer
                        )
                        annotation = possible_syns[best_idx]
                    else:
                        logging.warning(
                            f"TF-IDF selection requested but no synonyms or vectorizer available for code {normalized_id} (entity '{entity_text}');"
                        )
                        continue
                elif selection_method == "levenshtein":
                    if code_to_syn is not None:
                        possible_syns = code_to_syn.get(normalized_id)
                    if possible_syns:
                        # Default to Levenshtein matching (previous behavior)
                        text = entity_text
                        dists = [nltk.edit_distance(text, syn) for syn in possible_syns]
                        best_syn = possible_syns[int(np.argmin(dists))]
                        annotation = best_syn
                if annotation is None:
                    # If no synonyms mapping, skip entity
                    logging.warning(
                        f"No synonyms found for code {normalized_id} (entity '{entity_text}'); skipping entity."
                    )
                    continue

                if isinstance(annotation, str):
                    annotations.append(clean_natural(annotation))

                # Define entity group
                entity_type = entity.get("type")
                groups = code_to_group.get(normalized_id, [])
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
                        f"Group is 'Unknown' for code {normalized_id} and entity type {entity_type}. skipping."
                    )
                    continue

                # Append only if different
                if group not in group_annotations:
                    group_annotations.append(group)
                    # Warning if multiple groups found
                    if len(group_annotations) > 1:
                        logging.warning(
                            f"Multiple groups {group_annotations} found for code {normalized_id} (entity '{entity_text}')"
                        )

            # Merge annotations in a string with | separator
            if not annotations:
                continue
            group_annotation = "<SEP>".join(group_annotations)
            annotation = "<SEP>".join(annotations)

            # Find the sentence that contains the entity start
            sent_text = passage_text
            sent_start_offset = 0
            for s_start, s_end, s_text in sentences:
                if s_start <= rel_start < s_end:
                    sent_text = s_text
                    sent_start_offset = s_start
                    break

            # Add entity markers around all occurrences of the entity
            marked_sent_text = sent_text
            # Get all offsets, convert to relative, and filter for this sentence
            all_spans_in_sent = []
            for off in entity["offsets"]:
                global_start_off, global_end_off = off
                if not (start_offset_passage <= global_start_off < end_offset_passage):
                    continue

                rel_start_off = global_start_off - start_offset_passage
                rel_end_off = global_end_off - start_offset_passage

                start_in_sent = rel_start_off - sent_start_offset
                end_in_sent = rel_end_off - sent_start_offset

                if 0 <= start_in_sent < len(sent_text) and 0 < end_in_sent <= len(
                    sent_text
                ):
                    all_spans_in_sent.append((start_in_sent, end_in_sent))

            # Sort spans in reverse to mark from the end, preventing offset shifts
            all_spans_in_sent.sort(key=lambda x: x[0], reverse=True)

            for start_in_sent, end_in_sent in all_spans_in_sent:
                marked_sent_text = (
                    marked_sent_text[:start_in_sent]
                    + start_entity
                    + marked_sent_text[start_in_sent:end_in_sent]
                    + end_entity
                    + marked_sent_text[end_in_sent:]
                )

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
                "sentence": marked_sent_text,
            }
            entity_id += 1
            tsv_lines.append(tsv_line)
            source_sentences.append(marked_sent_text)
            target_entity_text = (
                start_entity
                + entity_text
                + end_entity
                + start_group
                + group_annotation
                + end_group
            )
            target_sentences.append(
                f"{target_entity_text} {transition_verb} {annotation}"
            )

    return source_sentences, target_sentences, tsv_lines


def parse_text_long(
    data,
    start_entity,
    end_entity,
    start_group,
    end_group,
    code_to_title,
    code_to_syn,
    code_to_group,
    cat_to_group,
    sem_to_group,
    transition_verb,
    corrected_code=None,
    selection_method: str = "levenshtein",
    encoder: Optional[TextEncoder] = None,
    tfidf_vectorizer=None,
    best_syn_map: Optional[dict[tuple[str, str], str]] = None,
) -> tuple[str, str, list[dict[str, str]], list[str], Optional[str]]:
    """Create simple (source, target) pairs per entity.

    For each entity in the BigBio page, returns one pair where:
      - source: the sentence text that contains the entity mention
      - target: "<entity> is <annotation>" where <annotation> is the best synonym
        if available (or the normalized id otherwise).
    """
    target_texts_dict: dict[tuple[tuple[int, int], ...], str] = {}
    target_text: str = ""
    tsv_lines_dict: dict[tuple[tuple[int, int], ...], dict[str, str]] = {}
    tsv_lines: list[dict[str, str]] = []
    source_text: str = ""
    passages = []
    all_annotations = {}
    db_name = None
    for passage in data.get("passages", []):
        passage_text = passage["text"][0]
        start_offset_passage = passage["offsets"][0][0]
        end_offset_passage = passage["offsets"][0][1]

        passage_text = clean_natural(passage_text)
        passages.append((passage_text, passage.get("type")))

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
                    f"Entity '{' '.join(entity['text'])}' has no code; skipping."
                )
                # No normalized id -> skip (no annotation)
                continue

            normalized_ids = entity["normalized"][0]["db_id"]
            if not db_name:
                db_name = entity["normalized"][0].get("db_name")
            if not normalized_ids:
                logging.warning(
                    f"Entity '{entity_text}' has empty code; skipping entity."
                )
                continue
            normalized_ids = normalized_ids.split("+")  # Handle multiple codes
            annotations = []
            group_annotations = []
            for i, normalized_id in enumerate(normalized_ids):
                if corrected_code and normalized_id in corrected_code:
                    normalized_id = corrected_code[normalized_id]
                    normalized_ids[i] = normalized_id
                    logging.info(
                        f"Corrected code {entity['normalized'][0]['db_id']} -> {normalized_id} for entity '{entity_text}'"
                    )

                possible_syns = None
                annotation = None
                # First try to get annotation from the mapping of all annotations
                if normalized_id in all_annotations:
                    annotation = all_annotations[normalized_id]
                # fallback to selection methods if not already found for this code
                else:
                    if selection_method == "title":
                        if code_to_title is not None:
                            annotation = code_to_title.get(normalized_id)
                        else:
                            logging.warning(
                                f"Title selection requested but no title mapping available for code {normalized_id} (entity '{entity_text}');"
                            )
                            continue
                    # Prefer precomputed best synonyms if provided
                    elif selection_method == "embedding":
                        if best_syn_map is not None:
                            pre_key = (normalized_id, entity_text)
                            if pre_key in best_syn_map:
                                annotation = best_syn_map[pre_key]
                        # Otherwise select from possible synonyms
                        if annotation is None and code_to_syn is not None:
                            possible_syns = code_to_syn.get(normalized_id)
                            if possible_syns and encoder is not None:
                                logging.warning(
                                    f"No precomputed best synonym map provided; Selecting best synonym by embedding for code {normalized_id} (entity '{entity_text}')"
                                )
                                best_syn, _ = best_by_cosine(
                                    encoder=encoder,
                                    mention=entity_text,
                                    candidates=list(possible_syns),
                                )  # type: ignore
                                annotation = best_syn
                    elif selection_method == "tfidf":
                        if code_to_syn is not None:
                            possible_syns = code_to_syn.get(normalized_id)
                        if possible_syns and tfidf_vectorizer is not None:
                            best_idx, best_score = cal_similarity_tfidf(
                                possible_syns, entity_text, tfidf_vectorizer
                            )
                            annotation = possible_syns[best_idx]
                        else:
                            logging.warning(
                                f"TF-IDF selection requested but no synonyms or vectorizer available for code {normalized_id} (entity '{entity_text}');"
                            )
                            continue
                    elif selection_method == "levenshtein":
                        if code_to_syn is not None:
                            possible_syns = code_to_syn.get(normalized_id)
                        if possible_syns:
                            # Default to Levenshtein matching (previous behavior)
                            text = entity_text
                            dists = [
                                nltk.edit_distance(text, syn) for syn in possible_syns
                            ]
                            best_syn = possible_syns[int(np.argmin(dists))]
                            annotation = best_syn
                if annotation is None:
                    # If no synonyms mapping, skip entity
                    logging.warning(
                        f"No synonyms found for code {normalized_id} (entity '{entity_text}'); skipping entity."
                    )
                    continue

                if isinstance(annotation, str):
                    annotations.append(clean_natural(annotation))
                    if normalized_id not in all_annotations:
                        all_annotations[normalized_id] = clean_natural(annotation)

                # Define entity group
                entity_type = entity.get("type")
                groups = code_to_group.get(normalized_id, [])
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
                        f"Group is 'Unknown' for code {normalized_id} and entity type {entity_type}. skipping."
                    )
                    continue

                # Append only if different
                if group not in group_annotations:
                    group_annotations.append(group)
                    # Warning if multiple groups found
                    if len(group_annotations) > 1:
                        logging.warning(
                            f"Multiple groups {group_annotations} found for code {normalized_id} (entity '{entity_text}')"
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
            entity_spans.sort(key=lambda x: x[0])
            final_spans = [entity_spans[0][0], entity_spans[-1][1]]
            entity_span_key = tuple(final_spans)
            all_spans.append(final_spans)

            # Emit the pair
            doc_id = data.get("document_id", "")
            tsv_line = {
                "filename": doc_id,
                "label": group_annotation,
                "start_span": final_spans[0],
                "end_span": final_spans[1],
                "span": entity_text,
                "code": "+".join(normalized_ids),
                "semantic_rel": "EXACT" if len(normalized_ids) == 1 else "COMPOSITE",
                "annotation": annotation,
            }
            tsv_lines_dict[entity_span_key] = tsv_line
            target_entity_text = (
                start_entity
                + entity_text
                + end_entity
                + start_group
                + group_annotation
                + end_group
            )
            target_texts_dict[entity_span_key] = (
                f"{target_entity_text} {transition_verb} {annotation}\n"
            )

        # Sort keys to have a deterministic order
        sorted_keys = sorted(target_texts_dict.keys(), key=lambda x: (x[0], x[1]))
        for entity_id, entity_span in enumerate(sorted_keys):
            target_text += target_texts_dict[entity_span]
            tsv_line = tsv_lines_dict[entity_span]
            tsv_line["mention_id"] = f"{data.get('document_id', '')}.{entity_id + 1}"
            tsv_lines.append(tsv_line)
        # Insert all entity markers in a single pass to avoid offset shifts
        passage_text = _insert_entity_markers(
            passage_text, all_spans, start_entity=start_entity, end_entity=end_entity
        )
        if source_text:
            source_text += "\n\n"
        source_text += passage_text

    return source_text, target_text, tsv_lines, passages, db_name


def process_bigbio_dataset(
    bigbio_dataset,
    start_entity,
    end_entity,
    start_group,
    end_group,
    code_to_title,
    code_to_syn,
    code_to_group,
    semantic_info: pl.DataFrame,
    encoder_name=None,
    tfidf_vectorizer_path: Optional[Path] = None,
    corrected_code=None,
    lang: str = "en",
    selection_method: str = "levenshtein",
    best_syn_map: Optional[dict[tuple[str, str], str]] = None,
    long_format: bool = False,
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
    # transition verb and sentence tokenizer depend on language (french, english, spanish)
    if lang == "fr":
        nlp = nltk.data.load("tokenizers/punkt/french.pickle")
        transition_verb = "est"
    elif lang == "es":
        nlp = nltk.data.load("tokenizers/punkt/spanish.pickle")
        transition_verb = "es"
    elif lang == "en":
        nlp = nltk.data.load("tokenizers/punkt/english.pickle")
        transition_verb = "is"
    else:
        raise ValueError(f"Unsupported language '{lang}' for transition verb.")

    target_data = []
    source_data = []
    tsv_data = []
    raw_data = {}
    db_name = None
    processed_bigbio_dataset = None
    if selection_method == "embedding" and encoder_name and best_syn_map is None:
        encoder = TextEncoder(model_name=encoder_name)
        print(f"Using embedding-based selection with encoder '{encoder_name}'.")
    else:
        encoder = None

    tfidf_vectorizer = None
    if selection_method == "tfidf":
        if tfidf_vectorizer_path:
            tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
            print(
                f"Using TF-IDF-based selection with vectorizer at '{tfidf_vectorizer_path}'."
            )
        else:
            raise ValueError(
                "TF-IDF selection method requested but no vectorizer path provided."
            )
    for page in tqdm(bigbio_dataset, total=len(bigbio_dataset)):
        if not long_format:
            source_texts, target_texts, tsv_lines = parse_text(
                page,
                start_entity,
                end_entity,
                start_group,
                end_group,
                nlp,
                code_to_title,
                code_to_syn,
                code_to_group,
                cat_to_group,
                sem_to_group,
                transition_verb,
                corrected_code,
                selection_method,
                encoder,
                tfidf_vectorizer,
                best_syn_map,
            )
            target_data.extend(target_texts)
            source_data.extend(source_texts)
            tsv_data.extend(tsv_lines)
        else:
            source_text, target_text, tsv_lines, passages, page_db_name = (
                parse_text_long(
                    page,
                    start_entity,
                    end_entity,
                    start_group,
                    end_group,
                    code_to_title,
                    code_to_syn,
                    code_to_group,
                    cat_to_group,
                    sem_to_group,
                    transition_verb,
                    corrected_code,
                    selection_method,
                    encoder,
                    tfidf_vectorizer,
                    best_syn_map,
                )
            )
            target_data.append(target_text)
            source_data.append(source_text)
            raw_data[page["document_id"]] = passages
            if not db_name:
                db_name = page_db_name
            tsv_data.extend(tsv_lines)
            processed_bigbio_dataset = convert_tsv_as_bigbio(
                tsv_data=tsv_data, raw_data=raw_data, db_name=db_name
            )

    return source_data, target_data, tsv_data, processed_bigbio_dataset


def compute_best_synonym_df(
    bigbio_dataset: Iterable[dict],
    code_to_syn: dict[str, Iterable[str]],
    encoder_name: str = "encoder/coder-all",
    batch_size: int = 4096,
    corrected_code: Optional[dict[str, str]] = None,
) -> "pl.DataFrame":
    """Precompute best synonyms per unique (code, entity) using batched embeddings.

    Returns a DataFrame with columns: [code, entity, best_synonym].

    Notes
    -----
    - Deduplicates pairs by (code, entity) for efficiency.
    - Uses cosine similarity via best_by_cosine with precomputed vectors.
    """
    # Gather unique (code, entity) pairs and the set of codes present in dataset
    unique_pairs: set[tuple[str, str]] = set()
    present_codes: set[str] = set()
    for page in bigbio_dataset:
        for ent in page.get("entities", []):
            if not ent.get("normalized"):
                continue
            code = ent["normalized"][0]["db_id"]
            if corrected_code and code in corrected_code:
                code = corrected_code[code]
            mention = clean_natural(" ".join(ent["text"]))
            unique_pairs.add((code, mention))
            present_codes.add(code)

    # Build per-code synonym lists (ensure non-empty)
    code_to_syns: dict[str, list[str]] = {}
    for code in present_codes:
        syns = list(code_to_syn.get(code, []))
        code_to_syns[code] = [clean_natural(s) for s in syns]

    # Initialize encoder
    encoder = TextEncoder(model_name=encoder_name)

    # Prepare batched encoding for all unique mentions
    pairs_list = sorted(unique_pairs)  # deterministic order
    mentions = [m for _, m in pairs_list]
    mention_vecs = encoder.encode(mentions, batch_size=batch_size, tqdm_bar=True)

    # Precompute embeddings for all codes' synonyms at once for efficiency
    all_syns = []
    code_syn_indices = {}
    idx = 0
    for code, syns in code_to_syns.items():
        code_syn_indices[code] = (idx, idx + len(syns))
        all_syns.extend(syns)
        idx += len(syns)
    all_syn_vecs = encoder.encode(all_syns, batch_size=batch_size, tqdm_bar=True)
    code_to_cvecs: dict[str, np.ndarray] = {
        code: all_syn_vecs[start:end] for code, (start, end) in code_syn_indices.items()
    }

    # Compute best synonym for each pair using precomputed vectors
    rows = []
    for (code, mention), m_vec in tqdm(
        zip(pairs_list, mention_vecs), total=len(pairs_list), desc="Match best syn"
    ):
        syns = code_to_syns[code]
        syn_vecs = code_to_cvecs[code]
        best_syn, best_score = best_by_cosine(
            encoder=encoder,
            mention=mention,
            candidates=syns,
            mention_vec=m_vec,
            candidates_vecs=syn_vecs,
        )
        rows.append({
            "code": code,
            "entity": mention,
            "best_synonym": best_syn,
            "score": best_score,
        })

    return pl.DataFrame(rows)

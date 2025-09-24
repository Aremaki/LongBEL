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

from syncabel.embeddings import TextEncoder, best_by_cosine


def span_tokenize_with_trailing_newlines(text, nlp):
    """
    Tokenize text into sentence spans, treating punctuation and line breaks as sentence boundaries.

    Newlines are NOT included in the resulting spans, and leading/trailing
    whitespace is trimmed so each span covers the sentence content only
    (no space before or after).

    Args:
        text (str): The input passage.
        nlp (PunktSentenceTokenizer): Pre-trained NLTK sentence tokenizer.

    Returns:
        List[Tuple[int, int]]: List of (start, end) indices for each sentence
        within the original text, without surrounding whitespace/newlines.
    """
    sent_spans = []
    offset = 0

    # Split text by lines, but keep \n in the chunks
    # Use regex to keep trailing newlines
    chunks = re.split(r"(\n+)", text)

    current_chunk = ""
    for part in chunks:
        if part == "":
            continue
        if re.fullmatch(r"\n+", part):
            # Newline sequence: treat as a boundary, but DO NOT include it in spans.
            if current_chunk:
                for start, end in nlp.span_tokenize(current_chunk):
                    abs_start = offset + start
                    abs_end = offset + end
                    # Trim leading/trailing whitespace from the original text
                    while abs_start < abs_end and text[abs_start].isspace():
                        abs_start += 1
                    while abs_end > abs_start and text[abs_end - 1].isspace():
                        abs_end -= 1
                    if abs_start < abs_end:
                        sent_spans.append((abs_start, abs_end))
            # Advance offset over both the chunk and the newline(s)
            offset += len(current_chunk) + len(part)
            current_chunk = ""
        else:
            # Non-newline text: accumulate until a newline or the end
            current_chunk += part

    # Handle any remaining chunk
    if current_chunk:
        for start, end in nlp.span_tokenize(current_chunk):
            abs_start = offset + start
            abs_end = offset + end
            # Trim leading/trailing whitespace from the original text
            while abs_start < abs_end and text[abs_start].isspace():
                abs_start += 1
            while abs_end > abs_start and text[abs_end - 1].isspace():
                abs_end -= 1
            if abs_start < abs_end:
                sent_spans.append((abs_start, abs_end))
        offset += len(current_chunk)

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


def parse_text(
    data,
    start_entity,
    end_entity,
    start_group,
    end_group,
    nlp,
    CUI_to_Title,
    CUI_to_Syn,
    cat_to_group,
    sem_to_group,
    natural=False,
    corrected_cui=None,
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
    source_sentences: list[str] = []
    source_with_group_sentences: list[str] = []
    target_sentences: list[str] = []

    # Build a fast lookup of sentence spans per passage
    for passage in data.get("passages", []):
        passage_text = passage["text"][0]
        start_offset_passage = passage["offsets"][0][0]
        end_offset_passage = passage["offsets"][0][1]

        if natural:
            passage_text = clean_natural(passage_text)

        # Compute sentence spans within the passage text
        sent_spans = span_tokenize_with_trailing_newlines(passage_text, nlp)

        # Pre-extract sentences with text for quick access
        sentences = [
            (s_start, s_end, passage_text[s_start:s_end])
            for s_start, s_end in sent_spans
        ]

        # Iterate over entities and emit one pair per entity found in this passage
        for entity in data.get("entities", []):
            if not entity.get("normalized"):
                logging.warning(
                    f"Entity '{' '.join(entity['text'])}' has no CUI; skipping."
                )
                # No normalized id -> skip (no annotation)
                continue

            global_start = entity["offsets"][0][0]
            # Keep only entities whose start falls inside this passage
            if not (start_offset_passage <= global_start < end_offset_passage):
                continue

            rel_start = global_start - start_offset_passage
            # rel_end isn't strictly required for sentence selection, but computed for completeness
            # rel_end = global_end - start_offset_passage

            entity_text = " ".join(entity["text"])
            if natural:
                entity_text = clean_natural(entity_text)

            normalized_id = entity["normalized"][0]["db_id"]
            if not normalized_id:
                logging.warning(
                    f"Entity '{entity_text}' has empty CUI; skipping entity."
                )
                continue
            if corrected_cui and normalized_id in corrected_cui:
                normalized_id = corrected_cui[normalized_id]
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

            if natural and isinstance(annotation, str):
                annotation = clean_natural(annotation)

            # Define CUI group
            entity_type = entity["type"]
            if entity_type in cat_to_group.keys():
                group = cat_to_group[entity_type]
            elif entity_type in sem_to_group.keys():
                group = sem_to_group[entity_type]
            else:
                group = "Unknown"
                logging.info(f"No group found for entity type {entity_type}.")

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

            marked_with_group_text = marked_sent_text
            for i, (start_in_sent, end_in_sent) in enumerate(all_spans_in_sent):
                if i == 0:
                    marked_with_group_text = (
                        marked_with_group_text[:start_in_sent]
                        + start_entity
                        + marked_with_group_text[start_in_sent:end_in_sent]
                        + end_entity
                        + start_group
                        + group
                        + end_group
                        + marked_with_group_text[end_in_sent:]
                    )
                else:
                    marked_with_group_text = (
                        marked_with_group_text[:start_in_sent]
                        + start_entity
                        + marked_with_group_text[start_in_sent:end_in_sent]
                        + end_entity
                        + marked_with_group_text[end_in_sent:]
                    )
                marked_sent_text = (
                    marked_sent_text[:start_in_sent]
                    + start_entity
                    + marked_sent_text[start_in_sent:end_in_sent]
                    + end_entity
                    + marked_sent_text[end_in_sent:]
                )
            marked_sent_text += f"</s>{entity_text} is"
            marked_with_group_text += f"</s>{entity_text} is"

            # Emit the pair
            source_sentences.append(marked_sent_text)
            source_with_group_sentences.append(marked_with_group_text)
            target_sentences.append(annotation)

    return source_sentences, source_with_group_sentences, target_sentences


def process_bigbio_dataset(
    bigbio_dataset,
    start_entity,
    end_entity,
    start_group,
    end_group,
    CUI_to_Title,
    CUI_to_Syn,
    semantic_info: pl.DataFrame,
    natural=False,
    encoder_name=None,
    tfidf_vectorizer_path: Optional[Path] = None,
    corrected_cui=None,
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
    # Load sentence tokenizer for requested language (default english).
    # Falls back to english if the specified model is unavailable.
    try:
        nlp = nltk.data.load(f"tokenizers/punkt/{language}.pickle")
    except LookupError:
        print(f"⚠️ Punkt model for '{language}' not found; falling back to English.")
        nlp = nltk.data.load("tokenizers/punkt/english.pickle")
    target_data = []
    source_data = []
    source_with_group_data = []
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
        source_texts, source_with_group_texts, target_texts = parse_text(
            page,
            start_entity,
            end_entity,
            start_group,
            end_group,
            nlp,
            CUI_to_Title,
            CUI_to_Syn,
            cat_to_group,
            sem_to_group,
            natural,
            corrected_cui,
            selection_method,
            encoder,
            tfidf_vectorizer,
            best_syn_map,
        )
        # Each entity yields one pair; extend the global lists accordingly.
        target_data.extend(target_texts)
        source_data.extend(source_texts)
        source_with_group_data.extend(source_with_group_texts)

    return source_data, source_with_group_data, target_data


def compute_best_synonym_df(
    bigbio_dataset: Iterable[dict],
    CUI_to_Syn: dict[str, Iterable[str]],
    encoder_name: str = "encoder/coder-all",
    batch_size: int = 4096,
    corrected_cui: Optional[dict[str, str]] = None,
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
            if corrected_cui and cui in corrected_cui:
                cui = corrected_cui[cui]
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

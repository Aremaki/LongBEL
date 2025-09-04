import re
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import nltk
import nltk.data
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from syncabel.embeddings import TextEncoder, best_by_cosine


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
    start_tag,
    end_tag,
    nlp,
    CUI_to_Syn=None,
    Syn_to_annotation=None,
    natural=False,
    tokenizer=None,
    corrected_cui=None,
    selection_method: str = "levenshtein",
    encoder: Optional[TextEncoder] = None,
    best_syn_map: Optional[dict[tuple[str, str], str]] = None,
):
    """Create simple (source, target) pairs per entity.

    For each entity in the BigBio page, returns one pair where:
      - source: the sentence text that contains the entity mention
      - target: "<entity> is <annotation>" where <annotation> is the best synonym
        if available (or the normalized id otherwise).

    Notes
    -----
    - start_entity/end_entity/start_tag/end_tag parameters are kept for
      signature compatibility but are not used in this simplified mode.
    - Tokenizer is not required.
    """
    source_sentences: list[str] = []
    target_sentences: list[str] = []

    # Build a fast lookup of sentence spans per passage
    for passage in data.get("passages", []):
        passage_text = passage["text"][0]
        start_offset_passage = passage["offsets"][0][0]
        end_offset_passage = passage["offsets"][0][1]

        if natural:
            passage_text = clean_natural(passage_text)

        # Compute sentence spans within the passage text
        sent_spans = list(nlp.span_tokenize(passage_text))  # type: ignore[attr-defined]

        # Pre-extract sentences with text for quick access
        sentences = [
            (s_start, s_end, passage_text[s_start:s_end])
            for s_start, s_end in sent_spans
        ]

        # Iterate over entities and emit one pair per entity found in this passage
        for entity in data.get("entities", []):
            if not entity.get("normalized"):
                # No normalized id -> skip (no annotation)
                continue

            global_start = entity["offsets"][0][0]
            # Keep only entities whose start falls inside this passage
            if not (start_offset_passage <= global_start < end_offset_passage):
                continue

            rel_start = global_start - start_offset_passage
            # rel_end isn't strictly required for sentence selection, but computed for completeness
            # rel_end = global_end - start_offset_passage

            entity_text = entity["text"][0]
            if natural:
                entity_text = clean_natural(entity_text)

            normalized_id = entity["normalized"][0]["db_id"]
            if corrected_cui and normalized_id in corrected_cui:
                normalized_id = corrected_cui[normalized_id]

            annotation = None
            if CUI_to_Syn is not None:
                possible_syns = CUI_to_Syn.get(normalized_id)
            else:
                possible_syns = None

            # Prefer precomputed best synonyms if provided
            if best_syn_map is not None:
                pre_key = (normalized_id, entity_text)
                if pre_key in best_syn_map:
                    annotation = best_syn_map[pre_key]
            # Otherwise select from possible synonyms
            if annotation is None and possible_syns:
                if selection_method == "embedding" and encoder is not None:
                    best_syn, _ = best_by_cosine(
                        encoder=encoder,
                        mention=entity_text,
                        candidates=list(possible_syns),
                    )  # type: ignore
                    annotation = best_syn if best_syn is not None else normalized_id
                else:
                    # Default to Levenshtein matching (previous behavior)
                    text = entity_text
                    dists = [nltk.edit_distance(text, syn) for syn in possible_syns]
                    best_syn = possible_syns[int(np.argmin(dists))]
                    annotation = best_syn
            else:
                # If no synonyms mapping, fall back to the normalized id
                annotation = normalized_id

            if natural and isinstance(annotation, str):
                annotation = clean_natural(annotation)

            # Find the sentence that contains the entity start
            sent_text = passage_text
            for s_start, s_end, s_text in sentences:
                if s_start <= rel_start < s_end:
                    sent_text = s_text
                    break

            # Emit the pair
            source_sentences.append(sent_text)
            target_sentences.append(f"{entity_text} is {annotation}")

    return source_sentences, target_sentences


def process_bigbio_dataset(
    bigbio_dataset,
    start_entity,
    end_entity,
    start_tag,
    end_tag,
    CUI_to_Syn=None,
    Syn_to_annotation=None,
    natural=False,
    model_name=None,
    encoder_name=None,
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
    # Load sentence tokenizer for requested language (default english).
    # Falls back to english if the specified model is unavailable.
    try:
        nlp = nltk.data.load(f"tokenizers/punkt/{language}.pickle")
    except LookupError:
        print(f"⚠️ Punkt model for '{language}' not found; falling back to English.")
        nlp = nltk.data.load("tokenizers/punkt/english.pickle")
    # Tokenizer is optional for the simplified pipeline; only load if a model name is provided.
    tokenizer = None
    if model_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, add_prefix_space=False
            )
        except Exception as hub_err:  # pragma: no cover - network / availability branch
            local_dir = Path("models") / str(model_name)
            if local_dir.exists():
                print(
                    f"⚠️ Hub load failed for '{model_name}' ({hub_err}); falling back to local path {local_dir}."
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    str(local_dir), add_prefix_space=False
                )
            else:
                print(
                    f"⚠️ Failed to load tokenizer for '{model_name}' from hub and no local fallback at {local_dir}. Proceeding without a tokenizer."
                )
                tokenizer = None
        if tokenizer is not None:
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        start_entity,
                        end_entity,
                        start_tag,
                        end_tag,
                    ]
                },
                replace_additional_special_tokens=False,
            )
    target_data = []
    source_data = []
    if selection_method == "embedding" and encoder_name and best_syn_map is None:
        encoder = TextEncoder(model_name=encoder_name)
        print(f"Using embedding-based selection with encoder '{encoder_name}'.")
    else:
        encoder = None
    for page in tqdm(bigbio_dataset, total=len(bigbio_dataset)):
        source_texts, target_texts = parse_text(
            page,
            start_entity,
            end_entity,
            start_tag,
            end_tag,
            nlp,
            CUI_to_Syn,
            Syn_to_annotation,
            natural,
            tokenizer,
            corrected_cui,
            selection_method,
            encoder,
            best_syn_map,
        )
        # Each entity yields one pair; extend the global lists accordingly.
        target_data.extend(target_texts)
        source_data.extend(source_texts)
    return source_data, target_data


def compute_best_synonym_df(
    bigbio_dataset: Iterable[dict],
    CUI_to_Syn: dict[str, Iterable[str]],
    encoder_name: str = "GanjinZero/coder-all",
    batch_size: int = 4096,
    corrected_cui: Optional[dict[str, str]] = None,
    language: str = "english",
) -> "pd.DataFrame":
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
            mention = clean_natural(ent["text"][0])
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

    return pd.DataFrame(rows)


def load_data(source_data, target_data, nlp, tokenizer, max_length=512):
    """
    Load and preprocess source and target data, ensuring that the target text is split into passages
    while maintaining token limit constraints and balanced entity markers.

    Args:
        source_data (list of str): List of source texts.
        target_data (list of str): List of target texts with annotated entities.
        nlp (nltk.tokenize): NLTK tokenizer with a `.tokenize()` method (e.g., nltk.sent_tokenize).
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for computing token lengths.
        max_length (int, optional): Maximum allowed token length for each target passage. Default is 512.

    Returns:
        dict: A dictionary containing:
            - "source" (list of str): Source passages with entity annotations removed.
            - "target" (list of str): Target passages containing entity annotations.
    """
    data = {"source": [], "target": []}
    for _, target in zip(source_data, target_data):
        tokens = 0
        target_passage = ""
        target_doc = nlp.tokenize(target)
        for sent in target_doc:
            sent_tokens = len(tokenizer.encode(sent))
            tokens += sent_tokens
            if (
                tokens < max_length
                or (target_passage.count("<s_e>") != target_passage.count("<e_e>"))
                or (target_passage.count("<s_m>") != target_passage.count("<e_e>"))
            ):
                target_passage += sent + " "
            else:
                target_passage = target_passage[:-1]
                data["target"].append(target_passage)
                source_passage = re.sub(r"<s_m>\s|\s<s_e>.*?<e_e>", "", target_passage)
                data["source"].append(source_passage)
                target_passage = sent + " "
                tokens = sent_tokens
        if tokens > 20:
            target_passage = target_passage[:-1]
            data["target"].append(target_passage)
            source_passage = re.sub(r"<s_m>\s|\s<s_e>.*?<e_e>", "", target_passage)
            data["source"].append(source_passage)
        elif tokens < 20:
            target_passage = data["target"][-1] + " " + target_passage[:-1]
            data["target"][-1] = target_passage
            source_passage = re.sub(r"<s_m>\s|\s<s_e>.*?<e_e>", "", target_passage)
            data["source"][-1] = source_passage

    return data

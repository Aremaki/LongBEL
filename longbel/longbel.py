"""
Core models for LongBEL
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pickle
import re
from typing import Optional

import nltk
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Display INFO and above
    format="%(levelname)s - %(message)s",
)


def get_prefix_allowed_tokens_fn(
    model,
    sources: list[str],
    prefix_templates: list[str],
    sem_groups: list[str],
    multiple_answers: bool = False,
):
    candidates_trie = model.candidate_trie  # type: ignore
    sep_token_id = model.tokenizer.sep_token_id
    eos_token_id = model.tokenizer.eos_token_id
    pad_token_id = model.tokenizer.pad_token_id
    plus_token_id = model.tokenizer.convert_tokens_to_ids("<+>")  # type: ignore
    prefix_templates = [
        model.tokenizer.encode(prefix, add_special_tokens=False)
        for prefix in prefix_templates
    ]

    def prefix_allowed_tokens_fn(batch_id, sent):
        sent = sent.tolist()
        if len(sent) > 1 and sent[-1] in [eos_token_id, pad_token_id, sep_token_id]:
            return [sep_token_id, pad_token_id, eos_token_id]
        prefix = prefix_templates[batch_id]
        # Remove the prefix from the sent
        index_sep = len(sent) - 1 - sent[::-1].index(sep_token_id)
        sent = sent[index_sep + 1 :]
        # Check if the prefix is present
        prefix_len = len(prefix)
        if sent[:prefix_len] == prefix:
            sent = sent[prefix_len - 1 :]
        else:
            raise ValueError("Prefix not found in the generated sentence.")
        sem_group = sem_groups[batch_id]
        # Remove everything up to last sep_token_id and add prefix and tgt_lang_id
        if multiple_answers and plus_token_id in sent:
            index_plus = len(sent) - 1 - sent[::-1].index(plus_token_id)
            # Start fresh with decoder start
            if index_plus == len(sent) - 1:
                sent = prefix[-1:]
            # If there are tokens after the last plus_token_id, keep them
            else:
                sent = prefix[-1:] + sent[index_plus + 1 :]
        trie_out = candidates_trie[
            sem_group  # type: ignore
        ].get(sent)
        if eos_token_id in trie_out:
            trie_out += [sep_token_id]
            if multiple_answers:
                trie_out += [plus_token_id]
        elif not trie_out:
            return [sep_token_id, pad_token_id, eos_token_id]
        return trie_out

    return prefix_allowed_tokens_fn


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


def parse_text_long(
    data,
    start_entity,
    end_entity,
    start_group,
    end_group,
    verb,
) -> tuple[list[str], list[dict[str, str]]]:
    """Create simple (source, target) pairs per entity.

    For each entity in the BigBio page, returns one pair where:
      - source: the sentence text that contains the entity mention
      - target: "<entity> is <annotation>" where <annotation> is the best synonym
        if available (or the normalized id otherwise).
    """
    target_texts_dict: dict[tuple[tuple[int, int], ...], str] = {}
    target_texts: list = []
    tsv_lines_dict: dict[tuple[tuple[int, int], ...], dict[str, str]] = {}
    tsv_lines: list[dict[str, str]] = []
    source_text: str = ""
    for passage in data.get("passages", []):
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

            # Define entity group
            entity_group = entity.get("type")

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
                "doc_id": doc_id,
                "semantic_group": entity_group,
                "start_span": final_spans[0],
                "end_span": final_spans[1],
                "mention": entity_text,
            }
            if entity.get("normalized"):
                tsv_line["gold_code"] = entity["normalized"][0]["db_id"]
            tsv_lines_dict[entity_span_key] = tsv_line
            target_entity_text = (
                start_entity
                + entity_text
                + end_entity
                + start_group
                + entity_group
                + end_group
            )
            target_texts_dict[entity_span_key] = f"{target_entity_text} {verb}"

        # Insert all entity markers in a single pass to avoid offset shifts
        passage_text = _insert_entity_markers(
            passage_text, all_spans, start_entity=start_entity, end_entity=end_entity
        )
        source_text += passage_text.rstrip("\n") + "\n"
    source_text += "<SEP>"
    # Sort keys to have a deterministic order
    sorted_keys = sorted(target_texts_dict.keys(), key=lambda x: (x[0], x[1]))
    for entity_id, entity_span in enumerate(sorted_keys):
        target_texts.append(target_texts_dict[entity_span])
        tsv_line = tsv_lines_dict[entity_span]
        tsv_line["mention_id"] = f"{data.get('document_id', '')}.{entity_id + 1}"
        tsv_lines.append(tsv_line)

    all_inputs = [source_text + target_texts[0]]
    for i in range(len(target_texts) - 1):
        all_inputs.append(target_texts[i + 1])

    return all_inputs, tsv_lines


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


def parse_text(
    data,
    start_entity,
    end_entity,
    start_group,
    end_group,
    nlp,
    verb,
) -> tuple[list[str], list[dict[str, str]]]:
    """Create simple (source, target) pairs per entity.

    For each entity in the BigBio page, returns one pair where:
      - source: the sentence text that contains the entity mention
      - target: "<entity> is <annotation>" where <annotation> is the best synonym
        if available (or the normalized id otherwise).
    """
    source_sentences: dict[tuple[tuple[int, int], ...], str] = {}
    target_sentences: dict[tuple[tuple[int, int], ...], str] = {}
    tsv_lines: list[dict[str, str]] = []
    tsv_lines_dict: dict[tuple[tuple[int, int], ...], dict[str, str]] = {}
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
            # min and max of all entity offsets to get the global span of the entity for filtering sentences
            global_start = min(off[0] for off in entity["offsets"])
            global_end = max(off[1] for off in entity["offsets"])
            # Keep only entities whose start falls inside this passage
            if not (start_offset_passage <= global_start < end_offset_passage):
                continue
            rel_start = global_start - start_offset_passage
            entity_text = " ".join(entity["text"])
            entity_text = clean_natural(entity_text)

            # Define entity group
            entity_group = entity.get("type")

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
                    + "<SEP>"
                )

            # Emit the pair
            doc_id = data.get("document_id", "")
            tsv_line = {
                "doc_id": doc_id,
                "semantic_group": entity_group,
                "start_span": global_start,
                "end_span": global_end,
                "mention": entity_text,
            }
            if entity.get("normalized"):
                tsv_line["gold_code"] = entity["normalized"][0]["db_id"]
            tsv_lines_dict[(global_start, global_end)] = tsv_line
            source_sentences[(global_start, global_end)] = marked_sent_text
            target_entity_text = (
                start_entity
                + entity_text
                + end_entity
                + start_group
                + entity_group
                + end_group
            )
            target_sentences[(global_start, global_end)] = (
                f"{target_entity_text} {verb}"
            )

    # Sort keys to have a deterministic order
    all_inputs = []
    sorted_keys = sorted(tsv_lines_dict.keys(), key=lambda x: (x[0], x[1]))
    for entity_id, entity_span in enumerate(sorted_keys):
        tsv_line = tsv_lines_dict[entity_span]
        source = source_sentences[entity_span]
        target = target_sentences[entity_span]
        tsv_line["mention_id"] = f"{data.get('document_id', '')}.{entity_id + 1}"
        tsv_lines.append(tsv_line)
        all_inputs.append(source + target)

    return all_inputs, tsv_lines


def parse_prediction(
    outputs: list[str],
    sem_groups: list[str],
    verb: str,
    text_to_code: Optional[dict[str, dict[str, str]]] = None,
    multiple_answers: bool = False,
) -> tuple[list[str], list[str]]:
    codes = []
    predictions = []
    for output, group in zip(outputs, sem_groups):
        splits = output.split("} " + verb)  # type: ignore
        if len(splits) > 1 and splits[-1].strip():
            prediction = splits[-1].strip().replace("<SEP>", "")
            if text_to_code:
                if multiple_answers:
                    prediction_list = prediction.split("<+>")  # type: ignore
                    code_list = []
                    for pred in prediction_list:
                        code_list.append(
                            text_to_code[group].get(pred.strip(), "NO_CODE")
                        )
                    code = "+".join(code_list)
                else:
                    code = text_to_code[group].get(prediction, "NO_CODE")
            else:
                code = "NO_CODE"
        else:
            print(
                "IndexError: splitting failed or empty prediction, adding empty string as prediction."
            )
            print(f"Full text: {output}")  # type: ignore
            prediction = "NO_PREDICTION"
            code = "NO_CODE"
        codes.append(code)
        predictions.append(prediction)
    return codes, predictions


def compute_score(outputs, tokenizer, prefix_len=0):
    sequences = outputs.sequences  # (N, seq_len)
    scores = outputs.scores  # list length T = # generated tokens

    N, total_len = sequences.shape
    T = len(scores)

    # keep only the generated part (completion)
    sequences = sequences[:, prefix_len : prefix_len + T]

    # Make sure score is not longer than sequences
    if len(scores) > sequences.size(1):
        scores = scores[: sequences.size(1)]

    # Compute as usual but now only for completion tokens
    mask = (
        (sequences != tokenizer.pad_token_id)
        & (sequences != tokenizer.eos_token_id)
        & (sequences != tokenizer.bos_token_id)
    )

    # log-prob for each generated token
    logprob_steps = []
    for t, logits in enumerate(scores):
        log_probs_t = F.log_softmax(logits, dim=-1)
        token_t = sequences[:, t]
        idx = torch.arange(N)
        logprob_steps.append(log_probs_t[idx, token_t])

    logprobs = torch.stack(logprob_steps, dim=1)
    logprobs.masked_fill_(~mask, 0)

    lengths = mask.sum(dim=1).clamp(min=1)
    confidence = torch.exp(logprobs.sum(dim=1) / lengths)

    return confidence.tolist()


def skip_undesired_tokens(outputs, tokenizer):
    sep_token = "<SEP>"
    plus_token = "<+>"
    # Build the list of special tokens to remove
    tokens_to_remove = tokenizer.all_special_tokens[:2]

    cleaned_outputs = []
    for sequence in outputs:
        # Remove undesired special tokens
        for token in tokens_to_remove:
            sequence = sequence.replace(token, "")

        # Remove spaces *immediately* after the sep_token (e.g. "<sep>  text" → "<sep>text")
        if sep_token:
            sequence = re.sub(rf"({re.escape(sep_token)})\s+", r"\1", sequence)
        if plus_token:
            sequence = re.sub(rf"({re.escape(plus_token)})\s+", r"\1", sequence)

        cleaned_outputs.append(sequence.strip())

    return cleaned_outputs


class _LongBELHubInterface:
    def sample(
        self,
        bigbio_pages: list[dict],  # type: ignore
        num_beams: int = 5,
        constrained: bool = True,
        multiple_answers: bool = False,
        batch_size: int = 8,
        start_entity: str = "[",
        end_entity: str = "]",
        start_group: str = "{",
        end_group: str = "}",
        show_progress: bool = True,
        long_format: bool = True,
        **kwargs,
    ) -> list[list[dict[str, str]]]:
        # Prepare input batch
        if self.lang == "fr":  # type: ignore
            nlp = nltk.data.load("tokenizers/punkt/french.pickle")
            verb = "est"
        elif self.lang == "en":  # type: ignore
            nlp = nltk.data.load("tokenizers/punkt/english.pickle")
            verb = "is"
        elif self.lang == "es":  # type: ignore
            nlp = nltk.data.load("tokenizers/punkt/spanish.pickle")
            verb = "es"
        else:
            raise ValueError(f"Unsupported language: {self.lang}")  # type: ignore

        print(
            f"Starting sampling on {len(bigbio_pages)} pages (lang={getattr(self, 'lang', 'unknown')}, constrained={constrained}, beams={num_beams}, batch_size={batch_size})"
        )

        def _progress(iterable, desc: str, total: Optional[int] = None):
            if show_progress:
                return tqdm(iterable, desc=desc, total=total)
            return iterable

        all_outputs = []
        all_examples = []
        all_entities_info = []
        for data in bigbio_pages:
            if long_format:
                examples, entities_info = parse_text_long(
                    data=data,
                    start_entity=start_entity,
                    end_entity=end_entity,
                    start_group=start_group,
                    end_group=end_group,
                    verb=verb,
                )
            else:
                examples, entities_info = parse_text(
                    data=data,
                    start_entity=start_entity,
                    end_entity=end_entity,
                    start_group=start_group,
                    end_group=end_group,
                    nlp=nlp,  # type: ignore
                    verb=verb,
                )
            all_examples.append(examples)
            all_entities_info.append(entities_info)

        total_batches = (len(all_examples) + batch_size - 1) // batch_size
        print(
            f"Input preparation completed. Running generation on {total_batches} batches."
        )
        for i in range(0, len(all_examples), batch_size):
            batch_examples = all_examples[i : i + batch_size]
            batch_entities = all_entities_info[i : i + batch_size]
            sentences = dict.fromkeys(range(len(batch_examples)), "")
            max_sentences = max(len(batch) for batch in batch_examples)
            for sent_id in _progress(
                range(max_sentences),
                desc=f"Batch {i // batch_size + 1}/{total_batches}",
                total=max_sentences,
            ):
                sem_groups = []
                mentions = []
                doc_ids = []
                mentions_id = []
                prefix_templates = []
                gold_codes = []
                start_spans = []
                end_spans = []
                for batch_id, (example, entity) in enumerate(
                    zip(batch_examples, batch_entities)
                ):
                    if sent_id < len(example):
                        if long_format:
                            sentences[batch_id] += example[sent_id]
                        else:
                            sentences[batch_id] = example[sent_id]
                        sem_groups.append(entity[sent_id]["semantic_group"])
                        mentions_id.append(entity[sent_id]["mention_id"])
                        mentions.append(entity[sent_id]["mention"])
                        doc_ids.append(entity[sent_id]["doc_id"])
                        start_spans.append(entity[sent_id]["start_span"])
                        end_spans.append(entity[sent_id]["end_span"])
                        gold_codes.append(entity[sent_id].get("gold_code", None))  # type: ignore
                        prefix_templates.append(
                            f"[{entity[sent_id]['mention']}]{{{entity[sent_id]['semantic_group']}}} {verb}"
                        )
                    # Remove the sentence
                    else:
                        sentences.pop(batch_id, None)

                # Encode input batch
                input_sentences = list(sentences.values())
                batch_ids = list(sentences.keys())
                input_args = {
                    k: v.to(self.device)  # type: ignore
                    for k, v in self.tokenizer.batch_encode_plus(  # type: ignore
                        input_sentences, padding="longest", return_tensors="pt"
                    ).items()
                }

                # Constrained decoding
                prefix_allowed_tokens_fn = None
                if constrained:
                    if self.candidate_trie is None:  # type: ignore
                        raise ValueError(
                            "candidate_trie is not loaded in the model. Use constrained=False."
                        )
                    prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
                        model=self,
                        sources=input_sentences,
                        prefix_templates=prefix_templates,
                        sem_groups=sem_groups,
                        multiple_answers=multiple_answers,
                    )
                outputs = self.generate(  # type: ignore
                    **input_args,
                    max_new_tokens=128,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    eos_token_id=self.tokenizer.sep_token_id,  # type: ignore
                    **kwargs,
                )
                decoded_sequences = self.tokenizer.batch_decode(  # type: ignore
                    outputs.sequences,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
                cleaned_output_sequences = skip_undesired_tokens(
                    decoded_sequences,
                    self.tokenizer,  # type: ignore
                )
                # Update sentences with the cleaned outputs
                if long_format:
                    for i, batch_id in enumerate(batch_ids):
                        clean_sentence = cleaned_output_sequences[num_beams * i]
                        clean_sentence = clean_sentence.rstrip("<SEP>") + "<SEP>"
                        sentences[batch_id] = clean_sentence

                prefix_len = input_args["input_ids"].size(1)

                # Duplicate sem_groups and mentions for each beam
                sem_groups = [x for x in sem_groups for _ in range(num_beams)]
                mentions = [x for x in mentions for _ in range(num_beams)]
                mentions_id = [x for x in mentions_id for _ in range(num_beams)]
                gold_codes = [x for x in gold_codes for _ in range(num_beams)]  # type: ignore
                start_spans = [x for x in start_spans for _ in range(num_beams)]
                end_spans = [x for x in end_spans for _ in range(num_beams)]
                doc_ids = [x for x in doc_ids for _ in range(num_beams)]
                # Parse predictions
                codes, predictions = parse_prediction(
                    cleaned_output_sequences,
                    sem_groups,
                    verb,
                    self.text_to_code,  # type: ignore
                    multiple_answers=multiple_answers,
                )
                scores = compute_score(
                    outputs,
                    self.tokenizer,  # type: ignore
                    prefix_len=prefix_len,
                )
                beam_scores = [
                    float(torch.exp(s)) if num_beams > 1 else float("nan")
                    for s in (
                        outputs.sequences_scores
                        if num_beams > 1
                        else [torch.tensor(float("nan"))] * len(scores)
                    )
                ]
                all_outputs.extend([
                    {
                        "mention": mention,
                        "doc_id": doc_id,
                        "mention_id": mention_id,
                        "start_span": start_span,
                        "end_span": end_span,
                        "semantic_group": group,
                        "gold_concept_code": gold_code,
                        "pred_concept_name": prediction,
                        "pred_concept_code": code,
                        "score": score,
                        "beam_score": beam_score,
                        "rank": rank + 1,
                    }
                    for score, beam_score, code, prediction, mention, doc_id, mention_id, start_span, end_span, group, gold_code, rank in zip(
                        scores,
                        beam_scores,
                        codes,
                        predictions,
                        mentions,
                        doc_ids,
                        mentions_id,
                        start_spans,
                        end_spans,
                        sem_groups,
                        gold_codes,
                        list(range(num_beams)) * batch_size,
                    )
                ])

                print(f"Sampling completed. Generated {len(all_outputs)} predictions.")
        return all_outputs  # type: ignore

    def encode(self, sentence):
        return self.tokenizer.encode(sentence, return_tensors="pt")[0]  # type: ignore


class LongBELHubInterface(_LongBELHubInterface, LlamaForCausalLM):
    pass


class LongBEL(LlamaForCausalLM):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        lang=None,
        text_to_code_path=None,
        candidate_trie_path=None,
    ):
        model = LongBELHubInterface.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        model.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        model.tokenizer.padding_side = "left"

        # Set language: explicit override > config > default
        model.lang = lang if lang is not None else getattr(model.config, "lang", "en")  # type: ignore
        logger.info(f"Model language set to: {model.lang}")

        # ------------------------
        # Load text_to_code
        # ------------------------
        text_to_code_file_local = (
            text_to_code_path
            if text_to_code_path is not None
            else os.path.join(model_name_or_path, "text_to_code.json")
        )
        try:
            if os.path.exists(text_to_code_file_local):
                with open(text_to_code_file_local, encoding="utf-8") as f:
                    model.text_to_code = json.load(f)
                logger.info(
                    f"Loaded text_to_code.json from local path: {text_to_code_file_local}"
                )
            else:
                text_to_code_path = hf_hub_download(
                    repo_id=model_name_or_path,
                    filename="text_to_code.json",
                )
                with open(text_to_code_path, encoding="utf-8") as f:
                    model.text_to_code = json.load(f)
                logger.info(
                    f"Loaded text_to_code.json from HF Hub: {text_to_code_path}"
                )
        except Exception:
            logger.warning("text_to_code.json not found (local or HF hub)")
            model.text_to_code = None  # type: ignore

        # ------------------------
        # Load candidate_trie
        # ------------------------
        candidate_trie_file_local = (
            candidate_trie_path
            if candidate_trie_path is not None
            else os.path.join(model_name_or_path, "candidate_trie.pkl")
        )
        try:
            if os.path.exists(candidate_trie_file_local):
                with open(candidate_trie_file_local, "rb") as f:
                    model.candidate_trie = pickle.load(f)
                logger.info(
                    f"Loaded candidate_trie.pkl from local path: {candidate_trie_file_local}"
                )
            else:
                candidate_trie_path = hf_hub_download(
                    repo_id=model_name_or_path,
                    filename="candidate_trie.pkl",
                )
                with open(candidate_trie_path, "rb") as f:
                    model.candidate_trie = pickle.load(f)
                logger.info(
                    f"Loaded candidate_trie.pkl from HF Hub: {candidate_trie_path}"
                )
        except Exception:
            logger.warning("candidate_trie.pkl not found (local or HF hub)")
            model.candidate_trie = None  # type: ignore
        return model

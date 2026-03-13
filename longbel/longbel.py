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

from longbel.parse_data import (
    parse_text,
    parse_text_hybrid_long,
    parse_text_hybrid_short,
    parse_text_hybrid_medium,
    parse_text_long,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Display INFO and above
    format="%(levelname)s - %(message)s",
)


def get_prefix_allowed_tokens_fn(
    model,
    sources: list[str],
    sem_groups: list[str],
    multiple_answers: bool = False,
):
    candidates_trie = model.candidate_trie  # type: ignore
    sep_token_id = model.tokenizer.sep_token_id
    eos_token_id = model.tokenizer.eos_token_id
    pad_token_id = model.tokenizer.pad_token_id
    plus_token_id = model.tokenizer.convert_tokens_to_ids("<+>")  # type: ignore
    end_group_token_id = model.tokenizer.convert_tokens_to_ids("}")  # type: ignore

    def prefix_allowed_tokens_fn(batch_id, sent):
        sent = sent.tolist()
        if len(sent) > 1 and sent[-1] in [eos_token_id, pad_token_id, sep_token_id]:
            if sep_token_id:
                return [sep_token_id, pad_token_id, eos_token_id]
            else:
                return [pad_token_id, eos_token_id]

        # Remove the prefix from the sent
        index_sep = len(sent) - 1 - sent[::-1].index(end_group_token_id)
        sent = sent[index_sep:]

        sem_group = sem_groups[batch_id]
        # Remove everything up to last sep_token_id and add prefix and tgt_lang_id
        if multiple_answers and plus_token_id in sent:
            index_plus = len(sent) - 1 - sent[::-1].index(plus_token_id)
            # Start fresh with decoder start
            if index_plus == len(sent) - 1:
                sent = [end_group_token_id]
            # If there are tokens after the last plus_token_id, keep them
            else:
                sent = [end_group_token_id] + sent[index_plus + 1 :]
        trie_out = candidates_trie[
            sem_group  # type: ignore
        ].get(sent)
        if eos_token_id in trie_out:
            if sep_token_id:
                trie_out += [sep_token_id]
            if multiple_answers:
                trie_out += [plus_token_id]
        elif not trie_out:
            if sep_token_id:
                return [sep_token_id, pad_token_id, eos_token_id]
            else:
                return [pad_token_id, eos_token_id]
        return trie_out

    return prefix_allowed_tokens_fn


def add_headers_to_prompt(source: str, target: str, previous_targets: str, context_format: str):
    if context_format == "long":
        input_sentence = f"### Context\n{source.rstrip()}\n\n### Predictions\n{previous_targets}{target.rstrip()}"
    elif context_format in ["short", "hybrid_medium"]:
        input_sentence = f"### Context\n{source.rstrip()}\n\n### Prediction\n{target.rstrip()}"
    elif context_format in ["hybrid_short", "hybrid_long"]:
        if not previous_targets:
            previous_targets = "None"
        input_sentence = f"### Context\n{source.rstrip()}\n\n### Previous Normalizations\n{previous_targets.rstrip()}\n\n### Prediction\n{target.rstrip()}"
    else:
        raise ValueError(f"Unknown context_format: {context_format}")
    return input_sentence


def parse_prediction(
    outputs: list[str],
    sem_groups: list[str],
    text_to_code: Optional[dict[str, dict[str, str]]] = None,
    multiple_answers: bool = False,
) -> tuple[list[str], list[str]]:
    codes = []
    predictions = []
    for output, group in zip(outputs, sem_groups):
        splits = output.split("} ")  # type: ignore
        if len(splits) > 1 and splits[-1].strip():
            prediction = splits[-1].strip().replace("<SEP>", "")
            if text_to_code:
                if multiple_answers:
                    prediction_list = prediction.split("<+>")  # type: ignore
                    code_list = set()
                    for pred in prediction_list:
                        code_list.add(text_to_code[group].get(pred.strip(), "NO_CODE"))
                    if len(code_list) > 1 and "NO_CODE" in code_list:
                        code_list.remove("NO_CODE")
                    code = "+".join(code_list)
                else:
                    code = text_to_code[group].get(prediction, "NO_CODE")
            else:
                code = "NO_CODE"
        else:
            print(
                "IndexError: splitting failed or empty prediction, adding empty string as prediction."
            )
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
    def predict_batch(
        self,
        all_outputs,
        batch_size,
        input_sentences,
        sem_groups,
        mentions,
        mentions_id,
        doc_ids,
        start_spans,
        end_spans,
        gold_concept_codes,
        gold_concept_names,
        constrained,
        multiple_answers,
        num_beams,
        **kwargs,
    ):
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
                sem_groups=sem_groups,
                multiple_answers=multiple_answers,
            )
        if self.tokenizer.sep_token_id:
            eos_token_id = self.tokenizer.sep_token_id
        else:
            eos_token_id = self.tokenizer.eos_token_id
        outputs = self.generate(  # type: ignore
            **input_args,
            max_new_tokens=128,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            eos_token_id=eos_token_id,  # type: ignore
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

        prefix_len = input_args["input_ids"].size(1)

        # Duplicate sem_groups and mentions for each beam
        sem_groups = [x for x in sem_groups for _ in range(num_beams)]
        mentions = [x for x in mentions for _ in range(num_beams)]
        mentions_id = [x for x in mentions_id for _ in range(num_beams)]
        gold_concept_codes = [x for x in gold_concept_codes for _ in range(num_beams)]  # type: ignore
        gold_concept_names = [x for x in gold_concept_names for _ in range(num_beams)]  # type: ignore
        start_spans = [x for x in start_spans for _ in range(num_beams)]
        end_spans = [x for x in end_spans for _ in range(num_beams)]
        doc_ids = [x for x in doc_ids for _ in range(num_beams)]
        # Parse predictions
        pred_concept_codes, pred_concept_names = parse_prediction(
            cleaned_output_sequences,
            sem_groups,
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
                "gold_concept_code": gold_concept_code,
                "gold_concept_name": gold_concept_name,
                "pred_concept_name": pred_concept_name,
                "pred_concept_code": pred_concept_code,
                "score": score,
                "beam_score": beam_score,
                "rank": rank + 1,
            }
            for score, beam_score, pred_concept_code, pred_concept_name, mention, doc_id, mention_id, start_span, end_span, group, gold_concept_code, gold_concept_name, rank in zip(
                scores,
                beam_scores,
                pred_concept_codes,
                pred_concept_names,
                mentions,
                doc_ids,
                mentions_id,
                start_spans,
                end_spans,
                sem_groups,
                gold_concept_codes,
                gold_concept_names,
                list(range(num_beams)) * batch_size,
            )
        ])
        print(f"Sampling completed. Generated {len(all_outputs)} predictions.")
        return all_outputs, cleaned_output_sequences

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
        context_format: str = "short",
        **kwargs,
    ) -> list[list[dict[str, str]]]:
        # Prepare input batch
        if self.lang == "fr":  # type: ignore
            nlp = nltk.data.load("tokenizers/punkt/french.pickle")
        elif self.lang == "en":  # type: ignore
            nlp = nltk.data.load("tokenizers/punkt/english.pickle")
        elif self.lang == "es":  # type: ignore
            nlp = nltk.data.load("tokenizers/punkt/spanish.pickle")
        else:
            raise ValueError(f"Unsupported language: {self.lang}")  # type: ignore

        print(
            f"Starting sampling on {len(bigbio_pages)} pages (lang={getattr(self, 'lang', 'unknown')}, constrained={constrained}, beams={num_beams}, batch_size={batch_size})"
        )

        def _progress(
            iterable, desc: str, total: Optional[int] = None, show: bool = True
        ):
            if show:
                return tqdm(iterable, desc=desc, total=total)
            return iterable

        all_outputs = []
        all_sources = []
        all_targets = []
        all_entities_info = []
        for data in bigbio_pages:
            if context_format == "long":
                sources, targets, entities_info = parse_text_long(
                    data=data,
                    start_entity=start_entity,
                    end_entity=end_entity,
                    start_group=start_group,
                    end_group=end_group,
                    train_mode=False,
                )
            elif context_format == "short":
                sources, targets, entities_info = parse_text(
                    data=data,
                    start_entity=start_entity,
                    end_entity=end_entity,
                    start_group=start_group,
                    end_group=end_group,
                    nlp=nlp,  # type: ignore
                    train_mode=False,
                )
            elif context_format == "hybrid_long":
                sources, targets, entities_info = parse_text_hybrid_long(
                    data=data,
                    start_entity=start_entity,
                    end_entity=end_entity,
                    start_group=start_group,
                    end_group=end_group,
                    nlp=nlp,  # type: ignore
                    train_mode=False,
                )
            elif context_format == "hybrid_short":
                sources, targets, entities_info = parse_text_hybrid_short(
                    data=data,
                    start_entity=start_entity,
                    end_entity=end_entity,
                    start_group=start_group,
                    end_group=end_group,
                    nlp=nlp,  # type: ignore
                    train_mode=False,
                )
            elif context_format == "hybrid_medium":
                sources, targets, entities_info = parse_text_hybrid_medium(
                    data=data,
                    start_entity=start_entity,
                    end_entity=end_entity,
                    start_group=start_group,
                    end_group=end_group,
                    nlp=nlp,  # type: ignore
                    train_mode=False,
                )
            else:
                raise ValueError(
                    f"Unsupported context_format value '{context_format}'."
                )
            all_sources.append(sources)
            all_targets.append(targets)
            all_entities_info.append(entities_info)

        if context_format == "short":
            # Flatten examples and entities_info for batch processing
            flat_sources = [ex for page in all_sources for ex in page]
            flat_targets = [ex for page in all_targets for ex in page]
            flat_entities = [info for page in all_entities_info for info in page]
            # Build batches with is_last always True (no sequential dependency)
            all_batches = []
            for i in range(0, len(flat_sources), batch_size):
                batch = [
                    (src, tgt, ent, True)
                    for src, tgt, ent in zip(
                        flat_sources[i : i + batch_size],
                        flat_targets[i : i + batch_size],
                        flat_entities[i : i + batch_size],
                    )
                ]
                all_batches.append(batch)
        else:
            # Build batches processing multiple pages in parallel.
            # Each slot is a different page, progressing sequentially.
            # When a page finishes, the next unstarted page takes its slot.
            page_queue = list(range(len(all_sources)))
            active_slots = []  # list of (page_idx, item_idx)
            all_batches = []

            while active_slots or page_queue:
                # Fill empty slots with new pages
                while len(active_slots) < batch_size and page_queue:
                    page_idx = page_queue.pop(0)
                    active_slots.append((page_idx, 0))

                if not active_slots:
                    break

                batch = []
                next_active = []
                for page_idx, item_idx in active_slots:
                    is_last = item_idx == len(all_sources[page_idx]) - 1
                    batch.append((
                        all_sources[page_idx][item_idx],
                        all_targets[page_idx][item_idx],
                        all_entities_info[page_idx][item_idx],
                        is_last,
                    ))
                    if not is_last:
                        next_active.append((page_idx, item_idx + 1))

                all_batches.append(batch)
                active_slots = next_active

        print(
            f"Input preparation completed. Running generation on {len(all_batches)} batches."
        )

        all_outputs = []
        batch_previous_targets = {}
        for batch in _progress(
            all_batches,
            desc="Processing batches",
            total=len(all_batches),
            show=show_progress,
        ):
            input_sentences = []
            sem_groups = []
            mentions = []
            doc_ids = []
            mentions_id = []
            gold_concept_codes = []
            gold_concept_names = []
            start_spans = []
            end_spans = []
            for batch_id, (source, target, entity, is_last_flag) in enumerate(batch):
                doc_id = entity["doc_id"]
                if doc_id not in batch_previous_targets:
                    batch_previous_targets[doc_id] = ""
                previous_targets = batch_previous_targets.get(doc_id)

                input_sentences.append(add_headers_to_prompt(source, target, previous_targets, context_format))
                sem_groups.append(entity["semantic_group"])
                mentions.append(entity["mention"])
                doc_ids.append(doc_id)
                mentions_id.append(entity["mention_id"])
                start_spans.append(entity["start_span"])
                end_spans.append(entity["end_span"])
                gold_concept_codes.append(
                    entity.get("gold_concept_code", None)
                )  # type: ignore
                gold_concept_names.append(
                    entity.get("gold_concept_name", None)
                )  # type: ignore
            all_outputs, cleaned_output_sequences = self.predict_batch(
                all_outputs=all_outputs,
                batch_size=batch_size,
                input_sentences=input_sentences,
                sem_groups=sem_groups,
                mentions=mentions,
                mentions_id=mentions_id,
                doc_ids=doc_ids,
                start_spans=start_spans,
                end_spans=end_spans,
                gold_concept_codes=gold_concept_codes,
                gold_concept_names=gold_concept_names,
                constrained=constrained,
                multiple_answers=multiple_answers,
                num_beams=num_beams,
                **kwargs,
            )
            if is_last_flag:
                batch_previous_targets[doc_id] = ""
            for i, doc_id in enumerate(doc_ids):
                clean_sentence = cleaned_output_sequences[num_beams * i]
                clean_sentence = start_entity + clean_sentence.split(start_entity)[-1]
                if context_format in ["hybrid_short", "hybrid_long"]:
                    clean_sentence = clean_sentence.rstrip() + "\n"
                elif context_format == "long":
                    clean_sentence = clean_sentence.rstrip("<SEP>") + "<SEP>"
                batch_previous_targets[doc_id] += clean_sentence

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

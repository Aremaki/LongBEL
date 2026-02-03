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

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
)

from longbel.guided_inference import get_prefix_allowed_tokens_fn
from longbel.utils import chunk_it

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Display INFO and above
    format="%(levelname)s - %(message)s",
)


def find_mention(text: str) -> str:
    # Find the bracketed content
    match = re.search(r"\[(.*?)\]", text)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("No mention found in the text.")


def find_sem_group(text: str) -> str:
    # Find the bracketed content
    match = re.search(r"\{(.*?)\}", text)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("No group type found in the text.")


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
        splits = output.split(f"] {verb}")  # type: ignore
        if len(splits) > 1 and splits[1].strip():
            prediction = splits[1].strip()
            if text_to_code:
                if multiple_answers:
                    prediction_list = prediction.split("<SEP>")  # type: ignore
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
    # Identify the separator token (if it exists)
    sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else None

    # Build the list of special tokens to remove
    if any("tag" in token for token in tokenizer.all_special_tokens):
        tokens_to_remove = tokenizer.all_special_tokens[:-3]
    elif any("{" in token for token in tokenizer.all_special_tokens):
        tokens_to_remove = tokenizer.all_special_tokens[:-4]
    else:
        tokens_to_remove = tokenizer.all_special_tokens

    # Keep the sep_token if defined
    if sep_token in tokens_to_remove:
        tokens_to_remove = [tok for tok in tokens_to_remove if tok != sep_token]

    cleaned_outputs = []
    for sequence in outputs:
        # Remove undesired special tokens
        for token in tokens_to_remove:
            sequence = sequence.replace(token, "")

        # Remove spaces *immediately* after the sep_token (e.g. "<sep>  text" → "<sep>text")
        if sep_token:
            sequence = re.sub(rf"({re.escape(sep_token)})\s+", r"\1", sequence)

        cleaned_outputs.append(sequence.strip())

    return cleaned_outputs


class _GENREHubInterface:
    def sample(
        self,
        sentences: str | list[str],  # type: ignore
        num_beams: int = 5,
        constrained: bool = True,
        multiple_answers: bool = False,
        **kwargs,
    ) -> list[list[dict[str, str]]]:

        if isinstance(sentences, str):
            sentences = [sentences]
        # Prepare input batch
        if self.lang == "fr":  # type: ignore
            verb = "est"
        elif self.lang == "en":  # type: ignore
            verb = "is"
        elif self.lang == "es":  # type: ignore
            verb = "es"
        else:
            raise ValueError(f"Unsupported language: {self.lang}")  # type: ignore
        prefix_templates = []
        complete_input_text = []
        sem_groups = []
        mentions = []
        for sent in sentences:
            sem_group = find_sem_group(sent)
            mention = find_mention(sent)
            prefix = f"[{mention}] {verb}"
            complete_input = f"{sent}<SEP>{prefix}"
            mentions.append(mention)
            prefix_templates.append(prefix)
            complete_input_text.append(complete_input)
            sem_groups.append(sem_group)

        # Encode input batch
        input_args = {
            k: v.to(self.device)  # type: ignore
            for k, v in self.tokenizer.batch_encode_plus(  # type: ignore
                complete_input_text, padding="longest", return_tensors="pt"
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
                self,
                sentences,
                prefix_templates,
                sem_groups,
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

        # Parse predictions
        codes, predictions = parse_prediction(
            cleaned_output_sequences,
            sem_groups,
            verb,
            self.text_to_code,  # type: ignore
            multiple_answers=multiple_answers,
        )
        scores = compute_score(outputs, self.tokenizer, prefix_len=prefix_len)  # type: ignore
        beam_scores = [
            float(torch.exp(s)) if num_beams > 1 else float("nan")
            for s in (
                outputs.sequences_scores
                if num_beams > 1
                else [torch.tensor(float("nan"))] * len(scores)
            )
        ]

        outputs = chunk_it(
            [
                {
                    "text": text,
                    "mention": mention,
                    "semantic_group": group,
                    "pred_concept_name": prediction,
                    "pred_concept_code": code,
                    "score": score,
                    "beam_score": beam_score,
                }
                for text, score, beam_score, code, prediction, mention, group in zip(
                    cleaned_output_sequences,
                    scores,
                    beam_scores,
                    codes,
                    predictions,
                    mentions,
                    sem_groups,
                )
            ],
            len(sentences),
        )

        return outputs  # type: ignore

    def encode(self, sentence):
        return self.tokenizer.encode(sentence, return_tensors="pt")[0]  # type: ignore


class LlamaGENREHubInterface(_GENREHubInterface, LlamaForCausalLM):
    pass


class Llama_GENRE(LlamaForCausalLM):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        lang=None,
        text_to_code_path=None,
        candidate_trie_path=None,
    ):
        model = LlamaGENREHubInterface.from_pretrained(
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

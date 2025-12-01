"""
Core models for SynCABEL
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    LlamaForCausalLM,
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
)

from syncabel.utils import chunk_it

logger = logging.getLogger(__name__)


def compute_score(outputs, tokenizer, prefix_len=0):
    sequences = outputs.sequences  # (N, seq_len)
    scores = outputs.scores  # list length T = # generated tokens

    N, total_len = sequences.shape
    T = len(scores)

    # keep only the generated part (completion)
    sequences = sequences[:, prefix_len : prefix_len + T]

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
        sentences: list[str],
        num_beams: int = 5,
        text_to_id: dict[str, str] = None,  # type: ignore
        marginalize: bool = False,
        prefix_templates: list[str] = None,  # type: ignore
        **kwargs,
    ) -> list[dict[str, str]]:
        input_args = {
            k: v.to(self.device)  # type: ignore
            for k, v in self.tokenizer.batch_encode_plus(  # type: ignore
                sentences, padding="longest", return_tensors="pt"
            ).items()
        }

        # ---------------------------------------
        # Encode and batch decoder prefixes
        # ---------------------------------------
        decoder_input_ids = None
        if prefix_templates is not None:
            # encode prefixes
            prefix_enc = self.tokenizer.batch_encode_plus(  # type: ignore
                prefix_templates,
                padding="longest",  # batch pad
                truncation=True,
                add_special_tokens=False,  # IMPORTANT for prefixes
                return_tensors="pt",
            )

            decoder_input_ids = prefix_enc["input_ids"].to(self.device)  # type: ignore
            prefix_len = decoder_input_ids.size(1)
        else:
            prefix_len = input_args["input_ids"].size(1)

        outputs = self.generate(  # type: ignore
            **input_args,
            # min_length=0,
            # max_length=128,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            decoder_input_ids=decoder_input_ids,
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
                    "score": score,
                    "beam_score": beam_score,
                }
                for text, score, beam_score in zip(
                    cleaned_output_sequences,
                    scores,
                    beam_scores,
                )
            ],
            len(sentences),
        )

        return outputs  # type: ignore

    def encode(self, sentence):
        return self.tokenizer.encode(sentence, return_tensors="pt")[0]  # type: ignore


class BARTGENREHubInterface(_GENREHubInterface, BartForConditionalGeneration):
    pass


class MBARTGENREHubInterface(_GENREHubInterface, MBartForConditionalGeneration):
    pass


class MT5GENREHubInterface(_GENREHubInterface, MT5ForConditionalGeneration):
    pass


class LlamaGENREHubInterface(_GENREHubInterface, LlamaForCausalLM):
    pass


class MBart_GENRE(MBartForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = MBARTGENREHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, add_prefix_space=False
        )
        return model


class Bart_GENRE(BartForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = BARTGENREHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, add_prefix_space=False
        )
        return model


class MT5_GENRE(MT5ForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = MT5GENREHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, add_prefix_space=False
        )
        return model


class Llama_GENRE(LlamaForCausalLM):
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = LlamaGENREHubInterface.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        model.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        return model

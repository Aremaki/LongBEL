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


def compute_score(outputs, tokenizer) -> list[float]:
    """
    Compute a confidence score for each generated sequence in outputs.

    Confidence = geometric mean of token probabilities,
    ignoring PAD and EOS tokens.

    Args:
        outputs: HuggingFace generate() output with
                 return_dict_in_generate=True and output_scores=True
        tokenizer: tokenizer used to generate sequences

    Returns:
        List of confidence scores (float) in (0,1] for each sequence
    """
    sequences = outputs.sequences  # (batch*num_return_sequences, seq_len)
    scores = outputs.scores  # list of logits per step

    N, L = sequences.shape
    T = len(scores)
    seq_len = sequences.size(1)

    # Case 1: causal LM → sequences include a BOS token that has no score
    if seq_len == T + 1:
        # Drop BOS
        sequences = sequences[:, 1:]
        seq_len = sequences.size(1)

    # Case 2: sequences shorter than scores (common in encoder-decoder models)
    if seq_len < T:
        # Truncate scores to match sequence length
        scores = scores[:seq_len]
        T = len(scores)

    # Case 3: sequences longer than scores (rare but possible with padding)
    elif seq_len > T:
        # Truncate sequences
        sequences = sequences[:, :T]
        seq_len = sequences.size(1)

    # If still inconsistent
    if seq_len != T:
        raise ValueError(f"Unrecoverable mismatch: sequences {seq_len} vs scores {T}")

    # Create mask to ignore PAD/EOS tokens
    mask = (
        (sequences != tokenizer.pad_token_id)
        & (sequences != tokenizer.eos_token_id)
        & (sequences != tokenizer.bos_token_id)
    )

    # Compute log-probabilities for chosen tokens
    logprob_steps = []
    for t, logits in enumerate(scores):
        log_probs_t = F.log_softmax(logits, dim=-1)  # (N, vocab)
        token_t = sequences[:, t]
        idx = torch.arange(N)
        logprob_steps.append(log_probs_t[idx, token_t])

    # Stack → (N, T)
    logprobs = torch.stack(logprob_steps, dim=1)

    # Apply mask to remove PAD/EOS/BOS tokens
    logprobs.masked_fill_(~mask, 0)
    lengths = mask.sum(dim=1)  # number of valid tokens per sequence

    # Compute geometric mean → confidence
    # Avoid division by zero
    lengths = torch.clamp(lengths, min=1)
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
        **kwargs,
    ) -> list[dict[str, str]]:
        input_args = {
            k: v.to(self.device)  # type: ignore
            for k, v in self.tokenizer.batch_encode_plus(  # type: ignore
                sentences, padding="longest", return_tensors="pt"
            ).items()
        }

        outputs = self.generate(  # type: ignore
            **input_args,
            min_length=0,
            max_length=128,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
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

        scores = compute_score(outputs, self.tokenizer)  # type: ignore
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
        model = LlamaGENREHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return model

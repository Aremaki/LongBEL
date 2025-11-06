import re
from typing import Optional

from syncabel.trie import Trie


def _get_tgt_lang_token_id(tokenizer):
    """Best-effort retrieval of target language token id from a HF tokenizer.

    Returns None when not available.
    """
    # Some tokenizers (MBart, M2M, NLLB) expose `tgt_lang` and different ways to map to ids
    tgt = getattr(tokenizer, "tgt_lang", None)
    if not tgt:
        return None

    # Common mapping dict
    try:
        lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
        if isinstance(lang_code_to_id, dict) and tgt in lang_code_to_id:
            return lang_code_to_id[tgt]
    except Exception:
        pass

    return None


def find_group_type(text: str) -> str:
    # Find the bracketed content
    match = re.search(r"\{(.*?)\}", text)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("No group type found in the text.")


def get_prefix_allowed_tokens_fn(
    model,
    decoder_start_token_id: int,
    sentences: list[str],
    candidates_trie: dict[str, Trie] = None,  # type: ignore
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        sentences,
        decoder_start_token_id,
        model.tokenizer.bos_token_id,
        model.tokenizer.eos_token_id,
        model.tokenizer.pad_token_id,
        model.tokenizer.sep_token_id,
        _get_tgt_lang_token_id(model.tokenizer),
        candidates_trie,
    )


def _get_end_to_end_prefix_allowed_tokens_fn(
    sentences: list[str],
    decoder_start_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    sep_token_id: int,
    tgt_lang_id: Optional[int],
    candidates_trie: dict[str, Trie] = None,  # type: ignore
):
    sent_sem_type = []
    for sent in sentences:
        sem_type = find_group_type(sent)
        sent_sem_type.append(sem_type)

    def prefix_allowed_tokens_fn(batch_id, sent):
        sent = sent.tolist()
        if len(sent) > 1 and sent[-1] in [eos_token_id, pad_token_id]:
            return [pad_token_id]
        sem_type = sent_sem_type[batch_id]
        # Remove everything up to last sep_token_id and add decoder_start_token_id
        if sep_token_id in sent:
            sep_index = len(sent) - 1 - sent[::-1].index(sep_token_id)
            if sep_index == len(sent) - 1:
                # Start fresh with decoder start (and optional tgt language token)
                sent = [decoder_start_token_id] + (
                    [tgt_lang_id] if tgt_lang_id is not None else []
                )
            else:
                sent = (
                    [decoder_start_token_id]
                    + ([tgt_lang_id] if tgt_lang_id is not None else [])
                    + sent[sep_index + 1 :]
                )
        if bos_token_id is not None:
            clean_sent = [x for x in sent if x != bos_token_id]
            trie_out = candidates_trie[
                sem_type  # type: ignore
            ].get(clean_sent)
            if eos_token_id in trie_out:
                trie_out.append(sep_token_id)
            return [bos_token_id] + trie_out
        else:
            trie_out = candidates_trie[
                sem_type  # type: ignore
            ].get(sent)
            if eos_token_id in trie_out:
                trie_out.append(sep_token_id)
            return trie_out

    return prefix_allowed_tokens_fn

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
    sources: list[str],
    prefix_templates: list[str],
    candidates_trie: dict[str, Trie] = None,  # type: ignore
    multiple_answers: bool = False,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        sources,
        [model.tokenizer.encode(prefix) for prefix in prefix_templates],
        model.tokenizer.bos_token_id,
        model.tokenizer.eos_token_id,
        model.tokenizer.pad_token_id,
        model.tokenizer.sep_token_id,
        _get_tgt_lang_token_id(model.tokenizer),
        candidates_trie,
        multiple_answers,
    )


def _get_end_to_end_prefix_allowed_tokens_fn(
    sources: list[str],
    prefix_templates: list[list[int]],
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    sep_token_id: int,
    tgt_lang_id: Optional[int],
    candidates_trie: dict[str, Trie] = None,  # type: ignore
    multiple_answers: bool = False,
):
    sent_sem_type = []
    for sent in sources:
        sem_type = find_group_type(sent)
        sent_sem_type.append(sem_type)

    def prefix_allowed_tokens_fn(batch_id, sent):
        sent = sent.tolist()
        prefix = prefix_templates[batch_id]
        # Remove the prefix from the sent
        index_sep = sent.index(sep_token_id)
        sent = sent[index_sep + 1 :]
        # Check if the prefix is present
        prefix_len = len(prefix)
        if sent[:prefix_len] == prefix:
            sent = sent[prefix_len - 1 :]
        else:
            raise ValueError("Prefix not found in the generated sentence.")
        if len(sent) > 1 and sent[-1] in [eos_token_id, pad_token_id]:
            return [pad_token_id, eos_token_id]
        sem_type = sent_sem_type[batch_id]
        # Remove everything up to last sep_token_id and add prefix and tgt_lang_id
        if multiple_answers and sep_token_id in sent:
            sep_index = len(sent) - 1 - sent[::-1].index(sep_token_id)
            if sep_index == len(sent) - 1:
                # Start fresh with decoder start (and optional tgt language token)
                sent = [prefix[-1]] + ([tgt_lang_id] if tgt_lang_id is not None else [])
            else:
                sent = (
                    [prefix[-1]]
                    + ([tgt_lang_id] if tgt_lang_id is not None else [])
                    + sent[sep_index + 1 :]
                )
        trie_out = candidates_trie[
            sem_type  # type: ignore
        ].get(sent)
        if multiple_answers and eos_token_id in trie_out:
            trie_out = [sep_token_id] + trie_out
        return trie_out

    return prefix_allowed_tokens_fn

import re

from syncabel.trie import Trie


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
        candidates_trie,
    )


def _get_end_to_end_prefix_allowed_tokens_fn(
    sentences: list[str],
    decoder_start_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    sep_token_id: int,
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
        # Remove everything up to sep_token_id and add decoder_start_token_id
        if sep_token_id in sent:
            sep_index = sent.index(sep_token_id)
            if sep_index == len(sent) - 1:
                sent = [decoder_start_token_id]
            else:
                sent = [decoder_start_token_id] + sent[sep_index + 1 :]
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

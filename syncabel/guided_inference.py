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
    sentences: list[str],
    candidates_trie: dict[str, Trie] = None,  # type: ignore
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        sentences,
        model.tokenizer.eos_token_id,
        candidates_trie,
    )


def _get_end_to_end_prefix_allowed_tokens_fn(
    sentences: list[str],
    eos_token_id: int,
    candidates_trie: dict[str, Trie] = None,  # type: ignore
):
    sent_sem_type = []
    for sent in sentences:
        sem_type = find_group_type(sent)
        sent_sem_type.append(sem_type)

    def prefix_allowed_tokens_fn(batch_id, sent):
        sent = sent.tolist()
        if sent[-1] == eos_token_id:
            return [eos_token_id]
        sem_type = sent_sem_type[batch_id]
        return candidates_trie[
            sem_type  # type: ignore
        ].get(sent)

    return prefix_allowed_tokens_fn

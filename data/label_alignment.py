def align_word_labels_to_tokens(word_labels: list[int], word_ids: list[int | None]) -> list[int]:
    token_labels = []
    for wid in word_ids:
        if wid is None:
            token_labels.append(-100)
        elif wid < len(word_labels):
            token_labels.append(word_labels[wid])
        else:
            token_labels.append(-100)
    return token_labels


def extract_word_labels_from_pair(original: str, compressed: str) -> list[int]:
    _PUNCT = str.maketrans("", "", ".,!?;:\"'")
    orig_words = [w.translate(_PUNCT) for w in original.lower().split()]
    comp_words = [w.translate(_PUNCT) for w in compressed.lower().split()]

    labels = [0] * len(orig_words)
    comp_ptr = 0
    for i, ow in enumerate(orig_words):
        if comp_ptr < len(comp_words) and ow == comp_words[comp_ptr]:
            labels[i] = 1
            comp_ptr += 1
    return labels

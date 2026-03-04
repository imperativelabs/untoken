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
    orig_words = original.lower().split()
    comp_set = set(compressed.lower().split())
    return [1 if w.strip(".,!?;:\"'") in comp_set else 0 for w in orig_words]

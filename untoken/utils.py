import re


def split_into_chunks(text: str, tokenizer, max_tokens: int = 480) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        toks = len(tokenizer.encode(sent, add_special_tokens=False))
        if current_len + toks > max_tokens and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sent)
        current_len += toks

    if current:
        chunks.append(" ".join(current))
    return chunks


def clean_compressed_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s([?.!,;:])', r'\1', text)
    return text.strip()


def decode_kept_tokens(input_ids: list[int], keep_mask: list[int], tokenizer) -> str:
    # Pull in any continuation subwords (##) whose prefix was kept, to avoid
    # split tokens like 503 → "50" when ##3 is dropped.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    mask = list(keep_mask)
    for i in range(1, len(tokens)):
        if mask[i - 1] and tokens[i] is not None and tokens[i].startswith("##"):
            mask[i] = 1
    kept = [tok for tok, keep in zip(input_ids, mask) if keep]
    return tokenizer.decode(kept, skip_special_tokens=True)

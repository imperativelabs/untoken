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
            current = [current[-1]] if current else []
            current_len = len(tokenizer.encode(current[0], add_special_tokens=False)) if current else 0
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
    kept = [tok for tok, keep in zip(input_ids, keep_mask) if keep]
    return tokenizer.decode(kept, skip_special_tokens=True)

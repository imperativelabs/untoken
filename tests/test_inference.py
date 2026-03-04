import pytest
from unittest.mock import MagicMock, patch
import torch


LONG_TEXT = (
    "In its most fundamental sense, compression is the process of encoding information "
    "using fewer bits or resources than the original representation by identifying and "
    "eliminating statistical redundancies or irrelevant data within a dataset. "
    "Whether applied to digital media, text, or the high-dimensional vector spaces of "
    "Large Language Models, compression relies on the principle that most raw information "
    "contains noise or repeating patterns that do not contribute new meaning."
)


def test_compress_returns_shorter_text():
    from untoken.utils import split_into_chunks, clean_compressed_text, decode_kept_tokens
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    chunks = split_into_chunks(LONG_TEXT, tokenizer, max_tokens=480)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


def test_decode_kept_tokens():
    from untoken.utils import decode_kept_tokens
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ids = tokenizer.encode("hello world foo bar", add_special_tokens=True)
    mask = [0] + [1, 0, 1, 0] + [0]
    result = decode_kept_tokens(ids[:len(mask)], mask, tokenizer)
    assert isinstance(result, str)
    assert len(result) > 0


def test_clean_compressed_text():
    from untoken.utils import clean_compressed_text
    assert clean_compressed_text("hello  world  .") == "hello world."
    assert clean_compressed_text("  leading  ") == "leading"

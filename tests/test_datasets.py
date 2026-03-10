import torch
import pytest
from transformers import AutoTokenizer
from data.label_alignment import align_word_labels_to_tokens, extract_word_labels_from_pair
from data.loaders import UntokenDataset, RealTextDataset


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


def test_label_alignment(tokenizer):
    word_labels = [1, 0, 1, 0, 1]
    text = "hello world foo bar baz"
    enc = tokenizer(text, return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    token_labels = align_word_labels_to_tokens(word_labels, word_ids)
    assert len(token_labels) == len(word_ids)
    for wid, tl in zip(word_ids, token_labels):
        if wid is None:
            assert tl == -100
        else:
            assert tl == word_labels[wid]


def test_extract_word_labels():
    orig = "the quick brown fox jumps over"
    comp = "quick brown fox jumps"
    labels = extract_word_labels_from_pair(orig, comp)
    assert labels == [0, 1, 1, 1, 1, 0]


def test_extract_word_labels_repeated_function_word():
    # "the" appears 3x in original, 1x in compressed at position 1 (cat sat the mat)
    # Only the positionally-matched "the" (3rd word of original → index 4) should get label=1
    orig = "the cat sat on the mat near the door"
    comp = "cat sat the mat"
    labels = extract_word_labels_from_pair(orig, comp)
    assert labels[0] == 0   # first "the" not matched
    assert labels[1] == 1   # "cat"
    assert labels[2] == 1   # "sat"
    assert labels[3] == 0   # "on"
    assert labels[4] == 1   # second "the" (positionally matched)
    assert labels[5] == 1   # "mat"
    assert labels[6] == 0   # "near"
    assert labels[7] == 0   # third "the" not matched
    assert labels[8] == 0   # "door"
    assert sum(labels) == 4  # exactly 4 words matched


def test_untoken_dataset_shape(tokenizer):
    records = [{"text": "Hello world this is a test sentence.", "labels": [1, 0, 1, 0, 1, 0, 1]}]
    ds = UntokenDataset(records, tokenizer, max_length=64)
    item = ds[0]
    assert item["input_ids"].shape == (64,)
    assert item["attention_mask"].shape == (64,)
    assert item["labels"].shape == (64,)


def test_real_text_dataset(tokenizer):
    ds = RealTextDataset(["Short headline text.", "Another headline."], tokenizer, max_length=32)
    item = ds[0]
    assert item["input_ids"].shape == (32,)
    assert "labels" not in item

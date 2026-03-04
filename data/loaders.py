from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from data.label_alignment import align_word_labels_to_tokens, extract_word_labels_from_pair


class UntokenDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = rec["text"]
        word_labels = rec.get("labels", None)

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        special_tokens_mask = enc["special_tokens_mask"].squeeze(0)

        if word_labels is not None:
            word_ids = enc.word_ids(batch_index=0)
            token_labels = align_word_labels_to_tokens(word_labels, word_ids)
            labels = torch.tensor(token_labels[:self.max_length], dtype=torch.long)
            if len(labels) < self.max_length:
                pad = torch.full((self.max_length - len(labels),), -100, dtype=torch.long)
                labels = torch.cat([labels, pad])
        else:
            labels = torch.full((self.max_length,), -100, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
            "labels": labels,
        }


class RealTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def _pick_field(row: dict, candidates: list[str]) -> str:
    for k in candidates:
        if k in row and isinstance(row[k], str) and row[k].strip():
            return row[k].strip()
    # fallback: longest string field
    best = max((v for v in row.values() if isinstance(v, str) and v.strip()),
               key=len, default="")
    return best


def load_meetingbank_records(data_dir: Path) -> list[dict]:
    ds = load_from_disk(str(data_dir / "meetingbank"))

    first_split = next(iter(ds.values()))
    if len(first_split) > 0:
        print(f"[meetingbank] columns: {list(first_split.features.keys())}")

    TEXT_KEYS = ["transcript", "source_doc", "text", "document", "input", "article"]
    COMP_KEYS = ["summary", "compressed", "target", "output", "abstract"]

    records = []
    for split in ds.values():
        for row in split:
            text = _pick_field(row, TEXT_KEYS)
            compressed = _pick_field(row, COMP_KEYS)
            if not text:
                continue
            word_labels = extract_word_labels_from_pair(text, compressed) if compressed else None
            records.append({"text": text, "labels": word_labels})

    print(f"[meetingbank] loaded {len(records)} records")
    return records


def load_cnn_records(data_dir: Path, max_samples: int = 100000) -> list[dict]:
    ds = load_from_disk(str(data_dir / "cnn_dailymail"))
    records = []
    for row in ds["train"].select(range(min(max_samples, len(ds["train"])))):
        records.append({"text": row["article"], "labels": None})
    return records


def load_cnn_highlights(data_dir: Path, max_samples: int = 50000) -> list[str]:
    ds = load_from_disk(str(data_dir / "cnn_dailymail"))
    results = []
    for row in ds["train"]:
        for line in row["highlights"].split("\n"):
            line = line.strip()
            if line:
                results.append(line)
            if len(results) >= max_samples:
                return results
    return results


def build_train_dataset(phase: int, data_dir: Path, tokenizer) -> UntokenDataset:
    data_dir = Path(data_dir)
    records = load_meetingbank_records(data_dir)
    if phase in (2, 3):
        records = records + load_cnn_records(data_dir)
    return UntokenDataset(records, tokenizer)


def build_real_text_dataset(data_dir: Path, tokenizer) -> RealTextDataset:
    highlights = load_cnn_highlights(Path(data_dir))
    return RealTextDataset(highlights, tokenizer)

"""Tests for evaluation/metrics.py — Fix 5: real BERTScore."""
from __future__ import annotations

import pytest


ORIGINALS = ["The cat sat on the mat.", "Machine learning is transforming industries."]
COMPRESSED = ["cat sat mat", "Machine learning transforms industries."]


def test_bertscore_f1_in_range():
    from evaluation.metrics import bertscore_f1
    score = bertscore_f1(ORIGINALS, COMPRESSED, device="cpu")
    assert 0.0 <= score <= 1.0, f"bertscore_f1 out of range [0,1]: {score}"


def test_bertscore_f1_differs_from_cosine():
    from evaluation.metrics import bertscore_f1, cosine_similarity_score
    bs = bertscore_f1(ORIGINALS, COMPRESSED, device="cpu")
    cos = cosine_similarity_score(ORIGINALS, COMPRESSED, device="cpu")
    assert bs != pytest.approx(cos, abs=1e-4), \
        f"bertscore_f1 ({bs:.4f}) should differ from cosine_similarity_score ({cos:.4f})"

from __future__ import annotations

import time

import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


def bertscore_f1(originals: list[str], compressed: list[str], device: str = "cpu") -> float:
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(compressed, originals, lang="en", device=device, verbose=False)
    return F1.mean().item()


def rouge_l(originals: list[str], compressed: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(o, c)["rougeL"].fmeasure for o, c in zip(originals, compressed)]
    return sum(scores) / len(scores)


def cosine_similarity_score(originals: list[str], compressed: list[str], device="cpu") -> float:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()
    with torch.no_grad():
        emb_o = model.encode(originals, convert_to_tensor=True, device=device)
        emb_c = model.encode(compressed, convert_to_tensor=True, device=device)
    return F.cosine_similarity(emb_o.float(), emb_c.float()).mean().item()


def compression_ratio(originals: list[str], compressed: list[str], tokenizer) -> float:
    orig_lens = [len(tokenizer.encode(t)) for t in originals]
    comp_lens = [len(tokenizer.encode(t)) for t in compressed]
    return sum(comp_lens) / sum(orig_lens)


def latency_ms(compressor_fn, text: str, n_runs: int = 20) -> float:
    for _ in range(3):
        compressor_fn(text)
    start = time.perf_counter()
    for _ in range(n_runs):
        compressor_fn(text)
    return (time.perf_counter() - start) / n_runs * 1000

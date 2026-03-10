from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk

from evaluation.metrics import bertscore_f1, cosine_similarity_score, rouge_l, compression_ratio


def run_baseline_random(text: str, tokenizer, ratio: float = 0.3) -> str:
    import random
    ids = tokenizer.encode(text, add_special_tokens=False)
    k = max(1, int(len(ids) * ratio))
    kept = sorted(random.sample(range(len(ids)), k))
    return tokenizer.decode([ids[i] for i in kept])


def run_baseline_stopwords(text: str, tokenizer=None, ratio: float = 0.3) -> str:
    """Remove stopwords first, then truncate to hit the target ratio.

    Pure stopword removal achieves ~0.76 compression, which is not comparable
    to a 0.3 target. We use stopword priority scoring but truncate to the
    actual target token count so all baselines are compared at equal compression.
    """
    STOPWORDS = {
        "the","a","an","is","are","was","were","be","been","being","have","has","had",
        "do","does","did","will","would","could","should","may","might","shall","to",
        "of","in","for","on","with","at","by","from","as","into","through","about",
        "and","or","but","nor","so","yet","both","either","neither","not","this","that",
    }
    words = text.split()
    # Score: non-stopwords first, then stopwords — keeps semantic words preferentially
    scored = [(i, w, 0 if w.lower().strip(".,!?;:\"'") not in STOPWORDS else 1)
              for i, w in enumerate(words)]
    if tokenizer is not None:
        total_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        target_tokens = max(1, int(total_tokens * ratio))
        # Greedily select words in priority order until we hit target token count
        scored_sorted = sorted(scored, key=lambda x: x[2])
        kept_indices = set()
        token_count = 0
        for idx, word, _ in scored_sorted:
            wt = len(tokenizer.encode(word, add_special_tokens=False))
            if token_count + wt > target_tokens:
                continue
            kept_indices.add(idx)
            token_count += wt
        kept = [w for i, w, _ in scored if i in kept_indices]
    else:
        k = max(1, int(len(words) * ratio))
        non_stop = [(i, w) for i, w, s in scored if s == 0]
        stop = [(i, w) for i, w, s in scored if s == 1]
        chosen = sorted((non_stop + stop)[:k], key=lambda x: x[0])
        kept = [w for _, w in chosen]
    return " ".join(kept) if kept else words[0]


def _eval_system(name: str, originals: list[str], compressed: list[str], tokenizer, device: str = "cpu") -> dict:
    return {
        "cosine_sim": cosine_similarity_score(originals, compressed, device=device),
        "bertscore_f1": bertscore_f1(originals, compressed, device=device),
        "rouge_l": rouge_l(originals, compressed),
        "compression_ratio": compression_ratio(originals, compressed, tokenizer),
    }


def benchmark(
    compress_fn,
    data_dir: Path,
    tokenizer,
    output_path: str = "results.json",
    ratios: list[float] | None = None,
    device: str = "cpu",
    n_samples: int = 200,
):
    """Run the full evaluation benchmark.

    Args:
        compress_fn: callable(text, ratio) -> compressed_text
        data_dir: path to dataset directory
        tokenizer: HF tokenizer
        output_path: where to save JSON results
        ratios: list of compression ratios to evaluate (default [0.2, 0.3, 0.4, 0.5])
        device: device for embedding models
        n_samples: number of samples to evaluate
    """
    if ratios is None:
        ratios = [0.2, 0.3, 0.4, 0.5]

    data_dir = Path(data_dir)
    all_results = {}

    # Load LLMLingua-2 baseline if available
    try:
        from llmlingua import PromptCompressor
        llmlingua = PromptCompressor(
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
        )
    except ImportError:
        llmlingua = None
        print("[warn] llmlingua not installed, skipping LLMLingua-2 baseline")
    except Exception as e:
        llmlingua = None
        print(f"[warn] LLMLingua-2 load failed: {e}")

    # Load eval texts from XSum (BBC News — held-out domain, not in UNTOKEN training data).
    # Fallback: BookSum, then BillSum.
    # CNN/DailyMail is intentionally excluded: UNTOKEN was trained on it, which would
    # give an unfair advantage over LLMLingua-2 (trained on MeetingBank).
    texts = []
    for ds_name, field in [("xsum", "document"), ("booksum", "chapter"), ("billsum", "text")]:
        if texts:
            break
        try:
            ds = load_from_disk(str(data_dir / ds_name))
            split = ds.get("test", list(ds.values())[0]) if hasattr(ds, "get") else ds
            for row in list(split)[:n_samples * 2]:
                t = row.get(field, "").strip() if field else ""
                if t and len(t) > 200:
                    texts.append(t)
                if len(texts) >= n_samples:
                    break
            if texts:
                print(f"[eval] Using {ds_name} ({len(texts)} samples)")
        except Exception:
            pass

    if not texts:
        print("No eval texts found, skipping benchmark.")
        return {}

    originals = [t[:2000] for t in texts[:n_samples]]

    for ratio in ratios:
        print(f"\n=== ratio={ratio} ===")
        ratio_results = {}

        compressed_ours = [compress_fn(t, ratio=ratio) for t in originals]
        ratio_results["ours"] = _eval_system("ours", originals, compressed_ours, tokenizer, device)

        compressed_random = [run_baseline_random(t, tokenizer, ratio=ratio) for t in originals]
        ratio_results["random_drop"] = _eval_system("random_drop", originals, compressed_random, tokenizer, device)

        compressed_stop = [run_baseline_stopwords(t, tokenizer=tokenizer, ratio=ratio) for t in originals]
        ratio_results["stopword_removal"] = _eval_system("stopword_removal", originals, compressed_stop, tokenizer, device)

        if llmlingua:
            try:
                compressed_ll = [
                    llmlingua.compress_prompt(t, rate=ratio)["compressed_prompt"] for t in originals
                ]
                ratio_results["llmlingua2"] = _eval_system("llmlingua2", originals, compressed_ll, tokenizer, device)
            except Exception as e:
                print(f"[warn] LLMLingua-2 eval failed at ratio={ratio}: {e}")

        all_results[str(ratio)] = ratio_results
        print(json.dumps(ratio_results, indent=2))

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return all_results

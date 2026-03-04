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


def run_baseline_stopwords(text: str) -> str:
    STOPWORDS = {
        "the","a","an","is","are","was","were","be","been","being","have","has","had",
        "do","does","did","will","would","could","should","may","might","shall","to",
        "of","in","for","on","with","at","by","from","as","into","through","about",
        "and","or","but","nor","so","yet","both","either","neither","not","this","that",
    }
    words = text.split()
    kept = [w for w in words if w.lower().strip(".,!?;:\"'") not in STOPWORDS]
    return " ".join(kept) if kept else text


def benchmark(compress_fn, data_dir: Path, tokenizer, output_path: str = "results.json"):
    data_dir = Path(data_dir)
    results = {}

    try:
        from llmlingua import PromptCompressor
        llmlingua = PromptCompressor("microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
    except Exception:
        llmlingua = None

    texts = []
    for ds_name, field in [("longbench", None), ("cnn_dailymail", "article"), ("meetingbank", "prompt")]:
        if texts:
            break
        try:
            ds = load_from_disk(str(data_dir / ds_name))
            split = list(ds.values())[0] if hasattr(ds, "values") else ds
            for row in list(split)[:200]:
                if field:
                    t = row.get(field, "")
                else:
                    t = row.get("context", row.get("input", ""))
                if t and len(t) > 100:
                    texts.append(t)
        except Exception:
            pass

    if not texts:
        print("No eval texts found, skipping benchmark.")
        return {}

    originals = [t[:2000] for t in texts[:200]]
    compressed_ours = [compress_fn(t, ratio=0.3) for t in originals]
    compressed_random = [run_baseline_random(t, tokenizer) for t in originals]
    compressed_stop = [run_baseline_stopwords(t) for t in originals]

    results["ours"] = {
        "bertscore": bertscore_f1(originals, compressed_ours),
        "rouge_l": rouge_l(originals, compressed_ours),
        "cosine_sim": cosine_similarity_score(originals, compressed_ours),
        "ratio": compression_ratio(originals, compressed_ours, tokenizer),
    }
    results["random_drop"] = {
        "bertscore": bertscore_f1(originals, compressed_random),
        "rouge_l": rouge_l(originals, compressed_random),
        "ratio": compression_ratio(originals, compressed_random, tokenizer),
    }
    results["stopword_removal"] = {
        "bertscore": bertscore_f1(originals, compressed_stop),
        "rouge_l": rouge_l(originals, compressed_stop),
        "ratio": compression_ratio(originals, compressed_stop, tokenizer),
    }

    if llmlingua:
        compressed_ll = [llmlingua.compress_prompt(t, rate=0.3)["compressed_prompt"] for t in originals]
        results["llmlingua2"] = {
            "bertscore": bertscore_f1(originals, compressed_ll),
            "rouge_l": rouge_l(originals, compressed_ll),
            "ratio": compression_ratio(originals, compressed_ll, tokenizer),
        }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results

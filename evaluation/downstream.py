from __future__ import annotations

from evaluation.metrics import bertscore_f1


def evaluate_downstream(
    examples: list[dict],
    compress_fn,
    llm_fn,
    ratio: float = 0.3,
) -> dict:
    results = {"bertscore_prompt": [], "bertscore_response": [], "exact_match": [], "ratio": []}

    for ex in examples:
        orig = ex["prompt"]
        comp = compress_fn(orig, ratio=ratio)

        resp_orig = llm_fn(orig)
        resp_comp = llm_fn(comp)

        results["bertscore_prompt"].append(bertscore_f1([orig], [comp]))
        results["bertscore_response"].append(bertscore_f1([resp_orig], [resp_comp]))
        results["exact_match"].append(int(resp_orig.strip() == resp_comp.strip()))
        results["ratio"].append(len(comp.split()) / max(len(orig.split()), 1))

    return {k: sum(v) / len(v) for k, v in results.items()}

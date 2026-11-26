import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_from_disk
from untoken import Untoken
from evaluation.benchmark import benchmark


DIVIDER = "─" * 80


def _tok_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def log_samples(ut, tokenizer, data_dir: Path, ratios: list[float], n: int = 5):
    """Print side-by-side original vs compressed outputs for visual quality check."""
    # Load sample texts (same priority as benchmark: xsum → booksum → billsum)
    texts = []
    source = None
    for ds_name, field in [("xsum", "document"), ("booksum", "chapter"), ("billsum", "text")]:
        try:
            ds = load_from_disk(str(data_dir / ds_name))
            split = ds.get("test", list(ds.values())[0]) if hasattr(ds, "get") else ds
            for row in list(split)[:n * 3]:
                t = row.get(field, "").strip()
                if t and len(t) > 200:
                    texts.append(t[:2000])
                if len(texts) >= n:
                    break
            if texts:
                source = ds_name
                break
        except Exception:
            pass

    if not texts:
        print("[warn] No texts found for sample logging.")
        return

    # Load LLMLingua-2 if available
    llmlingua = None
    try:
        from llmlingua import PromptCompressor
        llmlingua = PromptCompressor(
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
        )
        print("[samples] LLMLingua-2 loaded for side-by-side comparison.")
    except Exception:
        print("[samples] LLMLingua-2 not available — showing UNTOKEN only.")

    # Load ctxprune if installed (pip install ctxprune). Same constructor arguments as LLMLingua-2
    # above, so the two rows are comparable; it is printed only, not added to the benchmark.
    ctxprune = None
    try:
        from ctxprune.compat import PromptCompressor as CtxPromptCompressor
        ctxprune = CtxPromptCompressor(
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
        )
        print("[samples] ctxprune loaded for side-by-side comparison.")
    except Exception:
        print("[samples] ctxprune not available — skipping the ctxprune column.")

    print(f"\n{'=' * 80}")
    print(f"  QUALITATIVE SAMPLE LOG  |  source={source}  |  n={len(texts)}")
    print(f"{'=' * 80}")

    for ratio in ratios:
        print(f"\n{'#' * 80}")
        print(f"  TARGET RATIO = {ratio}")
        print(f"{'#' * 80}")

        for i, orig in enumerate(texts):
            orig_tokens = _tok_count(tokenizer, orig)

            # UNTOKEN
            ours = ut.compress(orig, ratio=ratio)
            ours_tokens = _tok_count(tokenizer, ours)
            ours_ratio = ours_tokens / max(orig_tokens, 1)

            print(f"\n{DIVIDER}")
            print(f"  SAMPLE {i + 1}/{len(texts)}  |  original={orig_tokens} tokens")
            print(DIVIDER)
            print(f"  ORIGINAL:\n  {orig[:500]}{'...' if len(orig) > 500 else ''}")

            print(f"\n  UNTOKEN  (actual ratio={ours_ratio:.3f}, {orig_tokens}→{ours_tokens} tokens):")
            print(f"  {ours[:500]}{'...' if len(ours) > 500 else ''}")

            if llmlingua:
                try:
                    ll_out = llmlingua.compress_prompt(orig, rate=ratio)["compressed_prompt"]
                    ll_tokens = _tok_count(tokenizer, ll_out)
                    ll_ratio = ll_tokens / max(orig_tokens, 1)
                    print(f"\n  LLMLINGUA-2  (actual ratio={ll_ratio:.3f}, {orig_tokens}→{ll_tokens} tokens):")
                    print(f"  {ll_out[:500]}{'...' if len(ll_out) > 500 else ''}")
                except Exception as e:
                    print(f"\n  LLMLINGUA-2: [failed: {e}]")

            if ctxprune:
                try:
                    cp_out = ctxprune.compress_prompt(orig, rate=ratio)["compressed_prompt"]
                    cp_tokens = _tok_count(tokenizer, cp_out)
                    cp_ratio = cp_tokens / max(orig_tokens, 1)
                    print(f"\n  CTXPRUNE  (actual ratio={cp_ratio:.3f}, {orig_tokens}→{cp_tokens} tokens):")
                    print(f"  {cp_out[:500]}{'...' if len(cp_out) > 500 else ''}")
                except Exception as e:
                    print(f"\n  CTXPRUNE: [failed: {e}]")

        print(f"\n{DIVIDER}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to exported compressor")
    parser.add_argument("--data-dir", default="/data/untoken")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--log-samples", type=int, default=5, metavar="N",
                        help="Print N side-by-side compression examples before benchmarking (0 to skip)")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    args = parser.parse_args()

    ut = Untoken(args.model)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    if args.log_samples > 0:
        log_samples(ut, tokenizer, Path(args.data_dir), args.ratios, n=args.log_samples)

    print(f"\n{'=' * 80}")
    print("  RUNNING FULL BENCHMARK")
    print(f"{'=' * 80}\n")
    benchmark(ut.compress, Path(args.data_dir), tokenizer, args.output, ratios=args.ratios)


if __name__ == "__main__":
    main()

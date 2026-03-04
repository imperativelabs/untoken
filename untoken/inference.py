from __future__ import annotations

import torch
from transformers import AutoConfig, AutoTokenizer

from untoken.compressor import UntokenCompressor, UntokenCompressorConfig
from untoken.utils import clean_compressed_text, decode_kept_tokens, split_into_chunks

AutoConfig.register("untoken_compressor", UntokenCompressorConfig)


class Untoken:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = UntokenCompressor.from_pretrained(model_path).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def compress(self, text: str, ratio: float = 0.3, return_stats: bool = False):
        chunks = split_into_chunks(text, self.tokenizer, max_tokens=480)
        compressed_parts = []

        for chunk in chunks:
            enc = self.tokenizer(
                chunk,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            ).to(self.device)

            scores = self.model(
                enc["input_ids"],
                enc["attention_mask"],
                enc["special_tokens_mask"],
            )
            keep_mask = self.model.get_keep_mask(scores, ratio, enc["attention_mask"])
            decoded = decode_kept_tokens(
                enc["input_ids"][0].tolist(),
                keep_mask[0].tolist(),
                self.tokenizer,
            )
            compressed_parts.append(decoded)

        result = clean_compressed_text(" ".join(compressed_parts))

        if return_stats:
            orig_tokens = sum(len(self.tokenizer.encode(c)) for c in chunks)
            comp_tokens = len(self.tokenizer.encode(result))
            stats = {
                "original_tokens": orig_tokens,
                "compressed_tokens": comp_tokens,
                "ratio": round(comp_tokens / max(orig_tokens, 1), 3),
                "savings_pct": round((1 - comp_tokens / max(orig_tokens, 1)) * 100, 1),
            }
            return result, stats

        return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--ratio", type=float, default=0.3)
    args = parser.parse_args()

    with open(args.input) as f:
        text = f.read()

    ut = Untoken(args.model)
    compressed, stats = ut.compress(text, ratio=args.ratio, return_stats=True)
    print(compressed)
    print(f"\n[{stats['original_tokens']} → {stats['compressed_tokens']} tokens, "
          f"{stats['savings_pct']}% savings]")


if __name__ == "__main__":
    main()

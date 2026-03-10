"""
Diagnose importance head score quality.
Run on GCP: python scripts/diagnose.py --model /data/untoken/untoken-v2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from untoken.compressor import UntokenCompressor

PROBES = [
    "The quick brown fox jumps over the lazy dog and then runs away into the forest.",
    "Scientists discovered a new species of deep-sea fish off the coast of Japan.",
    "The server returned a 503 error after the deployment failed during the migration step.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = UntokenCompressor.from_pretrained(args.model, local_files_only=True).to(device)
    model.eval()

    with torch.inference_mode():
        for text in PROBES:
            enc = tok(
                text,
                return_tensors="pt",
                return_special_tokens_mask=True,
                padding="max_length",
                max_length=512,
                truncation=True,
            ).to(device)

            scores = model(enc["input_ids"], enc["attention_mask"], enc["special_tokens_mask"])

            seq_len = enc["attention_mask"][0].sum().item()
            tokens = tok.convert_ids_to_tokens(enc["input_ids"][0][:seq_len].tolist())
            s = scores[0, :seq_len].tolist()

            ranked = sorted(zip(s, tokens), reverse=True)

            print(f"\nText: {text[:60]}")
            print(f"  Score range: min={min(s):.4f}  max={max(s):.4f}  mean={sum(s)/len(s):.4f}  std={(sum((x - sum(s)/len(s))**2 for x in s)/len(s))**0.5:.4f}")
            print(f"  Top tokens (score → token):")
            for score, token in ranked[:8]:
                print(f"    {score:.4f}  {token}")
            print(f"  Bottom tokens:")
            for score, token in ranked[-4:]:
                print(f"    {score:.4f}  {token}")


if __name__ == "__main__":
    main()

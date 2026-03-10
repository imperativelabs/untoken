import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from transformers import AutoTokenizer
from untoken.compressor import UntokenCompressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Phase3 checkpoint dir")
    parser.add_argument("--output", required=True, help="Export destination dir")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model = UntokenCompressor.from_pretrained(str(ckpt / "compressor"), local_files_only=True)
    model.save_pretrained(str(out))

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=False)
    tokenizer.save_pretrained(str(out))

    print(f"Exported compressor-only model to {out}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params * 2 / 1e6:.1f} MB in bf16)")


if __name__ == "__main__":
    main()

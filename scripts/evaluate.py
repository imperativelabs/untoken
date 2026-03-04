import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from transformers import AutoTokenizer
from untoken import Untoken
from evaluation.benchmark import benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to exported compressor")
    parser.add_argument("--data-dir", default="/data/untoken")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    ut = Untoken(args.model)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    benchmark(ut.compress, Path(args.data_dir), tokenizer, args.output)


if __name__ == "__main__":
    main()

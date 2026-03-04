import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from untoken import Untoken


def main():
    parser = argparse.ArgumentParser(description="Compress text using Untoken")
    parser.add_argument("--model", required=True, help="Path or HF repo for compressor")
    parser.add_argument("--input", help="Input file (reads stdin if omitted)")
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    text = open(args.input).read() if args.input else sys.stdin.read()

    ut = Untoken(args.model)
    if args.stats:
        out, stats = ut.compress(text, ratio=args.ratio, return_stats=True)
        print(out)
        print(f"\n[{stats['original_tokens']} → {stats['compressed_tokens']} tokens, "
              f"{stats['savings_pct']}% savings]", file=sys.stderr)
    else:
        print(ut.compress(text, ratio=args.ratio))


if __name__ == "__main__":
    main()

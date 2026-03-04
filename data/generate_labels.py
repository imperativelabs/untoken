import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm


PROMPT = (
    "Remove all tokens that don't contribute essential meaning. "
    "Keep ONLY the words that preserve the full semantic content. "
    "Output the compressed version using ONLY words from the original text, "
    "maintaining their original order. Do not add any new words.\n\n"
    "Original: {text}\nCompressed:"
)


def compress_with_llm(text: str, client, model: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
    )
    return resp.content[0].text.strip()


def extract_labels(original: str, compressed: str) -> list[int]:
    orig_words = original.lower().split()
    comp_set = set(compressed.lower().split())
    return [1 if w.strip(".,!?;:\"'") in comp_set else 0 for w in orig_words]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL file with {text} records")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--limit", type=int, default=50000)
    args = parser.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    out_path = Path(args.output)
    existing = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                existing.add(rec["text"][:50])

    with open(args.input) as f_in, open(out_path, "a") as f_out:
        for i, line in enumerate(tqdm(f_in)):
            if i >= args.limit:
                break
            rec = json.loads(line)
            text = rec["text"]
            if text[:50] in existing:
                continue
            try:
                compressed = compress_with_llm(text, client, args.model)
                labels = extract_labels(text, compressed)
                f_out.write(json.dumps({"text": text, "labels": labels}) + "\n")
                f_out.flush()
            except Exception as e:
                print(f"Error on sample {i}: {e}")


if __name__ == "__main__":
    main()

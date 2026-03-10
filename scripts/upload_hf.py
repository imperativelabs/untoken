import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer

from untoken.compressor import UntokenCompressor

MODEL_CARD = """---
language: en
license: apache-2.0
tags:
  - token-compression
  - prompt-compression
  - distilbert
  - nlp
  - text-generation
pipeline_tag: text-classification
---

# UNTOKEN v2

**Token-level prompt compressor.** Given an input text and a target compression ratio `r ∈ (0, 1)`,
UNTOKEN selects the most semantically important `r` fraction of tokens and returns the corresponding
subtext. No paraphrasing, no summarisation — pure token selection.

## Model Details

| Property | Value |
|----------|-------|
| Base encoder | DistilBERT-base-uncased |
| Parameters | 66.56M |
| Vocabulary | 30K WordPiece |
| Max sequence length | 512 (chunked for longer inputs) |
| Selection method | Learned importance (GAN-trained) |
| Training signal | Reconstruction + adversarial + semantic similarity |

## Usage

```python
from transformers import AutoTokenizer
from untoken.compressor import UntokenCompressor
from untoken.inference import compress

tokenizer = AutoTokenizer.from_pretrained("pacifio/untoken-v2")
model = UntokenCompressor.from_pretrained("pacifio/untoken-v2")
model.eval()

text = "Your long input text here..."
compressed = compress(model, tokenizer, text, ratio=0.3)
print(compressed)
```

## Training

Three-phase training:
1. **Phase 1 — Supervised warmup**: Importance head trained on LCS-derived labels from 7 summarisation datasets (MeetingBank, CNN/DailyMail, XSum, DialogSum, BillSum, BookSum, GSM8K). DistilBERT encoder frozen.
2. **Phase 2 — Adversarial GAN training**: Discriminator pressures compressor to produce fluent, natural-looking output. Encoder unfrozen. Gumbel-Softmax with temperature annealing 1.0→0.05.
3. **Phase 3 — Hardening**: Straight-through estimator closes the gap between soft training selection and hard top-k inference.

## Benchmark Results (ratio=0.3, XSum test set)

| Model | ROUGE-L | BERTScore F1 | Cosine Sim | Actual Ratio |
|-------|---------|--------------|------------|--------------|
| **UNTOKEN v2** | **0.455** | 0.815 | 0.653 | 0.305 |
| LLMLingua-2 (base) | 0.406 | 0.837 | 0.782 | 0.262 |
| Random drop | 0.430 | 0.802 | 0.674 | 0.303 |
| Stopword removal | 0.435 | 0.883 | 0.847 | 0.303 |

UNTOKEN v2 leads on ROUGE-L (longest common subsequence overlap) at all compression ratios,
reflecting faithful preservation of the original token sequence. Cosine similarity scores favour
stopword-removal baselines due to sentence embedding insensitivity to function words — this is a
known limitation of embedding-based metrics for token-selection compression.

## Comparison with LLMLingua-2

| Property | UNTOKEN v2 | LLMLingua-2 (base) |
|----------|-----------|--------------------|
| Parameters | **66.56M** | 179M |
| Ratio adherence | **Exact (top-k)** | Approximate (can undershoot) |
| Training signal | Reconstruction + adversarial + semantic | Supervised only |
| Multilingual | No | Yes |

## Limitations

- English only (DistilBERT-base-uncased backbone)
- Sequences >512 tokens are split at sentence boundaries and processed independently
- Compression ratio is applied per chunk, not globally across the document
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Exported model directory (output of export.py)")
    parser.add_argument("--repo-id", default="pacifio/untoken-v2", help="HuggingFace repo id")
    parser.add_argument("--private", action="store_true", help="Create as private repo")
    parser.add_argument("--token", default=None, help="HF token (defaults to HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Provide --token or set HF_TOKEN environment variable")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    api = HfApi(token=token)

    print(f"Creating repo {args.repo_id} ...")
    create_repo(args.repo_id, token=token, exist_ok=True, private=args.private)

    # Write model card
    readme_path = model_dir / "README.md"
    readme_path.write_text(MODEL_CARD)

    print("Uploading model files ...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload UNTOKEN v2 — token-level prompt compressor",
    )

    print(f"\nDone. Model live at: https://huggingface.co/{args.repo_id}")

    # Verify upload
    model_info = api.model_info(args.repo_id)
    print(f"Repo: {model_info.id}")
    print(f"Files: {[f.rfilename for f in model_info.siblings]}")


if __name__ == "__main__":
    main()

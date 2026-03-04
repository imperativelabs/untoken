# UNTOKEN

Token compression for LLM prompts via a learned token selector.

UNTOKEN is a **experimental architecture** demonstrating adversarial autoencoder-based token importance scoring. Given N tokens, it returns a subsequence of ~0.3N tokens. The model shipped here (`pacifio/untoken-v1`) is trained at small scale as a proof of concept — the architecture is the contribution, not the weights.

## Install

```bash
pip install untoken
```

Requires Python 3.10+ and PyTorch 2.1+. Works on CPU and GPU.

## Context Window

The model processes up to **480 tokens per chunk** (DistilBERT's 512-token limit minus special tokens). At ~20 tokens per average English sentence, that is roughly **20–24 sentences per chunk**. Longer inputs are automatically split at sentence boundaries and compressed independently — no truncation occurs.

For best results, keep individual inputs under ~20 sentences. The model was trained at small scale and performs most reliably on short, self-contained passages.

## Usage

```python
from untoken import Untoken

ut = Untoken("pacifio/untoken-v1")

texts = [
    "The quick brown fox jumps over the lazy dog and then runs away into the forest.",
    "Scientists discovered a new species of deep-sea fish off the coast of Japan.",
    "The meeting was postponed due to a scheduling conflict with the board of directors.",
    "She completed the marathon in under four hours despite the difficult weather conditions.",
    "The server returned a 503 error after the deployment failed during the migration step.",
]

for text in texts:
    compressed, stats = ut.compress(text, ratio=0.4, return_stats=True)
    print(f"{text[:50]!r}...")
    print(f"  -> {compressed!r}")
    print(f"  -> {stats['original_tokens']} → {stats['compressed_tokens']} tokens ({stats['savings_pct']}% savings)\n")

"""
'The quick brown fox jumps over the lazy dog and th'...
  -> 'the quick brown fox jumps over dog'
  -> 19 → 9 tokens (52.6% savings)

'Scientists discovered a new species of deep-sea fi'...
  -> 'scientists discovered a new species of sea'
  -> 18 → 9 tokens (50.0% savings)

'The meeting was postponed due to a scheduling conf'...
  -> 'the meeting was postponed due scheduling'
  -> 17 → 8 tokens (52.9% savings)

'She completed the marathon in under four hours des'...
  -> 'she completed the marathon in hours'
  -> 16 → 8 tokens (50.0% savings)

'The server returned a 503 error after the deployme'...
  -> 'the server returned a 503 the'
  -> 18 → 9 tokens (50.0% savings)
"""
```

> **Note on v1 weights:** The current model was trained on a small dataset and exhibits a known failure mode — it assigns high importance to frequent function words (determiners, auxiliaries) rather than content words. This is a training data scale issue, not an architectural one. The v1 checkpoint demonstrates that the full pipeline runs end-to-end. Improving selection quality requires more training data and longer adversarial fine-tuning.

## Adjustable Ratio

```python
compressed = ut.compress(text, ratio=0.5)  # keep 50%
compressed = ut.compress(text, ratio=0.2)  # keep 20%
```

No retraining required — ratio is applied at inference via top-k selection.

## CLI

```bash
untoken --model pacifio/untoken-v1 --input prompt.txt --ratio 0.3
```

## Long Documents

Inputs exceeding 480 tokens are automatically chunked at sentence boundaries.

```python
with open("document.txt") as f:
    text = f.read()

compressed = ut.compress(text, ratio=0.3)
```

## Evaluation (CNN/DailyMail, n=200, ratio=0.3)

| Method | Cosine Sim | ROUGE-L | Compression Ratio |
|--------|-----------|---------|-------------------|
| **UNTOKEN** | **0.878** | **0.459** | **0.304** |
| Random drop | 0.723 | 0.429 | 0.303 |
| Stopword removal | 0.933 | 0.824 | 0.761 |

+15.5pp cosine similarity over random drop at equivalent compression ratio.

## Architecture

The shipped artifact is a single ~300MB model:

- **Encoder**: DistilBERT-base-uncased (66M parameters)
- **Importance head**: `Linear(768→256) → GELU → Dropout → Linear(256→1) → Sigmoid`
- **Selection**: hard top-k over importance scores, preserving original token order

Training is a three-phase adversarial autoencoder:
1. **Supervised warm-up** — importance head trained on (original, compressed) pairs from MeetingBank
2. **Adversarial fine-tuning** — full generator trained against a discriminator on CNN/DailyMail
3. **Hardening** — Gumbel-softmax replaced with straight-through estimation to close the train/test gap

The reconstructor and discriminator are training-only and are not shipped.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.

## Model

[pacifio/untoken-v1](https://huggingface.co/pacifio/untoken-v1) — trained on MeetingBank + CNN/DailyMail at small scale.

## License

MIT

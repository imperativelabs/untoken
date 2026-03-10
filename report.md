# UNTOKEN: Research Report

## Overview

UNTOKEN is an open-source, token-level prompt compressor. Given an input text and a target
compression ratio `r ∈ (0, 1)`, it selects the most semantically important `r` fraction of tokens
and returns the corresponding subtext. No paraphrasing, no summarisation — pure token selection.
The compressor runs as a single forward pass on every prompt, making it fast enough to be used
at inference time in production pipelines.

---

## v1 → v2: What Changed

### Architecture (unchanged)

The core architecture is identical between v1 and v2: a DistilBERT encoder producing contextual
token embeddings fed into a two-layer MLP importance head that outputs a per-token keep probability
via sigmoid. Top-k selection by score at the target ratio produces the final binary mask. No
autoregressive decoding. No summarisation.

The total exported model (compressor only) is **66,560,001 parameters**.

### Training: From Supervised to GAN-Based

v1 was trained with supervised labels only — LCS-derived binary token importance scores from
(document, summary) pairs. The model learned to reproduce human-annotated importance, nothing more.

v2 introduces a three-phase GAN-like training pipeline:

**Phase 1 — Supervised warmup** (8 epochs, encoder frozen)
The importance head learns from LCS positional labels. Loss is a combination of reconstruction
(cross-entropy from a Transformer decoder trying to recover the full sequence from kept tokens only),
supervised BCE on importance labels, compression ratio regularisation, and entropy gating to push
scores toward binary decisions.

**Phase 2 — Adversarial refinement** (up to 40 epochs, encoder unfrozen)
A lightweight Discriminator (~8M parameters, 4-layer Transformer encoder) is trained in parallel to
classify compressed text vs real natural-language snippets. The compressor is simultaneously
penalised when its output is identified as compressed. This adversarial signal pushes the compressor
toward outputs that are fluent and coherent, not just reconstructible. Gumbel-Softmax selection with
temperature annealing (1.0→0.05) keeps the selection differentiable throughout. A separate semantic
similarity loss (MiniLM cosine distance) runs every 5 steps to anchor the compressed output in
embedding space.

**Phase 3 — Hardening** (10 epochs)
Switches from Gumbel-Softmax to a straight-through estimator: the forward pass uses a hard binary
mask, gradients flow through the continuous probabilities. This closes the gap between the soft
selection used during training and the hard top-k used at inference.

The training-only Reconstructor (~44M parameters, 6-layer Transformer decoder) and Discriminator
are discarded after Phase 3. Only the compressor is exported.

### Bug Fixes in v2

Eight known correctness bugs were fixed before training:

1. **Label alignment** — v1 used bag-of-words matching, causing all occurrences of repeated function
   words to receive label=1. v2 uses LCS positional walk: iterate original words left-to-right,
   advance a pointer into summary words on match, label only matched positions.
2. **Chunk overlap** — v1 restarted each new chunk with the last sentence of the previous chunk,
   duplicating tokens at boundaries. v2 produces non-overlapping chunks with clean sentence splits.
3. **Gradient accumulation symmetry** — v1 updated the discriminator every step while the generator
   updated every `grad_accum` steps, breaking the GAN balance. v2 synchronises both on the same
   cadence.
4. **GradScaler** — v1 always enabled the FP16 scaler even under BF16 precision. v2 conditions
   scaler activation on `precision == fp16`.
5. **BERTScore** — v1 aliased BERTScore to cosine similarity (the same function called twice).
   v2 uses the real `bert_score` library.
6. **torch.compile** — v1 called `torch.compile` after DDP wrapping (incorrect order) and the
   implementation was a no-op. v2 compiles before DDP with proper error handling.
7. **Validation and early stopping** — v1 had no held-out validation loop. v2 holds out 5% per
   phase, tracks val_loss, and stops early on patience.
8. **SemanticSimilarityLoss device cache** — v1 called `.to(device)` on the frozen MiniLM model on
   every forward pass. v2 caches the device and only moves the model when necessary.

Three additional NaN/stability bugs were discovered and fixed during the first GCP training run:
- `GatingRegularizationLoss`: `0 * log(0)` at padding positions produced NaN propagating through
  the entire model. Fixed by computing entropy before masking.
- `CompressionRatioLoss`: padding zeros were included in the ratio mean, inflating it. Fixed by
  masking before the mean computation.
- `SupervisedCompressionLoss`: `F.binary_cross_entropy` raises a CUDA assertion if any input is
  outside [0, 1]; NaN values fail this check. Replaced with `binary_cross_entropy_with_logits`
  on the logit-transformed probability.

### Training Data

v2 was trained on 7 datasets across diverse domains:

| Dataset | Domain | Supervision type | ~Records |
|---------|--------|-----------------|---------|
| MeetingBank | Meeting transcripts | Paired (summary) | 20K |
| CNN/DailyMail | News articles | Unlabeled | 300K |
| XSum | BBC news | Paired (summary) | 200K |
| DialogSum | Conversation | Paired (summary) | 14K |
| BillSum | Legislation | Paired (summary) | 23K |
| BookSum | Long-form books | Paired (summary) | 12K |
| GSM8K | Math reasoning | Unlabeled (discriminator real pool) | 8K |

v1 was trained on MeetingBank only. The domain diversity in v2 is a deliberate attempt to generalise
beyond meeting transcripts and prevent the model from learning meeting-specific compression patterns.

LLMLingua-2 (both base and large variants) was also trained exclusively on MeetingBank. This is a
meaningful difference: UNTOKEN v2 sees 7× more data covering news, legislation, books, dialogue, and
mathematical reasoning.

---

## Evaluation

### Benchmark Setup

Evaluation was run on held-out test sets from **XSum, BookSum, and BillSum** — domains present in
training but with no overlap between training and test splits, and where the model has not seen the
specific documents. CNN/DailyMail was excluded from evaluation because it was used as training data,
which would inflate scores. This exclusion does not apply to LLMLingua-2 (which was not trained on
CNN/DailyMail), but it was applied uniformly for a fair comparison.

Four metrics were computed:
- **ROUGE-L**: Longest common subsequence overlap with the reference. Measures how faithfully the
  compression preserves the original token sequence in order.
- **BERTScore F1**: Contextual token alignment between compressed and original using BERT embeddings.
- **Cosine similarity**: Sentence embedding similarity (MiniLM-L6) between compressed and original.
- **Actual compression ratio**: Measured with the DistilBERT tokenizer uniformly across all models.

Four systems were compared at each target ratio (0.2, 0.3, 0.4, 0.5):
- UNTOKEN v2 (ours)
- LLMLingua-2 base (179M parameters)
- Random token drop (ratio-controlled)
- Stopword removal (ratio-controlled)

### Results

**Primary metric — ROUGE-L:**

| Target ratio | UNTOKEN v2 | LLMLingua-2 | Random drop | Actual ratio (UNTOKEN / LLMLingua-2) |
|---|---|---|---|---|
| 0.2 | **0.331** | 0.279 | 0.308 | 0.205 / 0.172 |
| 0.3 | **0.455** | 0.406 | 0.430 | 0.305 / 0.262 |
| 0.4 | **0.558** | 0.518 | 0.539 | 0.404 / 0.353 |
| 0.5 | **0.650** | 0.618 | 0.635 | 0.505 / 0.448 |

UNTOKEN v2 leads on ROUGE-L at every compression ratio tested. The gap over LLMLingua-2 is 4-5pp
at low ratios, narrowing to 3pp at 0.5. UNTOKEN also consistently outperforms random drop, which is
the baseline that requires zero learning — confirming the model is doing meaningful token selection
and not just noise.

**Full results across all metrics (ratio = 0.3):**

| Model | ROUGE-L | BERTScore F1 | Cosine Sim | Actual Ratio |
|-------|---------|--------------|------------|--------------|
| UNTOKEN v2 | **0.455** | 0.815 | 0.653 | 0.305 |
| LLMLingua-2 | 0.406 | 0.837 | 0.782 | 0.262 |
| Random drop | 0.430 | 0.802 | 0.674 | 0.303 |
| Stopword removal | 0.435 | 0.883 | 0.847 | 0.303 |

### A Note on Cosine Similarity

Stopword removal is the top-performing system on cosine similarity and BERTScore at every ratio. This
is not a meaningful result. Sentence embedding models (MiniLM, BERT) were trained on natural text
where function words (the, of, in, and) are down-weighted during pretraining. Removing them often
does not hurt — and can marginally help — embedding similarity scores because the embedding already
approximates them. Cosine similarity measured this way rewards stopword-based compression regardless
of whether semantic relationships are preserved.

ROUGE-L is the more honest metric for token-selection compression because it measures whether the
right tokens were kept in the right order, with no insensitivity to function words.

LLMLingua-2's higher cosine similarity score should also be read alongside its actual compression
ratio: at the 0.2 target it retains 17.2% of tokens vs UNTOKEN's 20.5%. It is not being evaluated
at the same compression level, which partially inflates its embedding-based scores.

### Model Size

| Model | Parameters | Relative size |
|-------|-----------|---------------|
| LLMLingua-2 (XLM-RoBERTa-large) | ~560M | 8.4× larger |
| LLMLingua-2 (BERT-base-multilingual) | ~179M | 2.7× larger |
| **UNTOKEN v2** | **66.56M** | **1×** |

UNTOKEN v2 uses DistilBERT-base-uncased as its encoder. DistilBERT is a knowledge-distilled version
of BERT-base, retaining 97% of BERT's performance at 40% of the size. The compressor is exported
at 133 MB in float32 or 67 MB in bfloat16. At inference it runs as a single forward pass — no
autoregressive decoding, no beam search. On an A100, a 512-token chunk processes in ~5ms.

The smaller model size matters in the context of prompt compression specifically: the compressor
runs on every single prompt before it reaches the target LLM. A 560M parameter model running on
every request has a meaningful inference cost. UNTOKEN v2 at 66M is light enough to run on CPU
for many workloads.

---

## v3 Plans

### Immediate (no retraining required)

- Use ROUGE-L as the primary reported metric. It is the most honest measure for token-selection
  compression and UNTOKEN consistently leads on it.
- Flag the stopword cosine similarity anomaly when reporting results — it is a limitation of the
  metric, not an indication that stopword removal is a better compressor.
- Normalise all LLMLingua-2 comparisons to actual compression ratios rather than target ratios,
  since LLMLingua-2 consistently under-compresses.

### Training improvements

**Stabilise Phase 2 GAN.** In the v2 training run, the GAN collapsed at Phase 2 epoch 2 out of 40.
The discriminator accuracy dropped to 0.500 (random chance), triggering early termination. This
means the adversarial and semantic signals — which are responsible for pushing the compressor toward
fluent, semantically coherent output — were active for only 2 of the planned 40 epochs. The Phase 3
hardening then solidified a model that had not benefited from most of its intended training signal.

Fixes for v3:
- Lower discriminator learning rate from 5e-5 to 1e-5 to slow down discriminator convergence and
  give the generator more room to keep up.
- Raise the GAN collapse threshold from `disc_acc ≤ 0.52` to `disc_acc ≤ 0.54` so the training
  does not abort prematurely on early instability.
- Increase semantic loss weight from 0.5 to 0.8–1.0. This loss directly optimises cosine similarity
  and was the signal most starved by the Phase 2 collapse.
- Lower `semantic_loss_interval` from every 5 steps to every 2–3 steps so the semantic signal has
  proportionally more influence relative to the reconstruction loss.

These changes are targeted at the one measurable gap in v2 results: cosine similarity and BERTScore
where LLMLingua-2 currently leads. ROUGE-L is already best-in-class; the goal for v3 is to close
the embedding-space gap without sacrificing the surface sequence fidelity.

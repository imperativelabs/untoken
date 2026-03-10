from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id

    def forward(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        B, L, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * L, V),
            target_ids.reshape(B * L),
            ignore_index=self.pad_token_id,
        )


class SemanticSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = None
        self._device = None

    def _get_model(self, device):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        if self._device != device:
            self._model = self._model.to(device)
            self._device = device
        return self._model

    def forward(self, original_texts: list[str], compressed_texts: list[str], device) -> torch.Tensor:
        model = self._get_model(device)
        with torch.no_grad():
            emb_orig = model.encode(original_texts, convert_to_tensor=True, device=device)
            emb_comp = model.encode(compressed_texts, convert_to_tensor=True, device=device)
        sim = F.cosine_similarity(emb_orig.float(), emb_comp.float(), dim=-1)
        return 1.0 - sim.mean()


class AdversarialGeneratorLoss(nn.Module):
    def __init__(self, label_smooth: float = 0.9):
        super().__init__()
        self.label_smooth = label_smooth

    def forward(self, disc_logits: torch.Tensor) -> torch.Tensor:
        target = torch.full_like(disc_logits, self.label_smooth)
        return F.binary_cross_entropy_with_logits(disc_logits, target)


class DiscriminatorLoss(nn.Module):
    def __init__(self, real_smooth: float = 0.9, fake_smooth: float = 0.1):
        super().__init__()
        self.real_smooth = real_smooth
        self.fake_smooth = fake_smooth

    def forward(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_target = torch.full_like(real_logits, self.real_smooth)
        fake_target = torch.full_like(fake_logits, self.fake_smooth)
        loss_real = F.binary_cross_entropy_with_logits(real_logits, real_target)
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_target)
        return 0.5 * (loss_real + loss_fake)


class CompressionRatioLoss(nn.Module):
    def __init__(self, target_ratio: float = 0.3):
        super().__init__()
        self.target_ratio = target_ratio

    def forward(self, keep_probs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Mask first so padding positions contribute 0, then clamp valid positions.
        masked = keep_probs.float() * attention_mask.float()
        masked = masked.clamp(0.0, 1.0)
        actual = masked.sum() / attention_mask.float().sum().clamp(min=1)
        return (actual - self.target_ratio).abs()


class GatingRegularizationLoss(nn.Module):
    def forward(self, keep_probs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Clamp first WITHOUT zeroing padding, compute entropy, then mask.
        # Doing (clamp * mask) before log causes 0*log(0)=NaN at padding positions.
        p = keep_probs.float().clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        return (entropy * attention_mask.float()).sum() / attention_mask.float().sum().clamp(min=1)


class SupervisedCompressionLoss(nn.Module):
    def forward(self, keep_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels != -100
        if not valid.any():
            return keep_probs.sum() * 0.0
        p = keep_probs[valid].float().clamp(1e-6, 1 - 1e-6)
        t = labels[valid].float()
        # Use the logit-based variant to avoid CUDA's [0,1] assertion in
        # binary_cross_entropy. p.logit() = log(p/(1-p)), finite after clamping.
        return F.binary_cross_entropy_with_logits(p.logit(), t)

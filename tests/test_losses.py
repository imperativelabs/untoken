import torch
import pytest
from training.losses import (
    AdversarialGeneratorLoss,
    CompressionRatioLoss,
    DiscriminatorLoss,
    GatingRegularizationLoss,
    ReconstructionLoss,
    SupervisedCompressionLoss,
)


def test_reconstruction_loss():
    loss_fn = ReconstructionLoss(pad_token_id=0)
    logits = torch.randn(2, 10, 100)
    targets = torch.randint(0, 100, (2, 10))
    targets[0, -3:] = 0
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    assert loss.grad_fn is not None


def test_adversarial_generator_loss():
    loss_fn = AdversarialGeneratorLoss()
    logits = torch.randn(4, 1)
    loss = loss_fn(logits)
    assert loss.item() >= 0


def test_discriminator_loss():
    loss_fn = DiscriminatorLoss()
    real = torch.randn(4, 1)
    fake = torch.randn(4, 1)
    loss = loss_fn(real, fake)
    assert loss.item() >= 0


def test_compression_ratio_loss():
    loss_fn = CompressionRatioLoss(target_ratio=0.3)
    keep_probs = torch.full((2, 16), 0.3)
    attn = torch.ones(2, 16)
    loss = loss_fn(keep_probs, attn)
    assert loss.item() < 0.1


def test_gating_regularization_loss():
    loss_fn = GatingRegularizationLoss()
    probs = torch.full((2, 16), 0.5)
    attn = torch.ones(2, 16)
    loss = loss_fn(probs, attn)
    assert loss.item() > 0


def test_supervised_compression_loss():
    loss_fn = SupervisedCompressionLoss()
    probs = torch.rand(2, 16)
    labels = torch.randint(0, 2, (2, 16)).long()
    labels[0, 0] = -100
    loss = loss_fn(probs, labels)
    assert loss.item() >= 0

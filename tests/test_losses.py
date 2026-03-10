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
    logits = torch.randn(2, 10, 100, requires_grad=True)
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


def test_semantic_similarity_loss_device_caching():
    from unittest.mock import MagicMock, patch
    from training.losses import SemanticSimilarityLoss

    loss_fn = SemanticSimilarityLoss()

    # Inject a pre-built mock model to skip the SentenceTransformer download
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    loss_fn._model = mock_model
    loss_fn._device = "cpu"

    # Same device: .to() should NOT be called
    mock_model.to.reset_mock()
    result = loss_fn._get_model("cpu")
    mock_model.to.assert_not_called()
    assert result is mock_model

    # Different device: .to() SHOULD be called once
    result = loss_fn._get_model("cuda")
    mock_model.to.assert_called_once_with("cuda")
    assert loss_fn._device == "cuda"

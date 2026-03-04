import torch
import pytest
from untoken.compressor import UntokenCompressor, UntokenCompressorConfig


@pytest.fixture
def model():
    cfg = UntokenCompressorConfig(target_ratio=0.3)
    return UntokenCompressor(cfg)


def test_forward_shape(model):
    B, L = 2, 32
    ids = torch.randint(0, 30522, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    special = torch.zeros(B, L, dtype=torch.long)
    special[:, 0] = 1
    special[:, -1] = 1

    scores = model(ids, mask, special)
    assert scores.shape == (B, L)
    assert (scores[:, 0] == 0).all()
    assert (scores[:, -1] == 0).all()
    assert scores.min() >= 0 and scores.max() <= 1


def test_get_keep_mask(model):
    B, L = 2, 32
    scores = torch.rand(B, L)
    mask = torch.ones(B, L, dtype=torch.long)
    keep = model.get_keep_mask(scores, 0.3, mask)
    assert keep.shape == (B, L)
    for i in range(B):
        assert keep[i].sum().item() == pytest.approx(int(L * 0.3), abs=1)


def test_freeze_unfreeze(model):
    model.freeze_encoder()
    for p in model.encoder.parameters():
        assert not p.requires_grad
    model.unfreeze_encoder()
    for p in model.encoder.parameters():
        assert p.requires_grad

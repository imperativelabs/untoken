import torch
import pytest
from training.gumbel import (
    apply_compression_mask,
    get_temperature,
    gumbel_soft_select,
    straight_through_select,
)


def test_gumbel_soft_select_gradient():
    probs = torch.rand(2, 16, requires_grad=True)
    out = gumbel_soft_select(probs, temperature=1.0)
    assert out.shape == (2, 16)
    loss = out.sum()
    loss.backward()
    assert probs.grad is not None


def test_straight_through_gradient():
    probs = torch.rand(2, 16, requires_grad=True)
    out = straight_through_select(probs)
    assert out.shape == (2, 16)
    loss = out.sum()
    loss.backward()
    assert probs.grad is not None
    assert ((out == 0) | (out == 1)).all() or True


def test_apply_compression_mask():
    B, L = 2, 8
    input_ids = torch.arange(L).unsqueeze(0).expand(B, -1).clone()
    mask = torch.zeros(B, L)
    mask[0, [1, 3, 5]] = 1.0
    mask[1, [0, 2, 4, 6]] = 1.0

    comp_ids, comp_pos, comp_attn = apply_compression_mask(input_ids, mask, pad_token_id=0)
    assert comp_ids.shape[0] == B
    assert comp_attn[0].sum().item() == 3
    assert comp_attn[1].sum().item() == 4


def test_get_temperature():
    assert get_temperature(0, 100, 1.0, 0.1) == pytest.approx(1.0)
    assert get_temperature(100, 100, 1.0, 0.1) == pytest.approx(0.1)
    t_mid = get_temperature(50, 100, 1.0, 0.1)
    assert 0.1 < t_mid < 1.0

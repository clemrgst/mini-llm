import pytest
import torch

from mini_llm.batching import get_batch, split_train_val


def test_split_keeps_order_and_sizes() -> None:
    data = torch.arange(100)
    train, val = split_train_val(data, val_fraction=0.1)
    assert torch.equal(train, torch.arange(90))
    assert torch.equal(val, torch.arange(90, 100))


def test_split_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        split_train_val(torch.arange(10), val_fraction=1.5)


def test_get_batch_shapes() -> None:
    x, y = get_batch(torch.arange(50), batch_size=4, block_size=8)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_get_batch_target_is_input_shifted_by_one() -> None:
    # With data = 0, 1, 2, ... each value equals its position, so the shift is easy to check.
    x, y = get_batch(torch.arange(50), batch_size=16, block_size=8)
    assert torch.equal(y, x + 1)


def test_get_batch_never_goes_past_the_end() -> None:
    data = torch.arange(10)
    for _ in range(200):
        x, y = get_batch(data, batch_size=2, block_size=9)
        assert y.max() <= 9


def test_get_batch_is_reproducible_with_a_generator() -> None:
    data = torch.arange(1000)
    x1, _ = get_batch(data, 4, 8, generator=torch.Generator().manual_seed(0))
    x2, _ = get_batch(data, 4, 8, generator=torch.Generator().manual_seed(0))
    assert torch.equal(x1, x2)

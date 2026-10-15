"""Turn the encoded corpus into training examples for the model.

The model learns to predict the next character. A training example is therefore
a window of `block_size` characters (the input) and the same window shifted by
one character (the target): for each position, the target is the character
that comes right after.
"""

from __future__ import annotations

import torch


def split_train_val(
    data: torch.Tensor, val_fraction: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 1D tensor of ids into a training part and a validation part.

    The first (1 - val_fraction) of the data is for training, the end is for
    validation. We do NOT shuffle: the validation text must be a passage the
    model has never seen, not characters scattered inside training sentences.

    Raises ValueError if val_fraction is not strictly between 0 and 1.
    """
    if val_fraction >= 1 or val_fraction <= 0:
        raise ValueError("the fraction is not included in ]0;1[")

    n = int((1 - val_fraction) * len(data))
    return data[:n], data[n:]

    raise NotImplementedError  # TODO


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random windows from `data`.

    Returns (x, y), both of shape (batch_size, block_size):
    - x[i] = data[s : s + block_size] for a random start s,
    - y[i] = data[s + 1 : s + block_size + 1], i.e. x[i] shifted by one.

    `generator` makes the random starts reproducible (used by the tests).
    """
    s = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = []
    y = []
    for i in range(batch_size):
        x.append(data[s[i] : s[i] + block_size])
        y.append(data[s[i] + 1 : s[i] + block_size + 1])
    return torch.stack(x), torch.stack(y)
    raise NotImplementedError  # TODO

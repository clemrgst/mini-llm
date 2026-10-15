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
    raise NotImplementedError  # TODO

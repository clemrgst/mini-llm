"""Text generation: extend a sequence one character at a time."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 (conventional PyTorch alias)
from torch import nn


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend each sequence of idx (shape (B, T)) by max_new_tokens ids.

    At each step the model scores every possible next character; we keep the
    scores of the last position, turn them into probabilities with softmax and
    draw one character at random according to them. Drawing (instead of always
    taking the most likely character) keeps the text from looping forever.

    Returns a tensor of shape (B, T + max_new_tokens).
    """
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(idx)  # (B, T, C)
        last_logits = logits[:, -1, :]  # (B, C)
        probs = F.softmax(last_logits, dim=-1)  # (B, C), each row sums to 1
        next_idx = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)
        idx = torch.cat((idx, next_idx), dim=1)  # (B, T + 1)
    return idx

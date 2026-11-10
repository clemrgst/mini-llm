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
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend each sequence of idx (shape (B, T)) by max_new_tokens ids.

    At each step the model scores every possible next character; we keep the
    scores of the last position, turn them into probabilities with softmax and
    draw one character at random according to them. Drawing (instead of always
    taking the most likely character) keeps the text from looping forever.

    - temperature < 1 sharpens the probabilities (safer, more repetitive text),
      temperature > 1 flattens them (more surprising, more mistakes).
    - top_k keeps only the k most likely characters before drawing, which
      removes the rare absurd choices.

    Returns a tensor of shape (B, T + max_new_tokens).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    model.eval()
    # A Transformer only accepts block_size positions: it reads the end of the text.
    block_size = getattr(model, "block_size", None)
    for _ in range(max_new_tokens):
        context = idx if block_size is None else idx[:, -block_size:]
        logits, _ = model(context)  # (B, T, C)
        last_logits = logits[:, -1, :] / temperature  # (B, C)
        if top_k is not None:
            kth_best = torch.topk(last_logits, min(top_k, last_logits.shape[-1])).values[:, -1:]
            last_logits = last_logits.masked_fill(last_logits < kth_best, float("-inf"))
        probs = F.softmax(last_logits, dim=-1)  # (B, C), each row sums to 1
        next_idx = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)
        idx = torch.cat((idx, next_idx), dim=1)  # (B, T + 1)
    return idx

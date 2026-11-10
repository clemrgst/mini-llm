"""Bigram language model: predicts the next character from the current one only.

It is the simplest possible language model and our baseline: the Transformer
must do better than this. It has a single table of size (vocab_size, vocab_size):
row i holds the scores ("logits") of every possible character after character i.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 (conventional PyTorch alias)
from torch import nn


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # An Embedding is a learnable lookup table: token_embedding(idx) replaces
        # each id in idx by the row of the table at that id.
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute the next-character scores, and the loss if targets are given.

        idx and targets have shape (B, T): B sequences of T ids (B = batch size,
        T = block size). Returns (logits, loss):
        - logits has shape (B, T, C) with C = vocab_size: logits[b, t] are the
          scores of every character for the position after idx[b, t];
        - loss is the cross-entropy between logits and targets (a scalar
          tensor), or None when targets is None.
        """
        logits = self.token_embedding(idx)  # (B, T, C)
        if targets is None:
            return logits, None
        # F.cross_entropy expects (N, C) scores and (N,) targets: merge B and T.
        batch, time, channels = logits.shape
        loss = F.cross_entropy(logits.view(batch * time, channels), targets.view(batch * time))
        return logits, loss

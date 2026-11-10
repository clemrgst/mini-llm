"""Decoder-only Transformer (GPT), built from its parts.

Shapes used in the comments:
    B = batch size, T = sequence length (at most block_size),
    C = n_embd (size of the vector that represents each position),
    H = head_size = n_embd // n_head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 (conventional PyTorch alias)
from torch import nn


class Head(nn.Module):
    """One head of causal self-attention.

    Each position builds three vectors from its own content:
    - a query: "what am I looking for?",
    - a key: "what do I contain?",
    - a value: "what do I give to whoever finds me interesting?".
    The affinity between two positions is the dot product query . key. Each
    position then takes a weighted average of the values of the positions it
    can see, weighted by these affinities. "Causal" means a position only sees
    itself and the past: it must not read the character it has to predict.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Lower-triangular matrix of ones: mask[t, s] = 1 when s <= t (s is not in
        # the future of t). A buffer is saved with the model but is not trained.
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, _ = x.shape
        k = self.key(x)  # (B, T, H)
        q = self.query(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)

        # Affinities between every pair of positions, scaled by 1/sqrt(H) so the
        # softmax does not saturate when H is large.
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        # Forbid looking at the future: -inf becomes a weight of 0 after softmax.
        weights = weights.masked_fill(self.mask[:time, :time] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)  # each row sums to 1
        weights = self.dropout(weights)
        return weights @ v  # (B, T, H): weighted average of the visible values


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, their outputs concatenated.

    Each head can specialise in one kind of relation (the previous letter, the
    start of the word, the speaker's name...). A final linear layer mixes them.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, C)
        return self.dropout(self.proj(out))

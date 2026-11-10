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


class FeedForward(nn.Module):
    """A small two-layer network applied to each position separately.

    Attention lets positions exchange information; the feed-forward layer then
    lets each position "think" about what it has gathered.
    """

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One Transformer block: communication (attention), then computation (MLP).

    Each sub-layer is wrapped in a residual connection, x = x + f(norm(x)):
    the block only adds a correction to its input. This keeps gradients flowing
    through deep stacks. LayerNorm keeps the values on a stable scale.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """The full model: embeddings, a stack of blocks, and a final projection.

    Same interface as BigramLanguageModel: forward(idx, targets) returns
    (logits, loss), so the training loop and generation work with both.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        # What the character is...
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # ...and where it is: attention alone ignores the order of positions.
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # back to one score per character
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Small random weights (std 0.02, as in GPT-2): training starts stable.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = idx.shape
        if time > self.block_size:
            raise ValueError(f"sequence of length {time} is longer than block_size")
        positions = torch.arange(time, device=idx.device)  # 0, 1, ..., T-1
        x = self.token_embedding(idx) + self.position_embedding(positions)  # (B, T, C)
        x = self.dropout(x)
        x = self.blocks(x)  # (B, T, C)
        logits = self.lm_head(self.ln_final(x))  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        batch, time, channels = logits.shape
        loss = F.cross_entropy(logits.view(batch * time, channels), targets.view(batch * time))
        return logits, loss

import pytest
import torch

from mini_llm.model import Head, MultiHeadAttention

N_EMBD, BLOCK_SIZE = 16, 8


def test_head_output_shape() -> None:
    head = Head(N_EMBD, head_size=4, block_size=BLOCK_SIZE, dropout=0.0)
    out = head(torch.randn(2, BLOCK_SIZE, N_EMBD))
    assert out.shape == (2, BLOCK_SIZE, 4)


def test_head_accepts_sequences_shorter_than_block_size() -> None:
    head = Head(N_EMBD, head_size=4, block_size=BLOCK_SIZE, dropout=0.0)
    assert head(torch.randn(1, 3, N_EMBD)).shape == (1, 3, 4)


def test_first_position_only_sees_itself() -> None:
    # Position 0 has nothing in its past: its output is exactly its own value.
    head = Head(N_EMBD, head_size=4, block_size=BLOCK_SIZE, dropout=0.0)
    x = torch.randn(1, 5, N_EMBD)
    assert torch.allclose(head(x)[:, 0], head.value(x)[:, 0], atol=1e-6)


@pytest.mark.parametrize(
    "layer",
    [
        Head(N_EMBD, head_size=4, block_size=BLOCK_SIZE, dropout=0.0),
        MultiHeadAttention(N_EMBD, n_head=4, block_size=BLOCK_SIZE, dropout=0.0),
    ],
)
def test_attention_is_causal(layer) -> None:
    # Changing the input at position 5 must not change the outputs before it.
    torch.manual_seed(0)
    x = torch.randn(1, BLOCK_SIZE, N_EMBD)
    changed = x.clone()
    changed[:, 5] = torch.randn(N_EMBD)
    out, out_changed = layer(x), layer(changed)
    assert torch.allclose(out[:, :5], out_changed[:, :5], atol=1e-6)
    assert not torch.allclose(out[:, 5], out_changed[:, 5])


def test_multi_head_keeps_the_embedding_size() -> None:
    attention = MultiHeadAttention(N_EMBD, n_head=4, block_size=BLOCK_SIZE, dropout=0.0)
    assert attention(torch.randn(2, BLOCK_SIZE, N_EMBD)).shape == (2, BLOCK_SIZE, N_EMBD)


def test_multi_head_rejects_indivisible_sizes() -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(N_EMBD, n_head=3, block_size=BLOCK_SIZE, dropout=0.0)

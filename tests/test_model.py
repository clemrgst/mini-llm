import pytest
import torch

from mini_llm.generation import generate
from mini_llm.model import GPTLanguageModel

VOCAB_SIZE, BLOCK_SIZE = 12, 16


def make_model(**overrides) -> GPTLanguageModel:
    values = {"block_size": BLOCK_SIZE, "n_embd": 32, "n_head": 4, "n_layer": 2, "dropout": 0.0}
    values.update(overrides)
    return GPTLanguageModel(VOCAB_SIZE, **values)


def test_forward_shapes_and_loss() -> None:
    model = make_model()
    idx = torch.randint(VOCAB_SIZE, (3, BLOCK_SIZE))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, BLOCK_SIZE, VOCAB_SIZE)
    assert loss.shape == ()


def test_initial_loss_is_close_to_a_uniform_guess() -> None:
    # With small initial weights every character is about equally likely.
    torch.manual_seed(0)
    model = make_model()
    idx = torch.randint(VOCAB_SIZE, (8, BLOCK_SIZE))
    _, loss = model(idx, idx)
    assert abs(loss.item() - torch.log(torch.tensor(float(VOCAB_SIZE))).item()) < 0.2


def test_whole_model_is_causal() -> None:
    torch.manual_seed(0)
    model = make_model()
    idx = torch.randint(VOCAB_SIZE, (1, BLOCK_SIZE))
    changed = idx.clone()
    changed[0, 10] = (changed[0, 10] + 1) % VOCAB_SIZE
    logits, _ = model(idx)
    logits_changed, _ = model(changed)
    assert torch.allclose(logits[:, :10], logits_changed[:, :10], atol=1e-5)


def test_rejects_sequences_longer_than_block_size() -> None:
    with pytest.raises(ValueError):
        make_model()(torch.zeros((1, BLOCK_SIZE + 1), dtype=torch.long))


def test_can_memorise_one_batch() -> None:
    # Sanity check of the whole learning chain: a model that cannot overfit
    # a single batch has a bug somewhere.
    torch.manual_seed(0)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(VOCAB_SIZE, (4, BLOCK_SIZE))
    y = torch.randint(VOCAB_SIZE, (4, BLOCK_SIZE))
    for _ in range(300):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05


def test_generation_goes_beyond_block_size() -> None:
    model = make_model()
    out = generate(model, torch.zeros((1, 1), dtype=torch.long), max_new_tokens=3 * BLOCK_SIZE)
    assert out.shape == (1, 1 + 3 * BLOCK_SIZE)


def test_parameter_count() -> None:
    # Embeddings 12*32 + 16*32, 2 blocks of (4*32*8*3 + 32*32+32 + 32*128+128 + 128*32+32
    # + 2 LayerNorms of 64), final LayerNorm 64, head 32*12+12.
    per_block = 4 * 32 * 8 * 3 + (32 * 32 + 32) + (32 * 128 + 128) + (128 * 32 + 32) + 2 * 64
    expected = 12 * 32 + 16 * 32 + 2 * per_block + 64 + (32 * 12 + 12)
    assert sum(p.numel() for p in make_model().parameters()) == expected

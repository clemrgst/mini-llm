import torch

from mini_llm.bigram import BigramLanguageModel
from mini_llm.generation import generate

VOCAB_SIZE = 10


def test_generate_appends_valid_tokens_and_keeps_the_prompt() -> None:
    model = BigramLanguageModel(VOCAB_SIZE)
    idx = torch.tensor([[3, 4]])
    out = generate(model, idx, max_new_tokens=7)
    assert out.shape == (1, 9)
    assert torch.equal(out[:, :2], idx)
    assert out.min() >= 0
    assert out.max() < VOCAB_SIZE


def test_generate_is_reproducible_with_a_generator() -> None:
    model = BigramLanguageModel(VOCAB_SIZE)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out1 = generate(model, idx, 20, generator=torch.Generator().manual_seed(0))
    out2 = generate(model, idx, 20, generator=torch.Generator().manual_seed(0))
    assert torch.equal(out1, out2)


def test_generate_follows_a_certain_model() -> None:
    # A model that is sure "after i comes i + 1" must generate 0, 1, 2, ...
    model = BigramLanguageModel(VOCAB_SIZE)
    with torch.no_grad():
        model.token_embedding.weight.fill_(-100.0)
        for i in range(VOCAB_SIZE):
            model.token_embedding.weight[i, (i + 1) % VOCAB_SIZE] = 100.0
    out = generate(model, torch.zeros((1, 1), dtype=torch.long), 12)
    assert out[0].tolist() == [i % VOCAB_SIZE for i in range(13)]


def make_peaked_model() -> BigramLanguageModel:
    """A model whose favourite next character is always i + 1, but not by much."""
    model = BigramLanguageModel(VOCAB_SIZE)
    with torch.no_grad():
        model.token_embedding.weight.zero_()
        for i in range(VOCAB_SIZE):
            model.token_embedding.weight[i, (i + 1) % VOCAB_SIZE] = 1.0
    return model


def test_top_k_of_one_always_takes_the_most_likely_character() -> None:
    out = generate(make_peaked_model(), torch.zeros((1, 1), dtype=torch.long), 15, top_k=1)
    assert out[0].tolist() == [i % VOCAB_SIZE for i in range(16)]


def test_low_temperature_is_almost_greedy() -> None:
    out = generate(
        make_peaked_model(),
        torch.zeros((1, 1), dtype=torch.long),
        15,
        temperature=0.01,
        generator=torch.Generator().manual_seed(0),
    )
    assert out[0].tolist() == [i % VOCAB_SIZE for i in range(16)]


def test_high_temperature_explores() -> None:
    out = generate(
        make_peaked_model(),
        torch.zeros((1, 1), dtype=torch.long),
        200,
        temperature=5.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert len(set(out[0].tolist())) == VOCAB_SIZE


def test_non_positive_temperature_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        generate(make_peaked_model(), torch.zeros((1, 1), dtype=torch.long), 1, temperature=0)

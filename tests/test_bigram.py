import torch
import torch.nn.functional as F  # noqa: N812

from mini_llm.bigram import BigramLanguageModel

VOCAB_SIZE = 10


def test_forward_shapes_without_targets() -> None:
    model = BigramLanguageModel(VOCAB_SIZE)
    idx = torch.randint(VOCAB_SIZE, (2, 5))
    logits, loss = model(idx)
    assert logits.shape == (2, 5, VOCAB_SIZE)
    assert loss is None


def test_forward_loss_is_the_cross_entropy() -> None:
    model = BigramLanguageModel(VOCAB_SIZE)
    idx = torch.randint(VOCAB_SIZE, (2, 5))
    targets = torch.randint(VOCAB_SIZE, (2, 5))
    logits, loss = model(idx, targets)
    assert loss.shape == ()  # a single number
    expected = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
    assert torch.allclose(loss, expected)


def test_model_learns_a_deterministic_sequence() -> None:
    # In 0, 1, 2, ..., 9, 0, 1, ... the next character is always known:
    # after a short training the loss must be close to 0.
    torch.manual_seed(0)
    model = BigramLanguageModel(VOCAB_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    data = torch.arange(101) % VOCAB_SIZE
    x, y = data[:-1].unsqueeze(0), data[1:].unsqueeze(0)
    for _ in range(200):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1

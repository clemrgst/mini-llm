"""Train the bigram model on the Molière corpus and print a generated sample.

Usage (from the project root, after scripts/prepare_data.py):
    uv run python scripts/train_bigram.py
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from mini_llm.batching import get_batch, split_train_val
from mini_llm.bigram import BigramLanguageModel
from mini_llm.generation import generate
from mini_llm.tokenizer import CharTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BATCH_SIZE = 32  # sequences per batch
BLOCK_SIZE = 8  # characters per sequence
MAX_STEPS = 5000  # number of optimisation steps
EVAL_INTERVAL = 500  # print the losses every EVAL_INTERVAL steps
EVAL_BATCHES = 100  # batches averaged to estimate a loss
LEARNING_RATE = 1e-2
SEED = 1337


@torch.no_grad()
def estimate_loss(model: BigramLanguageModel, data: torch.Tensor) -> float:
    """Average loss over EVAL_BATCHES random batches (less noisy than one batch)."""
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    torch.manual_seed(SEED)

    text = (DATA_DIR / "moliere.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer.load(DATA_DIR / "tokenizer.json")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = split_train_val(data)
    print(f"train: {len(train_data):,} characters, val: {len(val_data):,} characters")

    model = BigramLanguageModel(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # A model that knows nothing gives every character the same probability:
    # its loss is ln(vocab_size). Training must bring the loss well below that.
    print(f"loss of a uniform guess: {math.log(tokenizer.vocab_size):.3f}")

    for step in range(MAX_STEPS + 1):
        if step % EVAL_INTERVAL == 0:
            train_loss = estimate_loss(model, train_data)
            val_loss = estimate_loss(model, val_data)
            print(f"step {step:5d} | train loss {train_loss:.3f} | val loss {val_loss:.3f}")

        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
        _, loss = model(x, y)  # 1. forward: how wrong are the predictions?
        optimizer.zero_grad()  # 2. forget the gradients of the previous step
        loss.backward()  # 3. backward: how should each weight change?
        optimizer.step()  # 4. update the weights a little in that direction

    start = torch.tensor([tokenizer.encode("\n")], dtype=torch.long)
    sample = generate(model, start, max_new_tokens=500)
    print("\n--- sample ---")
    print(tokenizer.decode(sample[0].tolist()))


if __name__ == "__main__":
    main()

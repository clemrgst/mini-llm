"""Generic training loop, shared by every model.

train(config) reads the corpus, builds the model named in the config, trains it
and saves a checkpoint holding everything needed to generate text later: the
weights, the config and the vocabulary.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import nn

from mini_llm.batching import get_batch, split_train_val
from mini_llm.bigram import BigramLanguageModel
from mini_llm.config import Config
from mini_llm.generation import generate
from mini_llm.model import GPTLanguageModel
from mini_llm.tokenizer import CharTokenizer

CHECKPOINT_NAME = "model.pt"


def resolve_device(name: str) -> torch.device:
    """Turn "auto" into "cuda" when a GPU is available, "cpu" otherwise."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def build_model(config: Config, vocab_size: int) -> nn.Module:
    """Create the untrained model described by the config."""
    if config.model == "bigram":
        return BigramLanguageModel(vocab_size)
    if config.model == "gpt":
        return GPTLanguageModel(
            vocab_size,
            block_size=config.block_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
            dropout=config.dropout,
        )
    raise ValueError(f"unknown model type: {config.model!r}")


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: torch.Tensor,
    config: Config,
    generator: torch.Generator | None = None,
) -> float:
    """Average loss over config.eval_batches random batches (less noisy than one)."""
    was_training = model.training
    model.eval()  # turns dropout off
    device = next(model.parameters()).device
    total = 0.0
    for _ in range(config.eval_batches):
        x, y = get_batch(data, config.batch_size, config.block_size, generator)
        _, loss = model(x.to(device), y.to(device))
        total += loss.item()
    model.train(was_training)
    return total / config.eval_batches


def save_checkpoint(
    path: str | Path, model: nn.Module, config: Config, vocab: list[str], history: list[dict]
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "vocab": vocab,
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[nn.Module, CharTokenizer, Config]:
    """Rebuild the trained model and its tokenizer from a checkpoint file."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = Config.from_dict(checkpoint["config"])
    tokenizer = CharTokenizer(checkpoint["vocab"])
    model = build_model(config, tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer, config


def train(config: Config, log: bool = True) -> list[dict]:
    """Train the model described by config and save it. Returns the loss history."""
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    text = Path(config.data_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = split_train_val(data, config.val_fraction)

    model = build_model(config, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    n_params = sum(p.numel() for p in model.parameters())
    if log:
        print(f"model: {config.model}, {n_params:,} parameters, device: {device}")
        print(f"train: {len(train_data):,} chars, val: {len(val_data):,} chars")
        print(f"loss of a uniform guess: {math.log(tokenizer.vocab_size):.3f}")

    # The evaluation batches use their own random generator, so evaluating does
    # not change which training batches are drawn.
    eval_generator = torch.Generator().manual_seed(config.seed)
    history: list[dict] = []
    start = time.perf_counter()
    for step in range(config.max_steps + 1):
        if step % config.eval_interval == 0 or step == config.max_steps:
            losses = {
                "step": step,
                "train_loss": estimate_loss(model, train_data, config, eval_generator),
                "val_loss": estimate_loss(model, val_data, config, eval_generator),
            }
            history.append(losses)
            if log:
                elapsed = time.perf_counter() - start
                print(
                    f"step {step:6d} | train {losses['train_loss']:.3f} "
                    f"| val {losses['val_loss']:.3f} | {elapsed:6.1f}s"
                )
        if step == config.max_steps:
            break

        x, y = get_batch(train_data, config.batch_size, config.block_size)
        _, loss = model(x.to(device), y.to(device))  # 1. forward: how wrong are we?
        optimizer.zero_grad(set_to_none=True)  # 2. forget the previous gradients
        loss.backward()  # 3. backward: how should each weight change?
        optimizer.step()  # 4. move every weight a little in that direction

    path = Path(config.out_dir) / CHECKPOINT_NAME
    save_checkpoint(path, model, config, tokenizer.vocab, history)
    if log:
        print(f"saved {path}")
        start_idx = torch.tensor([tokenizer.encode(text[0])], dtype=torch.long, device=device)
        sample = generate(model, start_idx, max_new_tokens=300)
        print("\n--- sample ---")
        print(tokenizer.decode(sample[0].tolist()))
    return history

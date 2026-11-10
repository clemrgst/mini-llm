import torch

from mini_llm.config import Config
from mini_llm.training import CHECKPOINT_NAME, load_checkpoint, train

# A tiny, very regular corpus: a model must learn it fast.
TEXT = "abcdefgh\n" * 300


def make_config(tmp_path, **overrides) -> Config:
    data_path = tmp_path / "corpus.txt"
    data_path.write_text(TEXT, encoding="utf-8")
    values = {
        "model": "bigram",
        "data_path": str(data_path),
        "out_dir": str(tmp_path / "run"),
        "batch_size": 8,
        "block_size": 8,
        "max_steps": 150,
        "learning_rate": 0.1,
        "eval_interval": 50,
        "eval_batches": 5,
        "device": "cpu",
    }
    values.update(overrides)
    return Config.from_dict(values)


def test_training_reduces_the_loss(tmp_path) -> None:
    history = train(make_config(tmp_path), log=False)
    assert history[0]["step"] == 0
    assert history[-1]["step"] == 150
    assert history[-1]["val_loss"] < history[0]["val_loss"] / 4


def test_training_is_reproducible(tmp_path) -> None:
    # Same config and seed: exactly the same losses, run after run.
    first = train(make_config(tmp_path), log=False)
    second = train(make_config(tmp_path), log=False)
    assert first == second


def test_checkpoint_round_trip(tmp_path) -> None:
    config = make_config(tmp_path)
    train(config, log=False)
    model, tokenizer, loaded_config = load_checkpoint(tmp_path / "run" / CHECKPOINT_NAME)
    assert loaded_config == config
    assert tokenizer.vocab == sorted(set(TEXT))
    # The trained model knows that "b" follows "a".
    logits, _ = model(torch.tensor([tokenizer.encode("a")]))
    assert tokenizer.decode([int(logits[0, -1].argmax())]) == "b"

"""Training configuration: every setting of a run, in one place.

A run is described by a JSON file in configs/. Keeping the settings out of the
code means we can compare runs, reproduce them, and train a new model without
editing Python.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

MODEL_TYPES = ("bigram", "gpt")


@dataclass
class Config:
    # What to train
    model: str = "bigram"  # one of MODEL_TYPES
    data_path: str = "data/moliere.txt"
    out_dir: str = "checkpoints/bigram"

    # Data
    batch_size: int = 32  # sequences per batch
    block_size: int = 8  # characters per sequence (context length)
    val_fraction: float = 0.1

    # Optimisation
    max_steps: int = 5000
    learning_rate: float = 1e-2
    eval_interval: int = 500  # evaluate every eval_interval steps
    eval_batches: int = 100  # batches averaged to estimate a loss
    seed: int = 1337
    device: str = "auto"  # "auto", "cpu" or "cuda"

    # Transformer size (ignored by the bigram model)
    n_embd: int = 64  # size of the vector that represents each character
    n_head: int = 4  # attention heads per block
    n_layer: int = 4  # number of Transformer blocks
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.model not in MODEL_TYPES:
            raise ValueError(f"model must be one of {MODEL_TYPES}, got {self.model!r}")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    @classmethod
    def from_dict(cls, values: dict) -> Config:
        """Build a config, refusing unknown keys (a typo would otherwise be ignored)."""
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Read a config from a JSON file. Missing keys keep their default value."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)

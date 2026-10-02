"""Character-level tokenizer: turns text into a list of integers and back.

A language model only manipulates numbers. The tokenizer is the bridge:
each distinct character of the training corpus gets an integer id.
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        """Build the tokenizer from a vocabulary (the list of known characters).

        The id of a character is its position in `vocab`.
        """
        self.vocab = vocab
        self.stoi = {}
        self.itos = {}
        for index, char in enumerate(vocab):
            self.stoi[char] = index
            self.itos[index] = char

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Create a tokenizer whose vocabulary is the sorted unique characters of `text`."""
        vocab = sorted(set(text))
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        """Number of distinct characters the tokenizer knows."""
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of ids.

        Raises ValueError if `text` contains a character that is not in the vocabulary.
        """
        ids = []
        for char in text:
            if char not in self.stoi:
                raise ValueError(f"text is containing unrecognized symbols: {char!r}")
            ids.append(self.stoi[char])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert a list of ids back into a string."""
        buffer = []
        for index in ids:
            buffer.append(self.itos[index])
        text = "".join(buffer)
        return text

    def save(self, path: str | Path) -> None:
        """Save the vocabulary to a JSON file."""
        vocab_json = json.dumps(self.vocab, ensure_ascii=False)
        Path(path).write_text(vocab_json, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        """Load a tokenizer previously written by `save`."""
        vocab_json = Path(path).read_text(encoding="utf-8")
        vocab = json.loads(vocab_json)
        return cls(vocab)

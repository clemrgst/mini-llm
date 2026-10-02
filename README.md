# mini-llm

> 🚧 Work in progress — a small GPT-style language model, implemented from scratch in PyTorch.

## Goals

- Implement a decoder-only Transformer from first principles (attention, embeddings, training loop).
- Train it on a French text corpus, on CPU for development and on a cloud GPU for full runs.
- Keep the code tested, typed and reproducible, and document every design choice.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/clemrgst/mini-llm.git
cd mini-llm
uv sync          # creates .venv and installs dependencies
uv run pytest    # runs the test suite
```

## Project structure

```
mini-llm/
├── src/mini_llm/   # model and training code (the Python package)
├── tests/          # unit tests (pytest)
├── scripts/        # entry points: data preparation, training, text generation
├── configs/        # training configurations
└── data/           # datasets (not tracked by Git)
```

## License

MIT

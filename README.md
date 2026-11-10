# mini-llm

A small GPT-style language model, implemented from scratch in PyTorch and trained on
seven plays by Molière. It writes French theatre one character at a time.

```
HARPAGON.

LUCINE.

Ma femme! je voudrais bien que je vous assurerai, et m'aime buter la faire
une fille devraine que je vous retarde qu'il la pièce.

- Sganarelle -

Non, je l'ai comme vous trempliquer de moi.
```

*Sample from the 0.8 M-parameter model after 20 minutes of training on a laptop CPU.*

## Results

| Model | Parameters | Context | Training (CPU) | Validation loss |
|---|---|---|---|---|
| Uniform guess | 0 | 0 | none | 4.61 (= ln 100) |
| Bigram (`configs/bigram.json`) | 10 k | 1 character | 8 s | 2.36 |
| GPT (`configs/gpt-small.json`) | 834 k | 128 characters | 20 min | **1.09** |

The loss is the cross-entropy of the next character, in nats. The bigram only sees the
current character: it learns spacing, capitals and common letter pairs, but no word.
The Transformer sees the previous 128 characters: it learns the layout of a play
(character names, stage directions, line breaks), real French words and short phrases.

Bigram sample, for comparison:

```
al Masi, des pagielese,«Hatre  MAR l'irèqute"Man juniéj'as celépoustière padoucoune, (de.
Lese le à. s  antrendili  :
```

Validation loss of the GPT model during training:

| Step | 0 | 1 000 | 2 000 | 3 000 | 4 000 | 5 000 |
|---|---|---|---|---|---|---|
| Val loss | 4.62 | 1.47 | 1.23 | 1.14 | 1.10 | 1.09 |

## How it works

```
Project Gutenberg ──> data.py (cleaning) ──> data/moliere.txt (600 k characters)
                                                    │
                       tokenizer.py (character ↔ id, 100 characters)
                                                    │
                       batching.py (90 % train / 10 % validation, random windows)
                                                    │
     configs/*.json ──> training.py (AdamW, evaluation, checkpoint) <── bigram.py or model.py
                                                    │
                                     checkpoints/<run>/model.pt
                                                    │
                       generation.py (temperature, top-k) ──> text
```

The Transformer (`src/mini_llm/model.py`) is a decoder-only GPT:

- token and learned position embeddings;
- `n_layer` pre-norm blocks, each made of causal multi-head self-attention and a
  feed-forward layer (`C → 4C → C`, GELU), both wrapped in residual connections;
- a final LayerNorm and a linear layer back to one score per character;
- GPT-2 style initialisation (normal, std 0.02) and dropout.

Every part is written by hand with `torch.nn` building blocks (no `nn.Transformer`,
no `scaled_dot_product_attention`) and covered by tests, including causality (changing
a future character never changes past outputs) and a sanity check that the model can
memorise a single batch.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Everything runs on CPU; a CUDA GPU is used
automatically when available.

```bash
git clone https://github.com/clemrgst/mini-llm.git
cd mini-llm
uv sync                                          # install dependencies
uv run pytest                                    # run the 51 tests

uv run python scripts/prepare_data.py            # download and clean the corpus
uv run python scripts/train.py configs/bigram.json       # baseline, a few seconds
uv run python scripts/train.py configs/gpt-small.json    # Transformer, about 20 min on CPU
uv run python scripts/generate.py checkpoints/gpt-small/model.pt --prompt "HARPAGON." --length 800
```

Useful options:

- `scripts/train.py --max-steps 200` runs a short training to check that everything works;
- `scripts/generate.py --temperature 0.7` makes the text safer, `1.2` more surprising;
  `--top-k 10` removes unlikely characters; `--seed 42` makes the output reproducible.

## Project structure

```
mini-llm/
├── src/mini_llm/
│   ├── data.py         # Gutenberg cleaning: licence, footnote markers, whitespace
│   ├── tokenizer.py    # character-level tokenizer
│   ├── batching.py     # train/validation split and random batches
│   ├── bigram.py       # baseline model
│   ├── model.py        # attention, Transformer block, GPT model
│   ├── config.py       # run settings, loaded from JSON
│   ├── training.py     # training loop, evaluation, checkpoints
│   └── generation.py   # sampling with temperature and top-k
├── scripts/            # prepare_data.py, train.py, generate.py
├── configs/            # bigram.json, gpt-small.json
├── tests/              # pytest suite
└── data/               # corpus (not tracked by Git)
```

## Limitations and next steps

- The corpus still contains some editorial text: footnote bodies (only the `[n]`
  markers are removed) and the transcriber's header of each book ("Produced by...").
- Character names use two formats depending on the edition (`SGANARELLE.` and
  `- Harpagon -`); the model mixes them.
- The model is small and the corpus is 600 k characters: the text looks like Molière
  but has no meaning beyond a few words. A larger model would overfit this corpus; more
  text (the complete works) is the first lever.
- Next experiments: a second corpus (cooking recipes) with the same pipeline, and a
  larger run on a GPU.

## License

MIT

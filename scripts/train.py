"""Train a model described by a JSON config file.

Usage (from the project root, after scripts/prepare_data.py):
    uv run python scripts/train.py configs/bigram.json
"""

from __future__ import annotations

import argparse

from mini_llm.config import Config
from mini_llm.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="path to a JSON config file, e.g. configs/bigram.json")
    parser.add_argument("--max-steps", type=int, help="override max_steps (quick tests)")
    parser.add_argument("--device", help='override the device: "auto", "cpu" or "cuda"')
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.device is not None:
        config.device = args.device
    train(config)


if __name__ == "__main__":
    main()

"""Generate text with a trained model.

Usage (from the project root, after training):
    uv run python scripts/generate.py checkpoints/gpt-small/model.pt
    uv run python scripts/generate.py checkpoints/gpt-small/model.pt --prompt "HARPAGON."
"""

from __future__ import annotations

import argparse

import torch

from mini_llm.generation import generate
from mini_llm.training import load_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="path to a model.pt written by scripts/train.py")
    parser.add_argument("--prompt", default="\n", help="beginning of the text")
    parser.add_argument("--length", type=int, default=500, help="characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="fix it to get the same text")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device=str(device))
    try:
        prompt_ids = tokenizer.encode(args.prompt)
    except ValueError as error:
        parser.error(f"the prompt uses a character the model never saw: {error}")
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    out = generate(
        model,
        idx,
        args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        generator=generator,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()

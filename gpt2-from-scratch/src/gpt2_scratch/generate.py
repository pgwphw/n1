from __future__ import annotations

import argparse

import torch

from .config import GPTConfig
from .data import CharTokenizer
from .model import GPT
from .utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_shakespeare.pt")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt = args.prompt
    if not prompt:
        prompt = "\n" if "\n" in tokenizer.stoi else tokenizer.chars[0]

    encoded = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(
        encoded,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()

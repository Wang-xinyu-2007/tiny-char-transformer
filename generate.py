"""Load a trained checkpoint and generate text from a prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from charformer.data import CharTokenizer
from charformer.model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="results/best_model.pt")
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="recompute the prefix every step instead of using a KV cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found; run `python train.py` first"
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = CharTokenizer(tuple(checkpoint["chars"]))
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("prompt must contain at least one known character")
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(
        idx,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_cache=not args.no_cache,
    )[0].tolist()
    print(tokenizer.decode(output))


if __name__ == "__main__":
    main()

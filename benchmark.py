"""Compare cached and uncached decoding on the trained checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from charformer.data import CharTokenizer
from charformer.model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--new_tokens", type=int, default=96)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/cache_benchmark.json")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GPT, CharTokenizer]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found; run `python train.py` first"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = CharTokenizer(tuple(checkpoint["chars"]))
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def timed_generate(
    model: GPT,
    prompt: torch.Tensor,
    new_tokens: int,
    use_cache: bool,
    repeat: int,
) -> float:
    # Warm up so lazy CPU kernels do not pollute the first measurement.
    model.generate(prompt, new_tokens, temperature=0, use_cache=use_cache)
    times = []
    for _ in range(repeat):
        started = time.perf_counter()
        model.generate(prompt, new_tokens, temperature=0, use_cache=use_cache)
        times.append(time.perf_counter() - started)
    return min(times)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, tokenizer = load_model(Path(args.checkpoint), device)
    prompt_ids = tokenizer.encode(args.prompt)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    if prompt.size(1) + args.new_tokens > model.config.block_size:
        raise ValueError(
            "prompt + new_tokens must fit in block_size for the cache benchmark; "
            f"got {prompt.size(1)} + {args.new_tokens} > {model.config.block_size}"
        )

    with torch.no_grad():
        uncached_output = model.generate(
            prompt, args.new_tokens, temperature=0, use_cache=False
        )
        cached_output = model.generate(
            prompt, args.new_tokens, temperature=0, use_cache=True
        )
    if not torch.equal(uncached_output, cached_output):
        raise RuntimeError("cached and uncached greedy outputs differ")

    uncached_sec = timed_generate(
        model, prompt, args.new_tokens, use_cache=False, repeat=args.repeat
    )
    cached_sec = timed_generate(
        model, prompt, args.new_tokens, use_cache=True, repeat=args.repeat
    )
    speedup = uncached_sec / cached_sec if cached_sec else float("inf")
    result = {
        "device": args.device,
        "block_size": model.config.block_size,
        "new_tokens": args.new_tokens,
        "uncached_seconds": round(uncached_sec, 3),
        "cached_seconds": round(cached_sec, 3),
        "speedup": round(speedup, 3),
        "uncached_tokens_per_sec": round(args.new_tokens / uncached_sec, 1),
        "cached_tokens_per_sec": round(args.new_tokens / cached_sec, 1),
        "greedy_outputs_match": True,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

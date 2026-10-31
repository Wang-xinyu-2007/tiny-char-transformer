"""Train a small character-level GPT on Tiny Shakespeare.

Example (CPU, about a few minutes):

    python train.py --max_steps 1500

The script writes:
    results/metrics.json     training history and final metrics
    results/loss_curve.png   train/validation loss curve
    results/samples.txt      a generated sample from the best checkpoint
    results/best_model.pt    checkpoint (git-ignored)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from charformer.data import DATASET_SHA256, ensure_dataset, get_batch, load_tokenized_dataset
from charformer.model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=str, default="data/tiny_shakespeare.txt")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=40)
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--sample_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_at(
    step: int,
    max_steps: int,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    """Linear warmup followed by cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (peak_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    model.eval()
    losses = torch.empty(eval_iters)
    for index in range(eval_iters):
        inputs, targets = get_batch(
            data, batch_size, block_size, device, generator=generator
        )
        _, loss = model(inputs, targets)
        losses[index] = loss.detach()
    model.train()
    return losses.mean()


def save_checkpoint(
    path: Path,
    model: GPT,
    config: GPTConfig,
    tokenizer_chars: tuple[str, ...],
    step: int,
    train_loss: float,
    val_loss: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "chars": list(tokenizer_chars),
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def plot_loss_curve(history: list[dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(7, 4.2))
    plt.plot(steps, train_loss, marker="o", markersize=3, label="train")
    plt.plot(steps, val_loss, marker="o", markersize=3, label="validation")
    plt.xlabel("optimizer step")
    plt.ylabel("cross-entropy loss")
    plt.title("Tiny Char Transformer: training on Tiny Shakespeare")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_sample(
    model: GPT,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    path: Path,
) -> str:
    ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=device
    )
    generated = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )[0].tolist()
    text = tokenizer.decode(generated)
    path.write_text(
        f"prompt: {prompt!r}\n"
        f"temperature: {temperature}, top_k: {top_k}, tokens: {max_new_tokens}\n\n"
        f"{text}\n",
        encoding="utf-8",
    )
    return text


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = ensure_dataset(args.data_path)
    tokenizer, train_data, val_data = load_tokenized_dataset(data_path)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    model.train()

    decay_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.dim() >= 2
    ]
    nodecay_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.dim() < 2
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": args.weight_decay},
            {"params": nodecay_parameters, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    print(f"device={device}  parameters={model.num_parameters:,}")
    print(
        f"vocab={tokenizer.vocab_size}  "
        f"train_tokens={train_data.numel():,}  val_tokens={val_data.numel():,}"
    )

    generator = torch.Generator().manual_seed(args.seed)
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_step = 0
    started_at = time.perf_counter()

    for step in range(args.max_steps):
        lr = learning_rate_at(
            step, args.max_steps, args.lr, args.min_lr, args.warmup_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(
            train_data,
            args.batch_size,
            args.block_size,
            device,
            generator=generator,
        )
        _, loss = model(inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            train_loss = estimate_loss(
                model,
                train_data,
                args.batch_size,
                args.block_size,
                args.eval_iters,
                device,
                generator,
            ).item()
            val_loss = estimate_loss(
                model,
                val_data,
                args.batch_size,
                args.block_size,
                args.eval_iters,
                device,
                generator,
            ).item()
            history.append(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                }
            )
            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                save_checkpoint(
                    out_dir / "best_model.pt",
                    model,
                    config,
                    tokenizer.chars,
                    step,
                    train_loss,
                    val_loss,
                )
                marker = "  saved best"
            print(
                f"step {step:4d} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | lr {lr:.2e}{marker}"
            )

        if step % args.log_interval == 0:
            print(f"  batch loss {loss.item():.4f}")

    elapsed = time.perf_counter() - started_at
    history_path = out_dir / "metrics.json"
    history_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "dataset": {
                    "path": str(data_path),
                    "sha256": DATASET_SHA256,
                    "vocab_size": tokenizer.vocab_size,
                    "train_tokens": train_data.numel(),
                    "val_tokens": val_data.numel(),
                },
                "training": {
                    "device": str(device),
                    "max_steps": args.max_steps,
                    "batch_size": args.batch_size,
                    "eval_iters": args.eval_iters,
                    "parameter_count": model.num_parameters,
                    "wall_time_sec": round(elapsed, 1),
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                    "best_val_perplexity": math.exp(min(best_val_loss, 20.0)),
                },
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_loss_curve(history, out_dir / "loss_curve.png")

    # Re-load the best checkpoint so the committed sample is not from an
    # overfit final step.
    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    prompt = "ROMEO:"
    sample = write_sample(
        model,
        tokenizer,
        device,
        prompt,
        args.sample_tokens,
        args.temperature,
        args.top_k,
        out_dir / "samples.txt",
    )

    print(f"\nbest val loss {best_val_loss:.4f} at step {best_step}")
    print(f"wrote {history_path}, {out_dir / 'loss_curve.png'}, {out_dir / 'samples.txt'}")
    print("\n--- sample ---")
    print(sample[:600])


if __name__ == "__main__":
    main()

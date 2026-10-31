"""Character tokenizer, dataset loading, and mini-batch sampling."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATASET_SHA256 = (
    "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
)


def ensure_dataset(
    path: str | Path,
    url: str = DATASET_URL,
    expected_sha256: str = DATASET_SHA256,
) -> Path:
    """Download Tiny Shakespeare once and verify its SHA-256.

    The corpus is a public-domain collection of Shakespeare plays used widely
    for character-level language modelling. It is not committed to this repo
    so that the repository stays small; the download is deterministic.
    """
    destination = Path(path)
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "charformer"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"dataset checksum mismatch: expected {expected_sha256}, got {digest}"
        )

    destination.write_bytes(payload)
    return destination


@dataclass(frozen=True)
class CharTokenizer:
    """Map each character to a contiguous integer id and back."""

    chars: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(tuple(sorted(set(text))))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @property
    def stoi(self) -> dict[str, int]:
        return {char: index for index, char in enumerate(self.chars)}

    @property
    def itos(self) -> dict[int, str]:
        return {index: char for index, char in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        mapping = self.stoi
        unknown = sorted(set(text) - set(mapping))
        if unknown:
            raise ValueError(f"characters not in vocabulary: {unknown!r}")
        return [mapping[char] for char in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
        mapping = self.itos
        try:
            return "".join(mapping[int(index)] for index in ids)
        except KeyError as error:
            raise ValueError(f"token id not in vocabulary: {error.args[0]}") from None


def load_tokenized_dataset(
    data_path: str | Path,
    val_fraction: float = 0.1,
) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    """Read the corpus, build the character vocabulary, and split train/val."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    text = Path(data_path).read_text(encoding="utf-8")
    if len(text) < 100_000:
        raise ValueError("dataset is unexpectedly small")

    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = int((1.0 - val_fraction) * len(encoded))
    train_data = encoded[:split_index]
    val_data = encoded[split_index:]
    return tokenizer, train_data, val_data


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random contiguous windows and their next-token targets."""
    if data.numel() <= block_size:
        raise ValueError(
            f"dataset split has {data.numel()} tokens, need more than {block_size}"
        )

    starts = torch.randint(
        0,
        data.numel() - block_size,
        (batch_size,),
        generator=generator,
    )
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack(
        [data[start + 1 : start + block_size + 1] for start in starts]
    )
    return inputs.to(device), targets.to(device)

"""Tests for dataset loading and split sizes."""

from __future__ import annotations

from pathlib import Path

import pytest

from charformer.data import load_tokenized_dataset


def test_load_tokenized_dataset_splits_train_and_validation(tmp_path: Path) -> None:
    alphabet = "abcdefghij"
    text = alphabet * 12_000  # 120,000 characters, above the safety check
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")

    tokenizer, train_data, val_data = load_tokenized_dataset(
        path, val_fraction=0.1
    )
    assert tokenizer.vocab_size == len(alphabet)
    assert train_data.numel() + val_data.numel() == len(text)
    assert train_data.numel() == 108_000
    assert val_data.numel() == 12_000
    assert tokenizer.decode(train_data[:10]) == alphabet


def test_load_rejects_short_corpus(tmp_path: Path) -> None:
    path = tmp_path / "short.txt"
    path.write_text("abc" * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="unexpectedly small"):
        load_tokenized_dataset(path)

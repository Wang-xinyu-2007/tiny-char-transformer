"""Tests for the model, attention mask, sampling, and tokenizer."""

from __future__ import annotations

import pytest
import torch

from charformer.data import CharTokenizer, get_batch
from charformer.model import GPT, GPTConfig


@pytest.fixture()
def model() -> GPT:
    config = GPTConfig(
        vocab_size=17,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    return GPT(config).eval()


def test_forward_shapes_and_loss(model: GPT) -> None:
    inputs = torch.randint(0, 17, (2, 8))
    targets = torch.randint(0, 17, (2, 8))
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 8, 17)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()
    assert model.transformer["wte"].weight.grad is not None


def test_causal_attention_cannot_see_future_tokens(model: GPT) -> None:
    torch.manual_seed(0)
    prefix = torch.randint(0, 17, (1, 5))
    future_a = torch.randint(0, 17, (1, 3))
    future_b = torch.randint(0, 17, (1, 3))
    sequence_a = torch.cat((prefix, future_a), dim=1)
    sequence_b = torch.cat((prefix, future_b), dim=1)

    logits_a, _ = model(sequence_a)
    logits_b, _ = model(sequence_b)
    assert torch.allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-6)


def test_generate_returns_requested_number_of_tokens(model: GPT) -> None:
    start = torch.zeros((1, 1), dtype=torch.long)
    output = model.generate(start, max_new_tokens=6, temperature=0.8, top_k=10)
    assert output.shape == (1, 7)
    assert output[0, 0].item() == 0


def test_sequence_longer_than_block_size_is_rejected(model: GPT) -> None:
    with pytest.raises(ValueError, match="exceeds block size"):
        model(torch.zeros((1, 9), dtype=torch.long))


def test_weights_are_tied(model: GPT) -> None:
    assert model.lm_head.weight is model.transformer["wte"].weight


def test_tokenizer_roundtrip() -> None:
    tokenizer = CharTokenizer.from_text("cabba")
    ids = tokenizer.encode("cab")
    assert ids == [2, 0, 1]
    assert tokenizer.decode(ids) == "cab"


def test_tokenizer_rejects_unknown_characters() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="not in vocabulary"):
        tokenizer.encode("abz")


def test_get_batch_targets_are_shifted_by_one() -> None:
    data = torch.arange(20)
    inputs, targets = get_batch(data, batch_size=3, block_size=4, device="cpu")
    assert inputs.shape == (3, 4)
    assert targets.shape == (3, 4)
    assert torch.equal(inputs + 1, targets)


def test_get_batch_rejects_short_split() -> None:
    with pytest.raises(ValueError, match="need more than"):
        get_batch(torch.arange(4), batch_size=2, block_size=4, device="cpu")

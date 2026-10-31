"""Tests for incremental decoding with a KV cache."""

from __future__ import annotations

import pytest
import torch

from charformer.model import GPT, GPTConfig


@pytest.fixture()
def model() -> GPT:
    config = GPTConfig(
        vocab_size=17,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    return GPT(config).eval()


def test_prefill_cache_matches_full_forward(model: GPT) -> None:
    prompt = torch.randint(0, 17, (1, 5))
    with torch.no_grad():
        full_logits, _ = model(prompt)
        cached_logits, caches = model.forward_with_cache(prompt)
    assert torch.allclose(full_logits, cached_logits, atol=1e-6)
    assert len(caches) == 2
    assert caches[0][0].shape == (1, 2, 5, 16)


def test_incremental_token_matches_full_sequence(model: GPT) -> None:
    prompt = torch.randint(0, 17, (1, 4))
    next_token = torch.tensor([[3]])
    with torch.no_grad():
        full_logits, _ = model(torch.cat((prompt, next_token), dim=1))
        _, caches = model.forward_with_cache(prompt)
        incremental_logits, new_caches = model.forward_with_cache(
            next_token,
            past_kv=caches,
            position_offset=4,
        )
    assert torch.allclose(incremental_logits[:, -1], full_logits[:, -1], atol=1e-6)
    assert new_caches[0][0].shape == (1, 2, 5, 16)


def test_cached_greedy_generation_matches_uncached(model: GPT) -> None:
    torch.manual_seed(0)
    prompt = torch.randint(0, 17, (1, 3))
    with torch.no_grad():
        cached = model.generate(prompt, max_new_tokens=5, temperature=0, use_cache=True)
        uncached = model.generate(prompt, max_new_tokens=5, temperature=0, use_cache=False)
    assert torch.equal(cached, uncached)


def test_long_generation_falls_back_and_still_matches(model: GPT) -> None:
    torch.manual_seed(1)
    prompt = torch.randint(0, 17, (1, 1))
    with torch.no_grad():
        cached = model.generate(prompt, max_new_tokens=20, temperature=0, use_cache=True)
        uncached = model.generate(prompt, max_new_tokens=20, temperature=0, use_cache=False)
    assert torch.equal(cached, uncached)


def test_cache_rejects_mismatched_position_offset(model: GPT) -> None:
    with pytest.raises(ValueError, match="does not match"):
        model.forward_with_cache(torch.zeros((1, 1), dtype=torch.long), position_offset=2)


def test_generate_rejects_zero_top_k(model: GPT) -> None:
    with pytest.raises(ValueError, match="top_k"):
        model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=2, top_k=0)

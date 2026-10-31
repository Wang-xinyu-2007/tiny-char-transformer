"""A small, GPT-style decoder-only Transformer.

The point of this module is to keep every moving part visible: causal
self-attention, the residual MLP block, pre-LayerNorm, weight tying, and
sampling are all implemented explicitly instead of hidden behind a
high-level `TransformerEncoder` wrapper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    """Hyper-parameters for the character-level model."""

    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a lower-triangular causal mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # One fused projection produces query, key, and value.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # `persistent=False` keeps this out of the checkpoint: it is derived
        # from the block size and can be rebuilt on load.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        if seq_len > self.causal_mask.shape[-1]:
            raise ValueError(
                f"sequence length {seq_len} exceeds block size "
                f"{self.causal_mask.shape[-1]}"
            )

        # (B, T, 3C) -> three (B, n_head, T, head_dim) tensors.
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        query = query.view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        # Scaled dot-product attention: QK^T / sqrt(d_k).
        attention = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = attention.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf"),
        )
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)

        output = attention @ value  # (B, n_head, T, head_dim)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, channels
        )
        return self.resid_dropout(self.c_proj(output))


class MLP(nn.Module):
    """Position-wise feed-forward network with a 4x expansion."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Pre-LayerNorm Transformer block: x + Attn(LN(x)); x + MLP(LN(x))."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "blocks": nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_weights:
            # Input and output embeddings share one matrix, saving vocab_size*d
            # parameters and tying token representations together.
            self.lm_head.weight = self.transformer.wte.weight

        # Scale the output projection of each residual branch. This keeps the
        # variance of activations from growing with depth at initialization.
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and, when targets are supplied, mean cross-entropy."""
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block size "
                f"{self.config.block_size}"
            )

        positions = torch.arange(seq_len, device=idx.device, dtype=torch.long)
        token_embedding = self.transformer["wte"](idx)
        position_embedding = self.transformer["wpe"](positions)
        x = self.transformer["drop"](token_embedding + position_embedding)

        for block in self.transformer["blocks"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample tokens, cropping context to block_size."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # logits for the next token

            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    top_values = torch.topk(logits, k).values
                    threshold = top_values[:, [-1]]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)
        return idx

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters, counting tied weights once."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

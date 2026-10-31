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
from torch import nn
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


KVCache = tuple[torch.Tensor, torch.Tensor]


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

    def _project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
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
        return query, key, value

    def _attend(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Attend to the current tokens plus an optional cached prefix."""
        batch_size, seq_len, channels = x.shape
        query, key, value = self._project_qkv(x)

        past_len = 0 if past_kv is None else past_kv[0].size(2)
        total_len = past_len + seq_len
        if total_len > self.causal_mask.shape[-1]:
            raise ValueError(
                f"sequence length {total_len} exceeds block size "
                f"{self.causal_mask.shape[-1]}"
            )
        if past_kv is not None:
            key = torch.cat((past_kv[0], key), dim=2)
            value = torch.cat((past_kv[1], value), dim=2)

        # Query positions start after the cached prefix. A query at position t
        # may attend to keys 0..t, including the token just appended.
        query_positions = torch.arange(
            past_len, total_len, device=x.device
        ).unsqueeze(1)
        key_positions = torch.arange(total_len, device=x.device).unsqueeze(0)
        allowed = key_positions <= query_positions

        attention = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = attention.masked_fill(~allowed, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)

        output = attention @ value
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, channels
        )
        return self.resid_dropout(self.c_proj(output)), (key, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self._attend(x)
        return output

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Return the attention output and the extended key/value cache."""
        return self._attend(x, past_kv)


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

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        attention, new_kv = self.attn.forward_with_cache(self.ln_1(x), past_kv)
        x = x + attention
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


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
    def forward_with_cache(
        self,
        idx: torch.Tensor,
        past_kv: list[KVCache] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Forward one or more tokens while reusing cached keys and values.

        `position_offset` must equal the number of cached positions. For a
        prefill call pass `past_kv=None, position_offset=0`; for each new token
        pass the cache returned by the previous call and
        `position_offset=idx.size(1)-1`.
        """
        if idx.size(1) == 0:
            raise ValueError("cannot forward an empty sequence")
        past_len = 0 if past_kv is None else past_kv[0][0].size(2)
        if position_offset != past_len:
            raise ValueError(
                f"position_offset {position_offset} does not match cached length {past_len}"
            )
        total_len = position_offset + idx.size(1)
        if total_len > self.config.block_size:
            raise ValueError(
                f"sequence length {total_len} exceeds block size {self.config.block_size}"
            )

        positions = torch.arange(
            position_offset, total_len, device=idx.device, dtype=torch.long
        )
        token_embedding = self.transformer["wte"](idx)
        position_embedding = self.transformer["wpe"](positions)
        x = self.transformer["drop"](token_embedding + position_embedding)

        caches: list[KVCache | None] = [None] * len(self.transformer["blocks"]) if past_kv is None else past_kv
        if len(caches) != len(self.transformer["blocks"]):
            raise ValueError("cache does not match the number of layers")
        new_caches: list[KVCache] = []
        for block, cache in zip(self.transformer["blocks"], caches, strict=True):
            x, new_cache = block.forward_with_cache(x, cache)
            new_caches.append(new_cache)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        return logits, new_caches

    @staticmethod
    def _sample_next(
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:
        logits = logits[:, -1, :]
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            top_values = torch.topk(logits, k).values
            threshold = top_values[:, [-1]]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probabilities = F.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively sample tokens.

        The KV cache avoids recomputing the prefix and is used when the whole
        sequence fits in the context window. For longer generations the method
        falls back to recomputing the sliding window, because absolute learned
        position embeddings cannot be shifted without re-encoding.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if idx.size(1) == 0:
            raise ValueError("prompt must contain at least one token")
        if max_new_tokens == 0:
            return idx

        self.eval()
        idx = idx[:, -self.config.block_size :]
        if use_cache and idx.size(1) + max_new_tokens <= self.config.block_size:
            logits, caches = self.forward_with_cache(idx)
            for step in range(max_new_tokens):
                next_token = self._sample_next(logits, temperature, top_k)
                idx = torch.cat((idx, next_token), dim=1)
                if step == max_new_tokens - 1:
                    break
                logits, caches = self.forward_with_cache(
                    next_token,
                    past_kv=caches,
                    position_offset=idx.size(1) - 1,
                )
            return idx

        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size :])
            next_token = self._sample_next(logits, temperature, top_k)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters, counting tied weights once."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

import torch
from torch import nn

from .norms import RMSNorm
from .triton_attention import AttentionBackend, TritonAttention, triton_attention
from .triton_ffn import FeedForwardBackend, triton_ffn_activation


NormType = Literal["layernorm", "rmsnorm"]
FeedForwardType = Literal["gelu", "silu", "swiglu"]
AttentionType = Literal["standard", "causal"]


@lru_cache(maxsize=16)
def build_causal_attention_mask(query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
    row_index = torch.arange(query_length, device=device).unsqueeze(1)
    column_index = torch.arange(key_length, device=device).unsqueeze(0)
    offset = key_length - query_length
    return column_index > (row_index + offset)


def _build_norm(hidden_dim: int, norm_type: NormType) -> nn.Module:
    normalized = str(norm_type).strip().lower()
    if normalized == "layernorm":
        return nn.LayerNorm(hidden_dim)
    if normalized == "rmsnorm":
        return RMSNorm(hidden_dim)
    raise ValueError(f"Unsupported norm_type '{norm_type}'")


def _apply_token_mask(hidden_states: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    if token_mask is None:
        return hidden_states
    return hidden_states * token_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)


def apply_token_specific_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    projected = torch.einsum("bnd,ndo->bno", hidden_states, weight)
    if bias is not None:
        projected = projected + bias.unsqueeze(0)
    return projected


class TokenSpecificLinear(nn.Module):
    def __init__(self, token_count: int, input_dim: int, output_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(token_count, input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(token_count, output_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return apply_token_specific_linear(hidden_states, self.weight, self.bias)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        *,
        ffn_type: FeedForwardType = "swiglu",
        dropout: float = 0.0,
        backend: FeedForwardBackend = "auto",
    ) -> None:
        super().__init__()
        normalized = str(ffn_type).strip().lower()
        self.ffn_type = normalized
        self.backend = backend
        self.dropout = nn.Dropout(dropout)
        if normalized == "swiglu":
            self.up_projection = nn.Linear(hidden_dim, ffn_dim * 2)
            self.down_projection = nn.Linear(ffn_dim, hidden_dim)
        elif normalized in {"gelu", "silu"}:
            self.up_projection = nn.Linear(hidden_dim, ffn_dim)
            self.down_projection = nn.Linear(ffn_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported ffn_type '{ffn_type}'")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.up_projection(hidden_states)
        projected = triton_ffn_activation(projected, self.ffn_type, backend=self.backend)
        projected = self.dropout(projected)
        return self.down_projection(projected)


class TaacTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        ffn_dim: int | None = None,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_type: NormType = "rmsnorm",
        ffn_type: FeedForwardType = "swiglu",
        attention_type: AttentionType = "standard",
        attention_backend: AttentionBackend = "auto",
        ffn_backend: FeedForwardBackend = "auto",
    ) -> None:
        super().__init__()
        resolved_attention_type = str(attention_type).strip().lower()
        if resolved_attention_type not in {"standard", "causal"}:
            raise ValueError(f"Unsupported attention_type '{attention_type}'")

        self.attention_type = resolved_attention_type
        self.attention_norm = _build_norm(hidden_dim, norm_type)
        self.self_attention = TritonAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            backend=attention_backend,
        )
        resolved_ffn_dim = ffn_dim if ffn_dim is not None else int(hidden_dim * ffn_multiplier)
        self.ffn_norm = _build_norm(hidden_dim, norm_type)
        self.feed_forward = FeedForwardNetwork(
            hidden_dim=hidden_dim,
            ffn_dim=resolved_ffn_dim,
            ffn_type=ffn_type,
            dropout=dropout,
            backend=ffn_backend,
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        normalized_states = self.attention_norm(hidden_states)
        attention_output = self.self_attention(
            normalized_states,
            normalized_states,
            normalized_states,
            key_mask=token_mask,
            query_mask=token_mask,
            is_causal=self.attention_type == "causal",
        )
        hidden_states = hidden_states + self.residual_dropout(attention_output)
        hidden_states = _apply_token_mask(hidden_states, token_mask)

        ffn_output = self.feed_forward(self.ffn_norm(hidden_states))
        hidden_states = hidden_states + self.residual_dropout(ffn_output)
        return _apply_token_mask(hidden_states, token_mask)


class TaacCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        ffn_dim: int | None = None,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_type: NormType = "layernorm",
        ffn_type: FeedForwardType = "gelu",
        attention_backend: AttentionBackend = "auto",
        ffn_backend: FeedForwardBackend = "auto",
    ) -> None:
        super().__init__()
        self.query_norm = _build_norm(hidden_dim, norm_type)
        self.context_norm = _build_norm(hidden_dim, norm_type)
        self.cross_attention = TritonAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            backend=attention_backend,
        )
        resolved_ffn_dim = ffn_dim if ffn_dim is not None else int(hidden_dim * ffn_multiplier)
        self.ffn_norm = _build_norm(hidden_dim, norm_type)
        self.feed_forward = FeedForwardNetwork(
            hidden_dim=hidden_dim,
            ffn_dim=resolved_ffn_dim,
            ffn_type=ffn_type,
            dropout=dropout,
            backend=ffn_backend,
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_states: torch.Tensor,
        context_states: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_queries = self.query_norm(query_states)
        normalized_context = self.context_norm(context_states)
        attention_output = self.cross_attention(
            normalized_queries,
            normalized_context,
            normalized_context,
            key_mask=context_mask,
            query_mask=query_mask,
        )
        query_states = query_states + self.residual_dropout(attention_output)
        query_states = _apply_token_mask(query_states, query_mask)

        ffn_output = self.feed_forward(self.ffn_norm(query_states))
        query_states = query_states + self.residual_dropout(ffn_output)
        return _apply_token_mask(query_states, query_mask)


class TaacMixedCausalAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ns_token_count: int,
        dropout: float,
        *,
        attention_backend: AttentionBackend = "auto",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.ns_token_count = ns_token_count
        self.attention_backend = attention_backend
        self.seq_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.seq_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.seq_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ns_query = TokenSpecificLinear(ns_token_count, hidden_dim, hidden_dim, bias=False)
        self.ns_key = TokenSpecificLinear(ns_token_count, hidden_dim, hidden_dim, bias=False)
        self.ns_value = TokenSpecificLinear(ns_token_count, hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = hidden_states.shape
        return hidden_states.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _merge_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, token_count, _ = hidden_states.shape
        return hidden_states.transpose(1, 2).contiguous().view(batch_size, token_count, self.hidden_dim)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        sequence_mask: torch.Tensor,
        ns_tokens: torch.Tensor,
        ns_mask: torch.Tensor,
        next_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_queries = sequence_tokens[:, -next_sequence_length:]
        sequence_query_mask = sequence_mask[:, -next_sequence_length:]
        query_mask = torch.cat([sequence_query_mask, ns_mask], dim=1)
        key_mask = torch.cat([sequence_mask, ns_mask], dim=1)

        query_states = torch.cat([self.seq_query(sequence_queries), self.ns_query(ns_tokens)], dim=1)
        key_states = torch.cat([self.seq_key(sequence_tokens), self.ns_key(ns_tokens)], dim=1)
        value_states = torch.cat([self.seq_value(sequence_tokens), self.ns_value(ns_tokens)], dim=1)

        total_sequence_length = sequence_tokens.shape[1]
        total_key_length = total_sequence_length + self.ns_token_count
        query_positions = torch.cat(
            [
                torch.arange(total_sequence_length - next_sequence_length, total_sequence_length, device=sequence_tokens.device),
                torch.arange(total_sequence_length, total_key_length, device=sequence_tokens.device),
            ]
        )
        key_positions = torch.arange(total_key_length, device=sequence_tokens.device)
        attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(0).expand(sequence_tokens.shape[0], -1, -1)

        attended = triton_attention(
            self._reshape_heads(query_states),
            self._reshape_heads(key_states),
            self._reshape_heads(value_states),
            attention_mask=attention_mask,
            query_mask=query_mask,
            key_mask=key_mask,
            backend=self.attention_backend,
        )
        attended = self.output_projection(self.dropout(self._merge_heads(attended)))
        sequence_output = attended[:, :next_sequence_length] * sequence_query_mask.unsqueeze(-1).float()
        ns_output = attended[:, next_sequence_length:] * ns_mask.unsqueeze(-1).float()
        return sequence_output, sequence_query_mask, ns_output


class TaacMixedCausalFeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        ns_token_count: int,
        dropout: float,
        *,
        backend: FeedForwardBackend = "auto",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.seq_up = nn.Linear(hidden_dim, ffn_dim)
        self.seq_down = nn.Linear(ffn_dim, hidden_dim)
        self.ns_up = TokenSpecificLinear(ns_token_count, hidden_dim, ffn_dim)
        self.ns_down = TokenSpecificLinear(ns_token_count, ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        sequence_mask: torch.Tensor,
        ns_tokens: torch.Tensor,
        ns_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_hidden = triton_ffn_activation(self.seq_up(sequence_tokens), "silu", backend=self.backend)
        sequence_hidden = self.seq_down(self.dropout(sequence_hidden))
        sequence_hidden = self.dropout(sequence_hidden) * sequence_mask.unsqueeze(-1).float()

        ns_hidden = triton_ffn_activation(self.ns_up(ns_tokens), "silu", backend=self.backend)
        ns_hidden = self.ns_down(self.dropout(ns_hidden))
        ns_hidden = self.dropout(ns_hidden) * ns_mask.unsqueeze(-1).float()
        return sequence_hidden, ns_hidden


class TaacMixedCausalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        ns_token_count: int,
        dropout: float,
        attention_dropout: float,
        *,
        attention_backend: AttentionBackend = "auto",
        ffn_backend: FeedForwardBackend = "auto",
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden_dim)
        self.ffn_norm = RMSNorm(hidden_dim)
        self.attention = TaacMixedCausalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ns_token_count=ns_token_count,
            dropout=attention_dropout,
            attention_backend=attention_backend,
        )
        self.ffn = TaacMixedCausalFeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            ns_token_count=ns_token_count,
            dropout=dropout,
            backend=ffn_backend,
        )

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        sequence_mask: torch.Tensor,
        ns_tokens: torch.Tensor,
        ns_mask: torch.Tensor,
        next_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_sequence = self.attention_norm(sequence_tokens)
        normalized_ns = self.attention_norm(ns_tokens)
        sequence_attention, next_sequence_mask, ns_attention = self.attention(
            normalized_sequence,
            sequence_mask,
            normalized_ns,
            ns_mask,
            next_sequence_length,
        )
        sequence_tokens = sequence_tokens[:, -next_sequence_length:] + sequence_attention
        sequence_mask = next_sequence_mask
        ns_tokens = ns_tokens + ns_attention

        normalized_sequence = self.ffn_norm(sequence_tokens)
        normalized_ns = self.ffn_norm(ns_tokens)
        sequence_ffn, ns_ffn = self.ffn(normalized_sequence, sequence_mask, normalized_ns, ns_mask)
        sequence_tokens = (sequence_tokens + sequence_ffn) * sequence_mask.unsqueeze(-1).float()
        ns_tokens = (ns_tokens + ns_ffn) * ns_mask.unsqueeze(-1).float()
        return sequence_tokens, sequence_mask, ns_tokens, ns_mask


__all__ = [
    "AttentionBackend",
    "AttentionType",
    "FeedForwardNetwork",
    "FeedForwardBackend",
    "FeedForwardType",
    "TaacMixedCausalAttention",
    "TaacMixedCausalBlock",
    "TaacMixedCausalFeedForward",
    "NormType",
    "TaacCrossAttentionBlock",
    "TaacTransformerBlock",
    "TokenSpecificLinear",
    "apply_token_specific_linear",
    "build_causal_attention_mask",
]
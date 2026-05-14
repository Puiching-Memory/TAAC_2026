"""Unified Symbiosis V2 backbone blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from taac2026.api import RMSNorm, scaled_dot_product_attention


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(d_model) * int(hidden_mult)
        self.up = nn.Linear(d_model, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(hidden_dim, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(tokens).chunk(2, dim=-1)
        return self.down(self.dropout(nn.functional.silu(gate) * value))


class UnifiedSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.query_norm = RMSNorm(d_model)
        self.key_norm = RMSNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query, key, value = self.qkv(tokens).chunk(3, dim=-1)
        query = self.query_norm(query).to(dtype=value.dtype)
        key = self.key_norm(key).to(dtype=value.dtype)
        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            num_heads=self.num_heads,
            attn_mask=attention_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.out(attended)


class UnifiedInteractionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = UnifiedSelfAttention(d_model, num_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, hidden_mult, dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_update = self.attention(self.attn_norm(tokens), attention_mask)
        attn_tokens = tokens + attn_update * (~padding_mask).to(attn_update.dtype).unsqueeze(-1)
        ffn_update = self.ffn(self.ffn_norm(attn_tokens))
        output = attn_tokens + ffn_update * (~padding_mask).to(ffn_update.dtype).unsqueeze(-1)
        return output, attn_tokens


__all__ = ["SwiGLUFeedForward", "UnifiedInteractionBlock", "UnifiedSelfAttention"]
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor, dim: int | Sequence[int] = 1) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    summed = (tokens * weights).sum(dim=dim)
    counts = weights.sum(dim=dim).clamp_min(1.0)
    return summed / counts


class MaskedMeanPool(nn.Module):
    def __init__(self, dim: int | Sequence[int] = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean(tokens, mask, dim=self.dim)


def _build_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "prelu":
        return nn.PReLU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}'")


class TargetAwarePool(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        scorer_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        include_difference: bool = True,
        include_absolute_difference: bool = False,
        include_product: bool = True,
    ) -> None:
        super().__init__()
        self.include_difference = include_difference
        self.include_absolute_difference = include_absolute_difference
        self.include_product = include_product

        feature_count = 2 + int(include_difference) + int(include_absolute_difference) + int(include_product)
        scorer_hidden_dim = scorer_hidden_dim or hidden_dim
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * feature_count, scorer_hidden_dim),
            _build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(scorer_hidden_dim, 1),
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        expanded_query = query.unsqueeze(1).expand_as(keys)
        attention_features = [expanded_query, keys]
        if self.include_difference:
            attention_features.append(expanded_query - keys)
        if self.include_absolute_difference:
            attention_features.append(torch.abs(expanded_query - keys))
        if self.include_product:
            attention_features.append(expanded_query * keys)

        attention_inputs = torch.cat(attention_features, dim=-1)
        attention_scores = self.scorer(attention_inputs).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~key_mask, -1.0e4)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights * key_mask.to(dtype=attention_weights.dtype)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)


__all__ = ["MaskedMeanPool", "TargetAwarePool", "masked_mean"]
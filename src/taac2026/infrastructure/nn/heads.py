from __future__ import annotations

from collections.abc import Sequence

from torch import nn


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


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int | Sequence[int],
        *,
        output_dim: int = 1,
        activation: str = "gelu",
        dropout: float | Sequence[float] = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        resolved_hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else list(hidden_dims)
        if not resolved_hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

        if isinstance(dropout, Sequence) and not isinstance(dropout, (str, bytes)):
            dropout_schedule = [float(value) for value in dropout]
        else:
            dropout_schedule = [float(dropout)] * len(resolved_hidden_dims)
        if len(dropout_schedule) != len(resolved_hidden_dims):
            raise ValueError("dropout schedule length must match hidden_dims length")

        layers: list[nn.Module] = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))

        current_dim = input_dim
        for next_dim, next_dropout in zip(resolved_hidden_dims, dropout_schedule, strict=True):
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(_build_activation(activation))
            layers.append(nn.Dropout(next_dropout))
            current_dim = next_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, hidden_states):
        return self.layers(hidden_states)


__all__ = ["ClassificationHead"]
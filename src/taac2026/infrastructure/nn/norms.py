from __future__ import annotations

import torch
from torch import nn


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if hidden_states.device.type == "cuda":
        from .triton_norm import triton_rms_norm

        return triton_rms_norm(hidden_states, weight, eps=eps)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_states * torch.rsqrt(variance + eps)
    return normalized * weight


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden_states, self.weight, eps=self.eps)


__all__ = ["RMSNorm", "rms_norm"]
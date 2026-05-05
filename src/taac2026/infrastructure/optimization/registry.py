"""Dense optimizer construction helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

from taac2026.infrastructure.optimization.muon import Muon


def dense_optimizer_display_name(dense_optimizer_type: str) -> str:
    if dense_optimizer_type == "fused_adamw":
        return "Fused AdamW"
    if dense_optimizer_type == "orthogonal_adamw":
        return "Orthogonal AdamW"
    if dense_optimizer_type == "muon":
        return "Muon"
    return "AdamW"


def build_dense_optimizer(
    parameters: list[nn.Parameter],
    *,
    dense_optimizer_type: str,
    lr: float,
) -> torch.optim.Optimizer:
    if dense_optimizer_type == "fused_adamw":
        try:
            return torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.98), fused=True)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                f"fused_adamw is not supported for the current runtime or parameter set: {error}"
            ) from error
    if dense_optimizer_type == "muon":
        return Muon(parameters, lr=lr, adamw_betas=(0.9, 0.98))
    return torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.98))


__all__ = ["build_dense_optimizer", "dense_optimizer_display_name"]
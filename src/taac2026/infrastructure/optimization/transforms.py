"""Optimizer update and gradient transforms."""

from __future__ import annotations

import torch


def orthogonalize_gradient(gradient: torch.Tensor) -> None:
    if gradient.ndim < 2:
        return
    with torch.no_grad():
        original_dtype = gradient.dtype
        original_norm = gradient.norm().clamp_min(1e-12)
        matrix = gradient.float().reshape(gradient.shape[0], -1)
        transposed = matrix.shape[0] > matrix.shape[1]
        if transposed:
            matrix = matrix.t()
        matrix = matrix / matrix.norm().clamp_min(1e-12)
        for _ in range(2):
            matrix = 1.5 * matrix - 0.5 * matrix @ (matrix.t() @ matrix)
        if transposed:
            matrix = matrix.t()
        gradient.copy_((matrix.reshape_as(gradient) * original_norm).to(original_dtype))


__all__ = ["orthogonalize_gradient"]
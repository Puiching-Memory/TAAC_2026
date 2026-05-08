"""Runtime support checks for the chunked gated-delta-rule kernels."""

from __future__ import annotations

import torch

from taac2026.infrastructure.accelerators.tensor_validation import require_cuda_tensors
from taac2026.infrastructure.accelerators.tilelang_runtime import tilelang_available


def chunk_gated_delta_rule_available(device: torch.device | None = None) -> bool:
    if not tilelang_available():
        return False
    if device is None:
        return torch.cuda.is_available()
    return device.type == "cuda"


def require_chunk_gated_delta_rule_runtime_support(*tensors: torch.Tensor | None) -> None:
    require_cuda_tensors("chunk_gated_delta_rule", *tensors)
    if not tilelang_available():
        raise RuntimeError("chunk_gated_delta_rule requires tilelang")


__all__ = [
    "chunk_gated_delta_rule_available",
    "require_chunk_gated_delta_rule_runtime_support",
]

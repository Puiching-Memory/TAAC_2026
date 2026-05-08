"""Shared tensor validation helpers for accelerator operators."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def concrete_tensors(tensors: Iterable[torch.Tensor | None]) -> tuple[torch.Tensor, ...]:
    return tuple(tensor for tensor in tensors if tensor is not None)


def require_same_device(operator_name: str, *tensors: torch.Tensor | None) -> torch.device | None:
    checked_tensors = concrete_tensors(tensors)
    if not checked_tensors:
        return None
    devices = {tensor.device for tensor in checked_tensors}
    if len(devices) != 1:
        raise RuntimeError(f"{operator_name} requires all tensors to live on the same device")
    return checked_tensors[0].device


def require_cuda_tensors(operator_name: str, *tensors: torch.Tensor | None) -> torch.device | None:
    device = require_same_device(operator_name, *tensors)
    if device is not None and device.type != "cuda":
        raise RuntimeError(f"{operator_name} requires CUDA tensors")
    return device


def require_same_dtype(operator_name: str, *tensors: torch.Tensor | None) -> torch.dtype | None:
    checked_tensors = concrete_tensors(tensors)
    if not checked_tensors:
        return None
    dtypes = {tensor.dtype for tensor in checked_tensors}
    if len(dtypes) != 1:
        raise ValueError(f"{operator_name} requires tensors to share the same dtype")
    return checked_tensors[0].dtype


__all__ = [
    "concrete_tensors",
    "require_cuda_tensors",
    "require_same_device",
    "require_same_dtype",
]

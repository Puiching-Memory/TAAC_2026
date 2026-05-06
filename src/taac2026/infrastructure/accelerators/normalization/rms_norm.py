"""RMSNorm operator boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from taac2026.infrastructure.accelerators.tilelang_runtime import (
    T,
    _ensure_tilelang_cuda_fp8_compatibility,
    tilelang_available,
    tilelang_dtype,
)
from taac2026.infrastructure.accelerators.normalization.kernels.tilelang import (
    build_rms_norm_backward_kernel,
    build_rms_norm_forward_kernel,
)


RMSNormKernel = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
RMSNormBackend = Literal["torch", "tilelang"]

_rms_norm_kernel: RMSNormKernel | None = None


@dataclass(frozen=True, slots=True)
class RMSNormKernelKey:
    rows: int
    cols: int
    dtype: torch.dtype
    eps: float
    block_rows: int


_rms_norm_forward_kernel_cache: dict[RMSNormKernelKey, Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = {}
_rms_norm_backward_kernel_cache: dict[
    RMSNormKernelKey, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
] = {}


def clear_rms_norm_kernel_cache() -> None:
    _rms_norm_forward_kernel_cache.clear()
    _rms_norm_backward_kernel_cache.clear()


def register_rms_norm_kernel(kernel: RMSNormKernel) -> None:
    global _rms_norm_kernel
    _rms_norm_kernel = kernel


def _torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * scale * weight


def _normalize_rms_norm_inputs(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    if x.ndim < 2:
        raise ValueError("rms_norm expects input with at least 2 dimensions")
    if weight.ndim != 1:
        raise ValueError("rms_norm weight must be a 1D tensor")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(f"last dimension {x.shape[-1]} does not match weight size {weight.shape[0]}")
    original_shape = x.shape
    matrix = x.reshape(-1, x.shape[-1]).contiguous()
    normalized_weight = weight.to(device=matrix.device, dtype=matrix.dtype).contiguous()
    return matrix, normalized_weight, original_shape


def _resolve_rms_norm_backend(x: torch.Tensor, backend: RMSNormBackend) -> Literal["torch", "tilelang"]:
    if backend == "torch":
        return "torch"
    if not tilelang_available():
        raise RuntimeError("tilelang backend requested but tilelang is not installed")
    if x.device.type != "cuda":
        raise RuntimeError("tilelang rms_norm currently requires CUDA tensors")
    if not _is_power_of_two(int(x.shape[-1])):
        raise RuntimeError("tilelang rms_norm currently requires the last dimension to be a power of two")
    return "tilelang"


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _rms_norm_registered_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if _rms_norm_kernel is None:
        raise RuntimeError("no registered rms_norm kernel is available")
    return _rms_norm_kernel(x, weight, eps)


def _run_torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if _rms_norm_kernel is not None:
        return _rms_norm_registered_kernel(x, weight, eps)
    return _torch_rms_norm(x, weight, eps)


def _rms_norm_cache_key(x: torch.Tensor, eps: float, block_rows: int | None) -> RMSNormKernelKey:
    return RMSNormKernelKey(
        rows=x.shape[0],
        cols=x.shape[1],
        dtype=x.dtype,
        eps=float(eps),
        block_rows=max(1, int(block_rows or 1)),
    )


def _compile_tilelang_rms_norm_kernel(key: RMSNormKernelKey) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    forward_kernel = _compile_tilelang_rms_norm_forward_kernel(key)

    def runner(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out, _inv_rms = forward_kernel(x, weight)
        return out

    return runner


def _compile_tilelang_rms_norm_forward_kernel(
    key: RMSNormKernelKey,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    if not tilelang_available():
        raise RuntimeError("tilelang is not installed")
    if key in _rms_norm_forward_kernel_cache:
        return _rms_norm_forward_kernel_cache[key]

    _ensure_tilelang_cuda_fp8_compatibility()
    tl_dtype = tilelang_dtype(key.dtype)
    accum_dtype = T.float32
    compiled = build_rms_norm_forward_kernel(key.rows, key.cols, key.block_rows, key.eps, tl_dtype, accum_dtype)

    def runner(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return compiled(x, weight)

    _rms_norm_forward_kernel_cache[key] = runner
    return runner


def _compile_tilelang_rms_norm_backward_kernel(
    key: RMSNormKernelKey,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    if not tilelang_available():
        raise RuntimeError("tilelang is not installed")
    if key in _rms_norm_backward_kernel_cache:
        return _rms_norm_backward_kernel_cache[key]

    _ensure_tilelang_cuda_fp8_compatibility()
    tl_dtype = tilelang_dtype(key.dtype)
    accum_dtype = T.float32
    compiled = build_rms_norm_backward_kernel(key.rows, key.cols, key.block_rows, tl_dtype, accum_dtype)

    def runner(
        x: torch.Tensor,
        weight: torch.Tensor,
        inv_rms: torch.Tensor,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compiled(x, weight, inv_rms, grad_out)

    _rms_norm_backward_kernel_cache[key] = runner
    return runner


def compile_rms_norm_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    block_rows: int | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    matrix, _normalized_weight, _original_shape = _normalize_rms_norm_inputs(x, weight)
    key = _rms_norm_cache_key(matrix, eps, block_rows)
    return _compile_tilelang_rms_norm_kernel(key)


class _TilelangRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float, block_rows: int) -> torch.Tensor:
        key = _rms_norm_cache_key(x, eps, block_rows)
        forward_kernel = _compile_tilelang_rms_norm_forward_kernel(key)
        out, inv_rms = forward_kernel(x, weight)
        ctx.save_for_backward(x, weight, inv_rms)
        ctx.key = key
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, inv_rms = ctx.saved_tensors
        backward_kernel = _compile_tilelang_rms_norm_backward_kernel(ctx.key)
        grad_x, grad_weight_partial = backward_kernel(x, weight, inv_rms, grad_out.contiguous())
        grad_weight = grad_weight_partial.sum(dim=0).to(weight.dtype)
        return grad_x, grad_weight, None, None


def _run_tilelang_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    block_rows: int | None,
) -> torch.Tensor:
    resolved_block_rows = max(1, int(block_rows or 1))
    if x.requires_grad or weight.requires_grad:
        return _TilelangRMSNormFunction.apply(x, weight, eps, resolved_block_rows)
    kernel = compile_rms_norm_kernel(x, weight, eps, block_rows=resolved_block_rows)
    return kernel(x, weight)


def resolved_rms_norm_backend(
    x: torch.Tensor,
    backend: RMSNormBackend,
    *,
    eps: float = 1e-6,
    block_rows: int | None = None,
) -> Literal["torch", "tilelang"]:
    del eps, block_rows
    matrix = x.reshape(-1, x.shape[-1]).contiguous() if x.ndim >= 2 else x
    return _resolve_rms_norm_backend(matrix, backend)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    backend: RMSNormBackend,
    block_rows: int | None = None,
) -> torch.Tensor:
    matrix, normalized_weight, original_shape = _normalize_rms_norm_inputs(x, weight)
    resolved_backend = resolved_rms_norm_backend(matrix, backend, eps=eps, block_rows=block_rows)
    if resolved_backend == "torch":
        return _run_torch_rms_norm(matrix, normalized_weight, eps).reshape(original_shape)
    return _run_tilelang_rms_norm(
        matrix,
        normalized_weight,
        eps,
        block_rows=block_rows,
    ).reshape(original_shape)


__all__ = [
    "RMSNormBackend",
    "RMSNormKernel",
    "RMSNormKernelKey",
    "clear_rms_norm_kernel_cache",
    "compile_rms_norm_kernel",
    "register_rms_norm_kernel",
    "resolved_rms_norm_backend",
    "rms_norm",
]

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl


FeedForwardBackend = Literal["auto", "torch", "triton", "te"]
FeedForwardActivation = Literal["gelu", "silu", "swiglu"]
FeedForwardPrecision = Literal["native", "fp8-e4m3fn", "fp8-e5m2"]
_FP8_DTYPE_MAP = {
    "fp8-e4m3fn": torch.float8_e4m3fn,
    "fp8-e5m2": torch.float8_e5m2,
}
_FP8_DTYPES = frozenset(_FP8_DTYPE_MAP.values())


@triton.jit
def _ffn_activation_kernel(
    input_ptr,
    output_ptr,
    total_elements,
    hidden_dim,
    MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_index = tl.program_id(0)
    offsets = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    if MODE == 2:
        row_offsets = offsets // hidden_dim
        col_offsets = offsets % hidden_dim
        value_offsets = row_offsets * hidden_dim * 2 + col_offsets
        gate_offsets = value_offsets + hidden_dim
        values = tl.load(input_ptr + value_offsets, mask=mask, other=0.0).to(tl.float32)
        gates = tl.load(input_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
        activated = values * (gates * tl.sigmoid(gates))
    else:
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if MODE == 0:
            inv_sqrt_2 = 0.7071067811865476
            activated = 0.5 * values * (1.0 + tl.erf(values * inv_sqrt_2))
        else:
            activated = values * tl.sigmoid(values)

    tl.store(output_ptr + offsets, activated, mask=mask)


def _normalize_activation_name(name: FeedForwardActivation | str) -> FeedForwardActivation:
    normalized = str(name).strip().lower()
    if normalized not in {"gelu", "silu", "swiglu"}:
        raise ValueError(f"Unsupported feed-forward activation '{name}'")
    return normalized  # type: ignore[return-value]


def _normalize_precision_name(name: FeedForwardPrecision | str) -> FeedForwardPrecision:
    normalized = str(name).strip().lower()
    if normalized not in {"native", *tuple(_FP8_DTYPE_MAP)}:
        raise ValueError(f"Unsupported feed-forward precision '{name}'")
    return normalized  # type: ignore[return-value]


def _apply_precision(projected: torch.Tensor, precision: FeedForwardPrecision | str) -> torch.Tensor:
    resolved_precision = _normalize_precision_name(precision)
    if resolved_precision == "native":
        return projected
    return projected.to(_FP8_DTYPE_MAP[resolved_precision]).to(dtype=projected.dtype)


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FP8_DTYPES


def _prepare_projected_tensor(projected: torch.Tensor, precision: FeedForwardPrecision) -> torch.Tensor:
    if precision == "native":
        return projected
    target_dtype = _FP8_DTYPE_MAP[precision]
    if projected.dtype == target_dtype:
        return projected
    return projected.to(dtype=target_dtype)


def _resolve_activation_output_dtype(projected: torch.Tensor) -> torch.dtype:
    if _is_fp8_dtype(projected.dtype):
        return torch.float16
    return projected.dtype


def _supports_triton_fp8(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    capability_major, _ = torch.cuda.get_device_capability(device)
    return capability_major >= 9


def _can_use_triton_activation(
    tensor: torch.Tensor,
    backend: FeedForwardBackend | str,
    precision: FeedForwardPrecision | str,
) -> bool:
    normalized_backend = str(backend).strip().lower()
    if normalized_backend in {"torch", "te"}:
        return False
    if tensor.device.type != "cuda" or tensor.requires_grad:
        return False
    resolved_precision = _normalize_precision_name(precision)
    if resolved_precision != "native" and not _supports_triton_fp8(tensor.device):
        return False
    allowed_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if resolved_precision != "native":
        allowed_dtypes.add(_FP8_DTYPE_MAP[resolved_precision])
    return tensor.dtype in allowed_dtypes


def reference_ffn_activation(
    projected: torch.Tensor,
    activation: FeedForwardActivation | str,
    *,
    precision: FeedForwardPrecision | str = "native",
) -> torch.Tensor:
    resolved = _normalize_activation_name(activation)
    projected = _apply_precision(projected, precision)
    if resolved == "swiglu":
        value, gate = projected.chunk(2, dim=-1)
        return value * F.silu(gate)
    if resolved == "gelu":
        return F.gelu(projected)
    return F.silu(projected)


def triton_ffn_activation(
    projected: torch.Tensor,
    activation: FeedForwardActivation | str,
    *,
    backend: FeedForwardBackend | str = "auto",
    precision: FeedForwardPrecision | str = "native",
) -> torch.Tensor:
    resolved = _normalize_activation_name(activation)
    resolved_precision = _normalize_precision_name(precision)
    if projected.ndim < 2:
        raise ValueError("projected tensor must have at least two dimensions")

    if not _can_use_triton_activation(projected, backend, resolved_precision):
        return reference_ffn_activation(projected, resolved, precision=resolved_precision)

    output_dtype = _resolve_activation_output_dtype(projected)
    projected_for_kernel = _prepare_projected_tensor(projected, resolved_precision)

    if resolved == "swiglu":
        hidden_dim = projected_for_kernel.shape[-1] // 2
        flattened_input = projected_for_kernel.contiguous().view(-1, hidden_dim * 2)
        output = torch.empty(flattened_input.shape[0] * hidden_dim, device=projected.device, dtype=output_dtype)
        total_elements = output.numel()
        mode = 2
    else:
        hidden_dim = projected_for_kernel.shape[-1]
        flattened_input = projected_for_kernel.contiguous().view(-1)
        output = torch.empty(flattened_input.shape[0], device=projected.device, dtype=output_dtype)
        total_elements = output.numel()
        mode = 0 if resolved == "gelu" else 1

    block_size = min(4096, triton.next_power_of_2(max(1, min(total_elements, 2048))))
    _ffn_activation_kernel[(triton.cdiv(total_elements, block_size),)](
        flattened_input,
        output,
        total_elements,
        hidden_dim,
        MODE=mode,
        BLOCK_SIZE=block_size,
        num_warps=4 if block_size <= 256 else 8,
    )

    if resolved == "swiglu":
        return output.view(*projected_for_kernel.shape[:-1], hidden_dim)
    return output.view_as(projected)


class TritonFeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        *,
        activation: FeedForwardActivation = "swiglu",
        dropout: float = 0.0,
        backend: FeedForwardBackend = "auto",
        precision: FeedForwardPrecision = "native",
    ) -> None:
        super().__init__()
        self.activation = _normalize_activation_name(activation)
        self.backend = backend
        self.precision = precision
        self.dropout = nn.Dropout(dropout)
        input_dim = ffn_dim * 2 if self.activation == "swiglu" else ffn_dim
        self.up_projection = nn.Linear(hidden_dim, input_dim)
        self.down_projection = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.up_projection(hidden_states)
        activated = triton_ffn_activation(projected, self.activation, backend=self.backend, precision=self.precision)
        activated = self.dropout(activated)
        return self.down_projection(activated)


__all__ = [
    "FeedForwardActivation",
    "FeedForwardBackend",
    "FeedForwardPrecision",
    "TritonFeedForward",
    "reference_ffn_activation",
    "triton_ffn_activation",
]
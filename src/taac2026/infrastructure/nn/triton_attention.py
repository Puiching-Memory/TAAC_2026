from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl


AttentionMode = Literal["softmax", "silu"]
AttentionBackend = Literal["auto", "torch", "triton", "te"]
AttentionPrecision = Literal["native", "fp8-e4m3fn", "fp8-e5m2"]
_FP8_DTYPE_MAP = {
    "fp8-e4m3fn": torch.float8_e4m3fn,
    "fp8-e5m2": torch.float8_e5m2,
}
_FP8_DTYPES = frozenset(_FP8_DTYPE_MAP.values())


@triton.jit
def _attention_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    bias_ptr,
    mask_ptr,
    output_ptr,
    query_bh_stride,
    query_seq_stride,
    query_dim_stride,
    key_bh_stride,
    key_seq_stride,
    key_dim_stride,
    value_bh_stride,
    value_seq_stride,
    value_dim_stride,
    bias_batch_stride,
    bias_query_stride,
    bias_key_stride,
    mask_batch_stride,
    mask_query_stride,
    mask_key_stride,
    output_bh_stride,
    output_seq_stride,
    output_dim_stride,
    num_heads,
    query_length,
    key_length,
    head_dim,
    scale,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    USE_DOT_FP8: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_head_index = tl.program_id(0)
    query_index = tl.program_id(1)

    batch_index = batch_head_index // num_heads
    dim_offsets = tl.arange(0, BLOCK_D)
    key_offsets = tl.arange(0, BLOCK_K)
    dim_mask = dim_offsets < head_dim
    key_mask = key_offsets < key_length

    query_values = tl.load(
        query_ptr + batch_head_index * query_bh_stride + query_index * query_seq_stride + dim_offsets * query_dim_stride,
        mask=dim_mask,
        other=0.0,
    )
    key_values = tl.load(
        key_ptr
        + batch_head_index * key_bh_stride
        + key_offsets[:, None] * key_seq_stride
        + dim_offsets[None, :] * key_dim_stride,
        mask=key_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    if USE_DOT_FP8:
        query_matrix = tl.expand_dims(query_values, 0)
        scores = tl.reshape(tl.dot(query_matrix, tl.trans(key_values), out_dtype=tl.float32), (BLOCK_K,)) * scale
    else:
        query_values = query_values.to(tl.float32)
        key_values = key_values.to(tl.float32)
        scores = tl.sum(key_values * tl.expand_dims(query_values, 0), axis=1) * scale

    if HAS_BIAS:
        bias_values = tl.load(
            bias_ptr + batch_index * bias_batch_stride + query_index * bias_query_stride + key_offsets * bias_key_stride,
            mask=key_mask,
            other=0.0,
        )
        scores = scores + bias_values

    if HAS_MASK:
        mask_values = tl.load(
            mask_ptr + batch_index * mask_batch_stride + query_index * mask_query_stride + key_offsets * mask_key_stride,
            mask=key_mask,
            other=0,
        )
        valid_mask = key_mask & (mask_values > 0)
    else:
        valid_mask = key_mask

    if MODE == 0:
        neg_inf = tl.full([BLOCK_K], -1.0e9, tl.float32)
        masked_scores = tl.where(valid_mask, scores, neg_inf)
        max_score = tl.max(masked_scores, axis=0)
        weights = tl.exp(masked_scores - max_score)
        weights = tl.where(valid_mask, weights, 0.0)
        denom = tl.sum(weights, axis=0)
        weights = weights / tl.maximum(denom, 1.0e-9)
    else:
        scores = tl.where(valid_mask, scores, 0.0)
        weights = scores * tl.sigmoid(scores)

    value_values = tl.load(
        value_ptr
        + batch_head_index * value_bh_stride
        + key_offsets[:, None] * value_seq_stride
        + dim_offsets[None, :] * value_dim_stride,
        mask=key_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    if USE_DOT_FP8:
        weight_matrix = tl.expand_dims(weights, 0)
        output_values = tl.reshape(tl.dot(weight_matrix, value_values.to(tl.float32), out_dtype=tl.float32), (BLOCK_D,))
    else:
        value_values = value_values.to(tl.float32)
        output_values = tl.sum(tl.expand_dims(weights, 1) * value_values, axis=0)
    tl.store(
        output_ptr + batch_head_index * output_bh_stride + query_index * output_seq_stride + dim_offsets * output_dim_stride,
        output_values,
        mask=dim_mask,
    )


def _normalize_attention_mode(mode: AttentionMode | str) -> AttentionMode:
    normalized = str(mode).strip().lower()
    if normalized not in {"softmax", "silu"}:
        raise ValueError(f"Unsupported attention mode '{mode}'")
    return normalized  # type: ignore[return-value]


def _normalize_attention_precision(precision: AttentionPrecision | str) -> AttentionPrecision:
    normalized = str(precision).strip().lower()
    if normalized not in {"native", *tuple(_FP8_DTYPE_MAP)}:
        raise ValueError(f"Unsupported attention precision '{precision}'")
    return normalized  # type: ignore[return-value]


def _apply_attention_precision(tensor: torch.Tensor, precision: AttentionPrecision | str) -> torch.Tensor:
    resolved_precision = _normalize_attention_precision(precision)
    if resolved_precision == "native":
        return tensor
    return tensor.to(_FP8_DTYPE_MAP[resolved_precision]).to(dtype=tensor.dtype)


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FP8_DTYPES


def _prepare_attention_tensor(tensor: torch.Tensor, precision: AttentionPrecision) -> torch.Tensor:
    if precision == "native":
        return tensor
    target_dtype = _FP8_DTYPE_MAP[precision]
    if tensor.dtype == target_dtype:
        return tensor
    return tensor.to(dtype=target_dtype)


def _resolve_attention_output_dtype(query: torch.Tensor, value: torch.Tensor) -> torch.dtype:
    if _is_fp8_dtype(query.dtype):
        if _is_fp8_dtype(value.dtype):
            return torch.float16
        return value.dtype
    return query.dtype


def _supports_triton_fp8(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    capability_major, _ = torch.cuda.get_device_capability(device)
    return capability_major >= 9


def resolve_attention_mask(
    *,
    query_length: int,
    key_length: int,
    batch_size: int,
    device: torch.device,
    attention_mask: torch.Tensor | None,
    query_mask: torch.Tensor | None,
    key_mask: torch.Tensor | None,
    is_causal: bool,
) -> torch.Tensor | None:
    combined_mask = None
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            combined_mask = attention_mask.to(dtype=torch.bool, device=device)
        else:
            combined_mask = attention_mask.to(device=device)
        if combined_mask.ndim == 2:
            combined_mask = combined_mask.unsqueeze(0)
        elif combined_mask.ndim == 4:
            combined_mask = combined_mask[:, 0]
        if combined_mask.shape[-2:] != (query_length, key_length):
            raise ValueError("attention_mask shape must match [query_length, key_length]")

    if key_mask is not None:
        key_visibility = key_mask.to(device=device, dtype=torch.bool).unsqueeze(1).expand(-1, query_length, -1)
        combined_mask = key_visibility if combined_mask is None else (combined_mask & key_visibility)

    if query_mask is not None:
        query_visibility = query_mask.to(device=device, dtype=torch.bool).unsqueeze(-1).expand(-1, -1, key_length)
        combined_mask = query_visibility if combined_mask is None else (combined_mask & query_visibility)

    if is_causal:
        causal_mask = torch.tril(torch.ones(query_length, key_length, dtype=torch.bool, device=device))
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        combined_mask = causal_mask if combined_mask is None else (combined_mask & causal_mask)

    return combined_mask


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    query_mask: torch.Tensor | None = None,
    key_mask: torch.Tensor | None = None,
    additive_bias: torch.Tensor | None = None,
    is_causal: bool = False,
    mode: AttentionMode | str = "softmax",
    precision: AttentionPrecision | str = "native",
) -> torch.Tensor:
    resolved_mode = _normalize_attention_mode(mode)
    query = _apply_attention_precision(query, precision)
    key = _apply_attention_precision(key, precision)
    value = _apply_attention_precision(value, precision)
    batch_size, _, query_length, head_dim = query.shape
    key_length = key.shape[-2]
    combined_mask = resolve_attention_mask(
        query_length=query_length,
        key_length=key_length,
        batch_size=batch_size,
        device=query.device,
        attention_mask=attention_mask,
        query_mask=query_mask,
        key_mask=key_mask,
        is_causal=is_causal,
    )

    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    if additive_bias is not None:
        scores = scores + additive_bias.to(device=query.device, dtype=scores.dtype).unsqueeze(1)

    if resolved_mode == "softmax":
        if combined_mask is not None:
            scores = scores.masked_fill(~combined_mask.unsqueeze(1), -1.0e9)
        weights = torch.softmax(scores, dim=-1).to(dtype=query.dtype)
        if combined_mask is not None:
            weights = weights * combined_mask.unsqueeze(1).to(dtype=weights.dtype)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
            weights = weights / denom
    else:
        weights = F.silu(scores).to(dtype=query.dtype)
        if combined_mask is not None:
            weights = weights * combined_mask.unsqueeze(1).to(dtype=weights.dtype)

    return torch.matmul(weights, value)


def _can_use_triton_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    backend: AttentionBackend | str,
    precision: AttentionPrecision | str,
) -> bool:
    normalized_backend = str(backend).strip().lower()
    if normalized_backend in {"torch", "te"}:
        return False
    if query.device.type != "cuda":
        return False
    if query.requires_grad or key.requires_grad or value.requires_grad:
        return False
    resolved_precision = _normalize_attention_precision(precision)
    allowed_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if resolved_precision != "native":
        allowed_dtypes.add(_FP8_DTYPE_MAP[resolved_precision])
    if query.dtype not in allowed_dtypes or key.dtype not in allowed_dtypes or value.dtype not in allowed_dtypes:
        return False
    if resolved_precision != "native" and not _supports_triton_fp8(query.device):
        return False
    if key.shape[-2] > 256 or query.shape[-2] > 256 or query.shape[-1] > 128:
        return False
    return True


def _attention_block_sizes(
    *,
    query_length: int,
    key_length: int,
    head_dim: int,
    precision: AttentionPrecision | str,
) -> tuple[int, int]:
    resolved_precision = _normalize_attention_precision(precision)
    min_head_dim = 32 if resolved_precision != "native" else 16
    block_k = min(256, triton.next_power_of_2(max(16, key_length)))
    block_d = min(128, triton.next_power_of_2(max(min_head_dim, head_dim)))
    if block_k < key_length:
        raise ValueError(f"key_length {key_length} exceeds supported BLOCK_K {block_k}")
    if block_d < head_dim:
        raise ValueError(f"head_dim {head_dim} exceeds supported BLOCK_D {block_d}")
    return block_k, block_d


def triton_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    query_mask: torch.Tensor | None = None,
    key_mask: torch.Tensor | None = None,
    additive_bias: torch.Tensor | None = None,
    is_causal: bool = False,
    mode: AttentionMode | str = "softmax",
    backend: AttentionBackend | str = "auto",
    precision: AttentionPrecision | str = "native",
) -> torch.Tensor:
    resolved_mode = _normalize_attention_mode(mode)
    resolved_precision = _normalize_attention_precision(precision)
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, heads, tokens, head_dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must share the same shape")
    if query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query, key, and value must agree on batch, heads, and head_dim")

    batch_size, num_heads, query_length, head_dim = query.shape
    key_length = key.shape[-2]
    combined_mask = resolve_attention_mask(
        query_length=query_length,
        key_length=key_length,
        batch_size=batch_size,
        device=query.device,
        attention_mask=attention_mask,
        query_mask=query_mask,
        key_mask=key_mask,
        is_causal=is_causal,
    )

    if not _can_use_triton_attention(query, key, value, backend, resolved_precision):
        return reference_attention(
            query,
            key,
            value,
            attention_mask=combined_mask,
            query_mask=None,
            key_mask=None,
            additive_bias=additive_bias,
            is_causal=False,
            mode=resolved_mode,
            precision=resolved_precision,
        )

    output_dtype = _resolve_attention_output_dtype(query, value)
    query_for_kernel = _prepare_attention_tensor(query, resolved_precision)
    key_for_kernel = _prepare_attention_tensor(key, resolved_precision)
    value_for_kernel = _prepare_attention_tensor(value, resolved_precision)

    flattened_query = query_for_kernel.contiguous().view(batch_size * num_heads, query_length, head_dim)
    flattened_key = key_for_kernel.contiguous().view(batch_size * num_heads, key_length, head_dim)
    flattened_value = value_for_kernel.contiguous().view(batch_size * num_heads, key_length, head_dim)
    output = torch.empty(
        (batch_size * num_heads, query_length, head_dim),
        device=query.device,
        dtype=output_dtype,
    )

    if additive_bias is None:
        additive_bias = torch.zeros(batch_size, query_length, key_length, device=query.device, dtype=torch.float32)
        has_bias = False
    else:
        additive_bias = additive_bias.to(device=query.device, dtype=torch.float32).contiguous()
        has_bias = True

    if combined_mask is None:
        combined_mask = torch.ones(batch_size, query_length, key_length, device=query.device, dtype=torch.int32)
        has_mask = False
    else:
        combined_mask = combined_mask.to(device=query.device, dtype=torch.int32).contiguous()
        has_mask = True

    block_k, block_d = _attention_block_sizes(
        query_length=query_length,
        key_length=key_length,
        head_dim=head_dim,
        precision=resolved_precision,
    )
    _attention_forward_kernel[(batch_size * num_heads, query_length)](
        flattened_query,
        flattened_key,
        flattened_value,
        additive_bias,
        combined_mask,
        output,
        flattened_query.stride(0),
        flattened_query.stride(1),
        flattened_query.stride(2),
        flattened_key.stride(0),
        flattened_key.stride(1),
        flattened_key.stride(2),
        flattened_value.stride(0),
        flattened_value.stride(1),
        flattened_value.stride(2),
        additive_bias.stride(0),
        additive_bias.stride(1),
        additive_bias.stride(2),
        combined_mask.stride(0),
        combined_mask.stride(1),
        combined_mask.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        query_length,
        key_length,
        head_dim,
        1.0 / math.sqrt(head_dim),
        HAS_BIAS=has_bias,
        HAS_MASK=has_mask,
        USE_DOT_FP8=resolved_precision != "native",
        MODE=0 if resolved_mode == "softmax" else 1,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4 if block_k <= 64 else 8,
    )
    return output.view(batch_size, num_heads, query_length, head_dim)


class TritonAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        mode: AttentionMode = "softmax",
        backend: AttentionBackend = "auto",
        precision: AttentionPrecision = "native",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mode = mode
        self.backend = backend
        self.precision = precision
        self.dropout = nn.Dropout(dropout)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _reshape_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = hidden_states.shape
        return hidden_states.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _merge_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, token_count, _ = hidden_states.shape
        return hidden_states.transpose(1, 2).contiguous().view(batch_size, token_count, self.hidden_dim)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        additive_bias: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        query = self._reshape_heads(self.query_projection(query_states))
        key = self._reshape_heads(self.key_projection(key_states))
        value = self._reshape_heads(self.value_projection(value_states))
        attended = triton_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            query_mask=query_mask,
            key_mask=key_mask,
            additive_bias=additive_bias,
            is_causal=is_causal,
            mode=self.mode,
            backend=self.backend,
            precision=self.precision,
        )
        projected = self.output_projection(self._merge_heads(attended))
        return self.dropout(projected)


__all__ = [
    "AttentionBackend",
    "AttentionMode",
    "AttentionPrecision",
    "TritonAttention",
    "reference_attention",
    "resolve_attention_mask",
    "triton_attention",
]
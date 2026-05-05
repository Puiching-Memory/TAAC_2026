"""Flash attention operator boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from taac2026.infrastructure.accelerators.tilelang_runtime import (
    T,
    _ensure_tilelang_cuda_fp8_compatibility,
    build_flash_attention_forward_kernel,
    tilelang_available,
    tilelang_dtype,
)


FlashAttentionKernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
FlashAttentionBackend = Literal["torch", "tilelang"]

_flash_attention_kernel: FlashAttentionKernel | None = None


@dataclass(frozen=True, slots=True)
class FlashAttentionKernelKey:
    batch: int
    heads: int
    query_len: int
    kv_len: int
    head_dim: int
    dtype: torch.dtype
    is_causal: bool
    block_m: int
    block_n: int
    num_stages: int
    threads: int


@dataclass(frozen=True, slots=True)
class FlashAttentionMaskPlan:
    key_lengths: torch.Tensor
    query_self_mask: torch.Tensor | None
    is_causal: bool


_flash_attention_forward_kernel_cache: dict[FlashAttentionKernelKey, FlashAttentionKernel] = {}


def clear_flash_attention_kernel_cache() -> None:
    _flash_attention_forward_kernel_cache.clear()


def register_flash_attention_kernel(kernel: FlashAttentionKernel) -> None:
    global _flash_attention_kernel
    _flash_attention_kernel = kernel


def _normalize_flash_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("flash_attention expects q, k, and v with shape (batch, heads, seq_len, head_dim)")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("flash_attention requires matching batch dimensions")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("flash_attention requires matching head counts")
    if k.shape[2] != v.shape[2]:
        raise ValueError("flash_attention requires matching key and value sequence lengths")
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise ValueError("flash_attention requires matching head dimensions")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("flash_attention requires q, k, and v to share the same dtype")
    return q.contiguous(), k.contiguous(), v.contiguous()


def _full_key_lengths(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.full((q.shape[0],), k.shape[2], dtype=torch.int32, device=q.device)


def _is_prefix_valid_mask(valid: torch.Tensor) -> bool:
    positions = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
    lengths = valid.to(torch.int32).sum(dim=1, keepdim=True)
    expected = positions < lengths
    return bool(torch.equal(valid, expected))


def _build_causal_valid_mask_from_lengths(lengths: torch.Tensor, token_count: int, num_heads: int) -> torch.Tensor:
    positions = torch.arange(token_count, device=lengths.device)
    padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
    causal = torch.ones(token_count, token_count, dtype=torch.bool, device=lengths.device).tril()
    key_valid = ~padding_mask
    mask = causal.unsqueeze(0) & key_valid.unsqueeze(1)
    query_invalid = padding_mask.unsqueeze(-1)
    fallback = torch.eye(token_count, dtype=torch.bool, device=lengths.device).unsqueeze(0)
    mask = torch.where(query_invalid, fallback, mask)
    return mask.unsqueeze(1).expand(lengths.shape[0], num_heads, token_count, token_count)


def _plan_tilelang_flash_attention_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_mask: torch.Tensor | None,
    *,
    is_causal: bool,
) -> FlashAttentionMaskPlan:
    if attn_mask is None:
        return FlashAttentionMaskPlan(
            key_lengths=_full_key_lengths(q, k),
            query_self_mask=None,
            is_causal=is_causal,
        )

    if attn_mask.dtype is not torch.bool:
        raise RuntimeError("tilelang flash_attention only supports structured bool attention masks")
    if attn_mask.ndim != 4 or attn_mask.shape != (q.shape[0], q.shape[1], q.shape[2], k.shape[2]):
        raise RuntimeError("tilelang flash_attention only supports 4D bool masks matching (batch, heads, query_len, kv_len)")

    head0_mask = attn_mask[:, :1, :, :]
    if not torch.equal(attn_mask, head0_mask.expand_as(attn_mask)):
        raise RuntimeError("tilelang flash_attention requires the bool attention mask to be shared across heads")

    broadcast_key_valid = head0_mask[:, 0, :1, :].expand(q.shape[0], q.shape[2], k.shape[2])
    if torch.equal(head0_mask[:, 0], broadcast_key_valid):
        key_valid = head0_mask[:, 0, 0, :]
        if not _is_prefix_valid_mask(key_valid):
            raise RuntimeError("tilelang flash_attention requires bool key masks to be prefix-valid")
        return FlashAttentionMaskPlan(
            key_lengths=key_valid.to(torch.int32).sum(dim=1),
            query_self_mask=None,
            is_causal=is_causal,
        )

    if q.shape[2] != k.shape[2]:
        raise RuntimeError("tilelang flash_attention only supports non-broadcast bool masks when query and key lengths match")

    query_lengths = head0_mask[:, 0, :, 0].to(torch.int32).sum(dim=1)
    expected = _build_causal_valid_mask_from_lengths(query_lengths, q.shape[2], q.shape[1])
    if not torch.equal(attn_mask, expected):
        raise RuntimeError(
            "tilelang flash_attention only supports prefix key masks or causal_valid_attention_mask-style bool masks"
        )
    positions = torch.arange(q.shape[2], device=q.device).unsqueeze(0)
    query_self_mask = positions >= query_lengths.unsqueeze(1)
    return FlashAttentionMaskPlan(
        key_lengths=query_lengths,
        query_self_mask=query_self_mask,
        is_causal=True,
    )


def _apply_query_self_mask_fallback(
    output: torch.Tensor,
    value: torch.Tensor,
    query_self_mask: torch.Tensor | None,
) -> torch.Tensor:
    if query_self_mask is None:
        return output
    mask = query_self_mask.unsqueeze(1).unsqueeze(-1)
    return torch.where(mask, value, output)


def _flash_attention_cache_key(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    is_causal: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
) -> FlashAttentionKernelKey:
    return FlashAttentionKernelKey(
        batch=q.shape[0],
        heads=q.shape[1],
        query_len=q.shape[2],
        kv_len=k.shape[2],
        head_dim=q.shape[3],
        dtype=q.dtype,
        is_causal=bool(is_causal),
        block_m=max(1, int(block_m)),
        block_n=max(1, int(block_n)),
        num_stages=max(1, int(num_stages)),
        threads=max(1, int(threads)),
    )


def _resolve_flash_attention_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: FlashAttentionBackend,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    training: bool,
    is_causal: bool,
) -> Literal["torch", "tilelang"]:
    del k, v, is_causal
    if backend == "torch":
        return "torch"
    if training or dropout_p != 0.0:
        raise RuntimeError("tilelang flash_attention only supports inference-style forward without dropout")
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("tilelang flash_attention currently supports forward-only execution without autograd")
    if not tilelang_available():
        raise RuntimeError("tilelang backend requested but tilelang is not installed")
    if q.device.type != "cuda":
        raise RuntimeError("tilelang flash_attention currently requires CUDA tensors")
    return "tilelang"


def _torch_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    training: bool,
    is_causal: bool,
) -> torch.Tensor:
    effective_dropout = dropout_p if training else 0.0
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=effective_dropout,
        is_causal=is_causal if attn_mask is None else False,
    )


def _compile_tilelang_flash_attention_forward_kernel(key: FlashAttentionKernelKey) -> FlashAttentionKernel:
    if not tilelang_available():
        raise RuntimeError("tilelang is not installed")
    if build_flash_attention_forward_kernel is None:
        raise RuntimeError("tilelang flash attention kernel builder is unavailable")
    if key in _flash_attention_forward_kernel_cache:
        return _flash_attention_forward_kernel_cache[key]

    _ensure_tilelang_cuda_fp8_compatibility()
    tl_dtype = tilelang_dtype(key.dtype)
    accum_dtype = T.float32
    compiled = build_flash_attention_forward_kernel(
        key.batch,
        key.heads,
        key.query_len,
        key.kv_len,
        key.head_dim,
        is_causal=key.is_causal,
        block_m=key.block_m,
        block_n=key.block_n,
        num_stages=key.num_stages,
        threads=key.threads,
        tl_dtype=tl_dtype,
        accum_dtype=accum_dtype,
    )

    def runner(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_lengths: torch.Tensor) -> torch.Tensor:
        q_bshd = q.permute(0, 2, 1, 3).contiguous()
        k_bshd = k.permute(0, 2, 1, 3).contiguous()
        v_bshd = v.permute(0, 2, 1, 3).contiguous()
        output = compiled(q_bshd, k_bshd, v_bshd, key_lengths.to(device=q.device, dtype=torch.int32).contiguous())
        if isinstance(output, tuple):
            output = output[0]
        return output.permute(0, 2, 1, 3).contiguous()

    _flash_attention_forward_kernel_cache[key] = runner
    return runner


def compile_flash_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    num_stages: int = 1,
    threads: int = 128,
) -> FlashAttentionKernel:
    normalized_q, normalized_k, normalized_v = _normalize_flash_attention_inputs(q, k, v)
    key = _flash_attention_cache_key(
        normalized_q,
        normalized_k,
        is_causal=is_causal,
        block_m=block_m,
        block_n=block_n,
        num_stages=num_stages,
        threads=threads,
    )
    del normalized_v
    return _compile_tilelang_flash_attention_forward_kernel(key)


def resolved_flash_attention_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: FlashAttentionBackend,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    is_causal: bool = False,
) -> Literal["torch", "tilelang"]:
    normalized_q, normalized_k, normalized_v = _normalize_flash_attention_inputs(q, k, v)
    if backend == "tilelang":
        _plan_tilelang_flash_attention_mask(
            normalized_q,
            normalized_k,
            attn_mask,
            is_causal=is_causal,
        )
    return _resolve_flash_attention_backend(
        normalized_q,
        normalized_k,
        normalized_v,
        backend,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        training=training,
        is_causal=is_causal,
    )


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: FlashAttentionBackend,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    is_causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    num_stages: int = 1,
    threads: int = 128,
) -> torch.Tensor:
    normalized_q, normalized_k, normalized_v = _normalize_flash_attention_inputs(q, k, v)
    mask_plan = None
    if backend == "tilelang":
        mask_plan = _plan_tilelang_flash_attention_mask(
            normalized_q,
            normalized_k,
            attn_mask,
            is_causal=is_causal,
        )
    resolved_backend = _resolve_flash_attention_backend(
        normalized_q,
        normalized_k,
        normalized_v,
        backend,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        training=training,
        is_causal=is_causal,
    )
    if resolved_backend == "torch":
        return _torch_flash_attention(
            normalized_q,
            normalized_k,
            normalized_v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            training=training,
            is_causal=is_causal,
        )
    if mask_plan is None:
        raise RuntimeError("tilelang flash_attention mask planning did not run")
    if _flash_attention_kernel is not None:
        output = _flash_attention_kernel(normalized_q, normalized_k, normalized_v, mask_plan.key_lengths)
        return _apply_query_self_mask_fallback(output, normalized_v, mask_plan.query_self_mask)
    kernel = compile_flash_attention_kernel(
        normalized_q,
        normalized_k,
        normalized_v,
        is_causal=mask_plan.is_causal,
        block_m=block_m,
        block_n=block_n,
        num_stages=num_stages,
        threads=threads,
    )
    output = kernel(normalized_q, normalized_k, normalized_v, mask_plan.key_lengths)
    return _apply_query_self_mask_fallback(output, normalized_v, mask_plan.query_self_mask)


__all__ = [
    "FlashAttentionBackend",
    "FlashAttentionKernel",
    "FlashAttentionKernelKey",
    "FlashAttentionMaskPlan",
    "clear_flash_attention_kernel_cache",
    "compile_flash_attention_kernel",
    "flash_attention",
    "register_flash_attention_kernel",
    "resolved_flash_attention_backend",
]
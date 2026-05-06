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
    tilelang_available,
    tilelang_dtype,
)
from taac2026.infrastructure.accelerators.tensor_validation import require_cuda_tensors, require_same_dtype
from taac2026.infrastructure.accelerators.attention.kernels.tilelang import (
    build_flash_attention_backward_kernel,
    build_flash_attention_backward_preprocess_kernel,
    build_flash_attention_forward_kernel,
    build_flash_attention_training_forward_kernel,
)


FlashAttentionKernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
FlashAttentionTrainingForwardKernel = Callable[..., tuple[torch.Tensor, torch.Tensor]]
FlashAttentionBackwardPreprocessKernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
FlashAttentionBackwardKernel = Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
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
    use_dropout: bool = False


@dataclass(frozen=True, slots=True)
class FlashAttentionBackwardPreprocessKey:
    batch: int
    heads: int
    query_len: int
    head_dim: int
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class FlashAttentionMaskPlan:
    key_lengths: torch.Tensor
    query_self_mask: torch.Tensor | None
    is_causal: bool


@dataclass(frozen=True, slots=True)
class FlashAttentionLaunchRule:
    config: tuple[int, int, int, int]
    training: bool | None = None
    use_dropout: bool | None = None
    min_seq_len: int = 1
    max_head_dim: int | None = None


_flash_attention_forward_kernel_cache: dict[FlashAttentionKernelKey, FlashAttentionKernel] = {}
_flash_attention_training_forward_kernel_cache: dict[FlashAttentionKernelKey, FlashAttentionTrainingForwardKernel] = {}
_flash_attention_backward_preprocess_kernel_cache: dict[
    FlashAttentionBackwardPreprocessKey, FlashAttentionBackwardPreprocessKernel
] = {}
_flash_attention_backward_kernel_cache: dict[FlashAttentionKernelKey, FlashAttentionBackwardKernel] = {}

_DEFAULT_TILELANG_FLASH_ATTENTION_LAUNCH = (64, 64, 1, 128)
_TILELANG_FLASH_ATTENTION_LAUNCH_RULES: tuple[FlashAttentionLaunchRule, ...] = (
    FlashAttentionLaunchRule(
        config=(64, 128, 2, 128),
        training=True,
        use_dropout=None,
        min_seq_len=128,
        max_head_dim=32,
    ),
)


def clear_flash_attention_kernel_cache() -> None:
    _flash_attention_forward_kernel_cache.clear()
    _flash_attention_training_forward_kernel_cache.clear()
    _flash_attention_backward_preprocess_kernel_cache.clear()
    _flash_attention_backward_kernel_cache.clear()


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
    require_same_dtype("flash_attention", q, k, v)
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
    use_dropout: bool = False,
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
        use_dropout=bool(use_dropout),
    )


def _flash_attention_backward_preprocess_key(x: torch.Tensor) -> FlashAttentionBackwardPreprocessKey:
    return FlashAttentionBackwardPreprocessKey(
        batch=x.shape[0],
        heads=x.shape[1],
        query_len=x.shape[2],
        head_dim=x.shape[3],
        dtype=x.dtype,
    )


def _tilelang_flash_attention_runtime_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float32:
        return torch.bfloat16
    return dtype


def _tilelang_flash_attention_launch_rule_matches(
    rule: FlashAttentionLaunchRule,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    training: bool,
    use_dropout: bool,
) -> bool:
    if rule.training is not None and rule.training != training:
        return False
    if rule.use_dropout is not None and rule.use_dropout != use_dropout:
        return False
    if min(q.shape[2], k.shape[2]) < rule.min_seq_len:
        return False
    if rule.max_head_dim is not None and q.shape[3] > rule.max_head_dim:
        return False
    return True


def _resolve_tilelang_flash_attention_launch_config(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    training: bool,
    use_dropout: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
) -> tuple[int, int, int, int]:
    requested = (block_m, block_n, num_stages, threads)
    if requested != _DEFAULT_TILELANG_FLASH_ATTENTION_LAUNCH:
        return requested
    for rule in _TILELANG_FLASH_ATTENTION_LAUNCH_RULES:
        if _tilelang_flash_attention_launch_rule_matches(
            rule,
            q,
            k,
            training=training,
            use_dropout=use_dropout,
        ):
            return rule.config
    return requested


def _prepare_tilelang_flash_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
    output_dtype = v.dtype
    runtime_dtype = _tilelang_flash_attention_runtime_dtype(q.dtype)
    if runtime_dtype == q.dtype:
        return q, k, v, output_dtype
    return q.to(runtime_dtype), k.to(runtime_dtype), v.to(runtime_dtype), output_dtype


def _empty_dropout_mask(device: torch.device) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.uint8, device=device)


def _build_tilelang_dropout_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    if dropout_p <= 0.0:
        return _empty_dropout_mask(q.device)
    if dropout_p >= 1.0:
        raise RuntimeError("tilelang flash_attention requires dropout_p to be in [0, 1)")
    return (torch.rand((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device) >= dropout_p).to(torch.uint8).contiguous()


def _tilelang_key_lengths_tensor(key_lengths: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if key_lengths.device == device and key_lengths.dtype == torch.int32 and key_lengths.is_contiguous():
        return key_lengths
    return key_lengths.to(device=device, dtype=torch.int32).contiguous()


def _flash_attention_requires_grad(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return q.requires_grad or k.requires_grad or v.requires_grad


def _run_tilelang_flash_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    key_lengths: torch.Tensor,
    is_causal: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if _flash_attention_kernel is not None:
        output = _flash_attention_kernel(q, k, v, key_lengths)
    else:
        kernel = compile_flash_attention_kernel(
            q,
            k,
            v,
            is_causal=is_causal,
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
            threads=threads,
        )
        output = kernel(q, k, v, key_lengths)
    if output.dtype != output_dtype:
        output = output.to(output_dtype)
    return output


def _compile_tilelang_flash_attention_training_forward_kernel(
    key: FlashAttentionKernelKey,
) -> FlashAttentionTrainingForwardKernel:
    if not tilelang_available():
        raise RuntimeError("tilelang is not installed")
    if key in _flash_attention_training_forward_kernel_cache:
        return _flash_attention_training_forward_kernel_cache[key]

    _ensure_tilelang_cuda_fp8_compatibility()
    tl_dtype = tilelang_dtype(key.dtype)
    accum_dtype = T.float32
    compiled = build_flash_attention_training_forward_kernel(
        key.batch,
        key.heads,
        key.query_len,
        key.kv_len,
        key.head_dim,
        is_causal=key.is_causal,
        use_dropout=key.use_dropout,
        block_m=key.block_m,
        block_n=key.block_n,
        num_stages=key.num_stages,
        threads=key.threads,
        tl_dtype=tl_dtype,
        accum_dtype=accum_dtype,
    )

    def runner(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_lengths: torch.Tensor,
        dropout_mask: torch.Tensor | None = None,
        dropout_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_lengths_runtime = _tilelang_key_lengths_tensor(key_lengths, device=q.device)
        if key.use_dropout:
            if dropout_mask is None or dropout_mask.numel() == 0:
                raise RuntimeError("tilelang flash_attention training forward requires a dropout mask")
            return compiled(
                q,
                k,
                v,
                key_lengths_runtime,
                dropout_mask,
                float(dropout_scale),
            )
        return compiled(q, k, v, key_lengths_runtime)

    _flash_attention_training_forward_kernel_cache[key] = runner
    return runner


def _compile_tilelang_flash_attention_backward_preprocess_kernel(
    key: FlashAttentionBackwardPreprocessKey,
) -> FlashAttentionBackwardPreprocessKernel:
    if not tilelang_available():
        raise RuntimeError("tilelang is not installed")
    if key in _flash_attention_backward_preprocess_kernel_cache:
        return _flash_attention_backward_preprocess_kernel_cache[key]

    _ensure_tilelang_cuda_fp8_compatibility()
    tl_dtype = tilelang_dtype(key.dtype)
    accum_dtype = T.float32
    compiled = build_flash_attention_backward_preprocess_kernel(
        key.batch,
        key.heads,
        key.query_len,
        key.head_dim,
        tl_dtype,
        accum_dtype,
    )

    def runner(output: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
        result = compiled(output, grad_out)
        if isinstance(result, tuple):
            return result[0]
        return result

    _flash_attention_backward_preprocess_kernel_cache[key] = runner
    return runner


def _compile_tilelang_flash_attention_backward_kernel(
    key: FlashAttentionKernelKey,
) -> FlashAttentionBackwardKernel:
    if not tilelang_available():
        raise RuntimeError("tilelang is not installed")
    if key in _flash_attention_backward_kernel_cache:
        return _flash_attention_backward_kernel_cache[key]

    _ensure_tilelang_cuda_fp8_compatibility()
    tl_dtype = tilelang_dtype(key.dtype)
    accum_dtype = T.float32
    compiled = build_flash_attention_backward_kernel(
        key.batch,
        key.heads,
        key.query_len,
        key.kv_len,
        key.head_dim,
        is_causal=key.is_causal,
        use_dropout=key.use_dropout,
        block_m=key.block_m,
        block_n=key.block_n,
        num_stages=key.num_stages,
        threads=key.threads,
        tl_dtype=tl_dtype,
        accum_dtype=accum_dtype,
    )

    def runner(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        grad_out: torch.Tensor,
        lse: torch.Tensor,
        delta: torch.Tensor,
        key_lengths: torch.Tensor,
        dropout_mask: torch.Tensor | None = None,
        dropout_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key_lengths_runtime = _tilelang_key_lengths_tensor(key_lengths, device=q.device)
        grad_q = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        grad_k = torch.empty(k.shape, dtype=torch.float32, device=k.device)
        grad_v = torch.empty(v.shape, dtype=torch.float32, device=v.device)
        if key.use_dropout:
            if dropout_mask is None or dropout_mask.numel() == 0:
                raise RuntimeError("tilelang flash_attention backward requires a dropout mask")
            compiled(
                q,
                k,
                v,
                grad_out,
                lse,
                delta,
                key_lengths_runtime,
                dropout_mask,
                float(dropout_scale),
                grad_q,
                grad_k,
                grad_v,
            )
        else:
            compiled(
                q,
                k,
                v,
                grad_out,
                lse,
                delta,
                key_lengths_runtime,
                grad_q,
                grad_k,
                grad_v,
            )
        return grad_q, grad_k, grad_v

    _flash_attention_backward_kernel_cache[key] = runner
    return runner


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
    if dropout_p < 0.0 or dropout_p >= 1.0:
        raise RuntimeError("tilelang flash_attention requires dropout_p to be in [0, 1)")
    if dropout_p != 0.0 and not training:
        raise RuntimeError("tilelang flash_attention only supports dropout during training")
    if not tilelang_available():
        raise RuntimeError("tilelang backend requested but tilelang is not installed")
    require_cuda_tensors("tilelang flash_attention", q)
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


class _TilelangFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_lengths: torch.Tensor,
        dropout_mask: torch.Tensor,
        dropout_p: float,
        is_causal: bool,
        block_m: int,
        block_n: int,
        num_stages: int,
        threads: int,
    ) -> torch.Tensor:
        tilelang_q, tilelang_k, tilelang_v, output_dtype = _prepare_tilelang_flash_attention_inputs(q, k, v)
        key = _flash_attention_cache_key(
            tilelang_q,
            tilelang_k,
            is_causal=is_causal,
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
            threads=threads,
            use_dropout=dropout_mask.numel() > 0,
        )
        forward_kernel = _compile_tilelang_flash_attention_training_forward_kernel(key)
        output_runtime, lse = forward_kernel(
            tilelang_q,
            tilelang_k,
            tilelang_v,
            key_lengths,
            dropout_mask if dropout_mask.numel() > 0 else None,
            1.0 / (1.0 - dropout_p) if dropout_p > 0.0 else 1.0,
        )
        ctx.save_for_backward(tilelang_q, tilelang_k, tilelang_v, output_runtime, lse, key_lengths, dropout_mask)
        ctx.key = key
        ctx.q_dtype = q.dtype
        ctx.k_dtype = k.dtype
        ctx.v_dtype = v.dtype
        ctx.output_dtype = output_dtype
        ctx.dropout_scale = 1.0 / (1.0 - dropout_p) if dropout_p > 0.0 else 1.0
        if output_runtime.dtype != output_dtype:
            return output_runtime.to(output_dtype)
        return output_runtime

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not any(ctx.needs_input_grad[:3]):
            return None, None, None, None, None, None, None, None, None, None, None

        tilelang_q, tilelang_k, tilelang_v, output_runtime, lse, key_lengths, dropout_mask = ctx.saved_tensors
        grad_out_runtime = grad_out.contiguous().to(device=output_runtime.device, dtype=output_runtime.dtype)
        preprocess_kernel = _compile_tilelang_flash_attention_backward_preprocess_kernel(
            _flash_attention_backward_preprocess_key(output_runtime)
        )
        delta = preprocess_kernel(output_runtime, grad_out_runtime)
        backward_kernel = _compile_tilelang_flash_attention_backward_kernel(ctx.key)
        grad_q_runtime, grad_k_runtime, grad_v_runtime = backward_kernel(
            tilelang_q,
            tilelang_k,
            tilelang_v,
            grad_out_runtime,
            lse,
            delta,
            key_lengths,
            dropout_mask if dropout_mask.numel() > 0 else None,
            ctx.dropout_scale,
        )
        grad_q = grad_q_runtime.to(ctx.q_dtype) if ctx.needs_input_grad[0] else None
        grad_k = grad_k_runtime.to(ctx.k_dtype) if ctx.needs_input_grad[1] else None
        grad_v = grad_v_runtime.to(ctx.v_dtype) if ctx.needs_input_grad[2] else None
        return grad_q, grad_k, grad_v, None, None, None, None, None, None, None, None


def _compile_tilelang_flash_attention_forward_kernel(key: FlashAttentionKernelKey) -> FlashAttentionKernel:
    if not tilelang_available():
        raise RuntimeError("tilelang is not installed")
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


def _run_tilelang_flash_attention_training_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    key_lengths: torch.Tensor,
    dropout_mask: torch.Tensor,
    dropout_p: float,
    is_causal: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    tilelang_q, tilelang_k, tilelang_v, output_dtype = _prepare_tilelang_flash_attention_inputs(q, k, v)
    key = _flash_attention_cache_key(
        tilelang_q,
        tilelang_k,
        is_causal=is_causal,
        block_m=block_m,
        block_n=block_n,
        num_stages=num_stages,
        threads=threads,
        use_dropout=dropout_mask.numel() > 0,
    )
    forward_kernel = _compile_tilelang_flash_attention_training_forward_kernel(key)
    output_runtime, lse = forward_kernel(
        tilelang_q,
        tilelang_k,
        tilelang_v,
        key_lengths,
        dropout_mask if dropout_mask.numel() > 0 else None,
        1.0 / (1.0 - dropout_p) if dropout_p > 0.0 else 1.0,
    )
    return output_runtime, lse, output_dtype


def _validate_tilelang_training_mask_plan(mask_plan: FlashAttentionMaskPlan) -> None:
    if mask_plan.query_self_mask is not None:
        raise RuntimeError("tilelang flash_attention training currently requires prefix key masks without query self fallback")


def _run_tilelang_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    training: bool,
    is_causal: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
) -> torch.Tensor:
    effective_block_m, effective_block_n, effective_num_stages, effective_threads = (
        _resolve_tilelang_flash_attention_launch_config(
            q,
            k,
            training=training or _flash_attention_requires_grad(q, k, v),
            use_dropout=training and dropout_p > 0.0,
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
            threads=threads,
        )
    )
    mask_plan = _plan_tilelang_flash_attention_mask(
        q,
        k,
        attn_mask,
        is_causal=is_causal,
    )
    key_lengths = mask_plan.key_lengths.to(device=q.device, dtype=torch.int32).contiguous()
    if training or _flash_attention_requires_grad(q, k, v):
        _validate_tilelang_training_mask_plan(mask_plan)
        dropout_mask = _build_tilelang_dropout_mask(q, k, dropout_p) if training and dropout_p > 0.0 else _empty_dropout_mask(q.device)
        if _flash_attention_requires_grad(q, k, v):
            if mask_plan.query_self_mask is not None:
                raise RuntimeError("tilelang flash_attention training autograd does not support query self fallback masks")
            return _TilelangFlashAttentionFunction.apply(
                q,
                k,
                v,
                key_lengths,
                dropout_mask,
                dropout_p,
                mask_plan.is_causal,
                effective_block_m,
                effective_block_n,
                effective_num_stages,
                effective_threads,
            )
        output_runtime, _lse, output_dtype = _run_tilelang_flash_attention_training_forward(
            q,
            k,
            v,
            key_lengths=key_lengths,
            dropout_mask=dropout_mask,
            dropout_p=dropout_p,
            is_causal=mask_plan.is_causal,
            block_m=effective_block_m,
            block_n=effective_block_n,
            num_stages=effective_num_stages,
            threads=effective_threads,
        )
        output = output_runtime.to(output_dtype) if output_runtime.dtype != output_dtype else output_runtime
        return _apply_query_self_mask_fallback(output, v, mask_plan.query_self_mask)

    if dropout_p != 0.0:
        raise RuntimeError("tilelang flash_attention only supports dropout during training")
    tilelang_q, tilelang_k, tilelang_v, output_dtype = _prepare_tilelang_flash_attention_inputs(q, k, v)
    output = _run_tilelang_flash_attention_kernel(
        tilelang_q,
        tilelang_k,
        tilelang_v,
        key_lengths=key_lengths,
        is_causal=mask_plan.is_causal,
        block_m=effective_block_m,
        block_n=effective_block_n,
        num_stages=effective_num_stages,
        threads=effective_threads,
        output_dtype=output_dtype,
    )
    return _apply_query_self_mask_fallback(output, v, mask_plan.query_self_mask)


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
        mask_plan = _plan_tilelang_flash_attention_mask(
            normalized_q,
            normalized_k,
            attn_mask,
            is_causal=is_causal,
        )
        if training:
            _validate_tilelang_training_mask_plan(mask_plan)
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
    return _run_tilelang_flash_attention(
        normalized_q,
        normalized_k,
        normalized_v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        training=training,
        is_causal=is_causal,
        block_m=block_m,
        block_n=block_n,
        num_stages=num_stages,
        threads=threads,
    )


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

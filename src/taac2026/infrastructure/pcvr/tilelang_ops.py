"""TileLang-backed operator boundary for PCVR kernels.

This module intentionally exposes only operator-level hooks. Model-layer
composition stays in experiment packages, while src owns optional accelerated
kernel entry points with torch fallbacks.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

try:
    import tilelang as tl  # type: ignore[import-not-found]
    import tilelang.language as T  # type: ignore[import-not-found]
    from taac2026.infrastructure.pcvr.tilelang_kernels import (
        build_flash_attention_forward_kernel,
        build_rms_norm_backward_kernel,
        build_rms_norm_forward_kernel,
    )
except ImportError:
    tl = None
    T = None
    build_flash_attention_forward_kernel = None
    build_rms_norm_backward_kernel = None
    build_rms_norm_forward_kernel = None


EmbeddingBagMeanKernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
FlashAttentionKernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
RMSNormKernel = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
FlashAttentionBackend = Literal["torch", "tilelang"]
RMSNormBackend = Literal["torch", "tilelang"]

_embedding_bag_mean_kernel: EmbeddingBagMeanKernel | None = None
_flash_attention_kernel: FlashAttentionKernel | None = None
_rms_norm_kernel: RMSNormKernel | None = None


@dataclass(frozen=True, slots=True)
class RMSNormKernelKey:
    rows: int
    cols: int
    dtype: torch.dtype
    eps: float
    block_rows: int


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


_rms_norm_forward_kernel_cache: dict[RMSNormKernelKey, Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = {}
_rms_norm_backward_kernel_cache: dict[
    RMSNormKernelKey, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
] = {}
_flash_attention_forward_kernel_cache: dict[FlashAttentionKernelKey, FlashAttentionKernel] = {}

_TILELANG_E8M0_ORIGINAL_GUARD = """// __nv_fp8_e8m0 is only available in CUDA 12.6+
#if __CUDACC_VER_MAJOR__ > 12 ||                                               \
        (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 6)
using fp8_e8_t = __nv_fp8_e8m0;
#define TL_HAS_FP8_E8M0 1
#else
// Placeholder for CUDA < 12.6
struct fp8_e8_t {
    unsigned char data;
};
#define TL_HAS_FP8_E8M0 0
#endif
"""

_TILELANG_E8M0_COMPAT_GUARD = """// Some CUDA 12.6 toolchains still ship without e8m0 symbols in cuda_fp8.h.
// Keep e8m0 disabled unless the toolchain is known to provide those APIs.
#if 0
using fp8_e8_t = __nv_fp8_e8m0;
#define TL_HAS_FP8_E8M0 1
#else
// Placeholder for CUDA < 12.6
struct fp8_e8_t {
    unsigned char data;
};
#define TL_HAS_FP8_E8M0 0
#endif
"""

_TILELANG_E8M0_GUARD_START = "// __nv_fp8_e8m0 is only available in CUDA 12.6+"


def _tilelang_e8m0_guard_bounds(content: str) -> tuple[int, int] | None:
    start = content.find(_TILELANG_E8M0_GUARD_START)
    if start < 0:
        return None
    end = content.find("#endif", start)
    if end < 0:
        return None
    end += len("#endif")
    block = content[start:end]
    if "using fp8_e8_t = __nv_fp8_e8m0;" not in block:
        return None
    if "#define TL_HAS_FP8_E8M0 0" not in block:
        return None
    if end < len(content) and content[end] == "\n":
        end += 1
    return start, end


def tilelang_available() -> bool:
    return tl is not None and T is not None


def _tilelang_cuda_fp8_header_path() -> Path | None:
    if tl is None:
        return None
    package_root = Path(tl.__file__).resolve().parent
    header_path = package_root / "src" / "tl_templates" / "cuda" / "cuda_fp8.h"
    return header_path if header_path.is_file() else None


def _cuda_fp8_header_candidates() -> tuple[Path, ...]:
    include_dirs: list[Path] = []
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            include_dirs.append(Path(env_value) / "include")
    include_dirs.append(Path("/usr/local/cuda/include"))
    include_dirs.extend(sorted(Path("/usr/local").glob("cuda-*/include")))

    headers: list[Path] = []
    for include_dir in include_dirs:
        headers.append(include_dir / "cuda_fp8.h")
        headers.append(include_dir / "cuda_fp8.hpp")
    return tuple(dict.fromkeys(headers))


def _cuda_fp8_supports_e8m0(header_paths: Sequence[Path] | None = None) -> bool:
    candidates = tuple(header_paths) if header_paths is not None else _cuda_fp8_header_candidates()
    for header_path in candidates:
        if not header_path.is_file():
            continue
        try:
            content = header_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "__nv_fp8_e8m0" in content:
            return True
    return False


def _ensure_tilelang_cuda_fp8_compatibility(
    *,
    tilelang_header: Path | None = None,
    cuda_header_paths: Sequence[Path] | None = None,
) -> bool:
    header_path = tilelang_header or _tilelang_cuda_fp8_header_path()
    if header_path is None or _cuda_fp8_supports_e8m0(cuda_header_paths):
        return False
    try:
        content = header_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"failed to read tilelang fp8 header: {header_path}") from error
    if _TILELANG_E8M0_COMPAT_GUARD in content:
        return False
    guard_bounds = _tilelang_e8m0_guard_bounds(content)
    if guard_bounds is None:
        return False
    start, end = guard_bounds
    try:
        header_path.write_text(content[:start] + _TILELANG_E8M0_COMPAT_GUARD + content[end:], encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"failed to patch tilelang fp8 header: {header_path}") from error
    return True


def clear_tilelang_kernel_cache() -> None:
    _flash_attention_forward_kernel_cache.clear()
    _rms_norm_forward_kernel_cache.clear()
    _rms_norm_backward_kernel_cache.clear()


def register_embedding_bag_mean_kernel(kernel: EmbeddingBagMeanKernel) -> None:
    global _embedding_bag_mean_kernel
    _embedding_bag_mean_kernel = kernel


def register_flash_attention_kernel(kernel: FlashAttentionKernel) -> None:
    global _flash_attention_kernel
    _flash_attention_kernel = kernel


def register_rms_norm_kernel(kernel: RMSNormKernel) -> None:
    global _rms_norm_kernel
    _rms_norm_kernel = kernel


def embedding_bag_mean(embedding_weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if _embedding_bag_mean_kernel is not None:
        return _embedding_bag_mean_kernel(embedding_weight, values)
    embedded = F.embedding(values, embedding_weight, padding_idx=0)
    valid = values.ne(0).to(embedded.dtype).unsqueeze(-1)
    return (embedded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


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
    return "tilelang"


def _rms_norm_registered_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if _rms_norm_kernel is None:
        raise RuntimeError("no registered rms_norm kernel is available")
    return _rms_norm_kernel(x, weight, eps)


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
    tl_dtype = _tilelang_dtype(key.dtype)
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


def _tilelang_dtype(dtype: torch.dtype):
    if T is None:
        raise RuntimeError("tilelang language module is unavailable")
    if dtype == torch.float16:
        return T.float16
    if dtype == torch.bfloat16:
        return T.bfloat16
    if dtype == torch.float32:
        return T.float32
    raise RuntimeError(f"tilelang rms_norm does not support dtype {dtype}")


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
    if build_rms_norm_forward_kernel is None:
        raise RuntimeError("tilelang kernel builders are unavailable")
    if key in _rms_norm_forward_kernel_cache:
        return _rms_norm_forward_kernel_cache[key]

    _ensure_tilelang_cuda_fp8_compatibility()
    tl_dtype = _tilelang_dtype(key.dtype)
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
    if build_rms_norm_backward_kernel is None:
        raise RuntimeError("tilelang backward kernel builders are unavailable")
    if key in _rms_norm_backward_kernel_cache:
        return _rms_norm_backward_kernel_cache[key]

    _ensure_tilelang_cuda_fp8_compatibility()
    tl_dtype = _tilelang_dtype(key.dtype)
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
    "clear_tilelang_kernel_cache",
    "compile_flash_attention_kernel",
    "compile_rms_norm_kernel",
    "embedding_bag_mean",
    "flash_attention",
    "register_embedding_bag_mean_kernel",
    "register_flash_attention_kernel",
    "register_rms_norm_kernel",
    "resolved_flash_attention_backend",
    "resolved_rms_norm_backend",
    "rms_norm",
    "tilelang_available",
]
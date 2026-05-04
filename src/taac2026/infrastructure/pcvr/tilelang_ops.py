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
        build_rms_norm_backward_kernel,
        build_rms_norm_forward_kernel,
    )
except ImportError:
    tl = None
    T = None
    build_rms_norm_backward_kernel = None
    build_rms_norm_forward_kernel = None


EmbeddingBagMeanKernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
RMSNormKernel = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
RMSNormBackend = Literal["torch", "tilelang"]

_embedding_bag_mean_kernel: EmbeddingBagMeanKernel | None = None
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
    _rms_norm_forward_kernel_cache.clear()
    _rms_norm_backward_kernel_cache.clear()


def register_embedding_bag_mean_kernel(kernel: EmbeddingBagMeanKernel) -> None:
    global _embedding_bag_mean_kernel
    _embedding_bag_mean_kernel = kernel


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
    matrix, normalized_weight, _original_shape = _normalize_rms_norm_inputs(x, weight)
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
    "compile_rms_norm_kernel",
    "embedding_bag_mean",
    "register_embedding_bag_mean_kernel",
    "register_rms_norm_kernel",
    "resolved_rms_norm_backend",
    "rms_norm",
    "tilelang_available",
]
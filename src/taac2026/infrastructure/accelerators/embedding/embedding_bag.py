"""Embedding fused operator boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from taac2026.infrastructure.accelerators.embedding.kernels.tilelang import (
	build_embedding_bag_mean_backward_kernel,
	build_embedding_bag_mean_forward_kernel,
)
from taac2026.infrastructure.accelerators.embedding.kernels.triton import (
	build_embedding_bag_mean_backward_kernel as build_triton_embedding_bag_mean_backward_kernel,
	build_embedding_bag_mean_forward_kernel as build_triton_embedding_bag_mean_forward_kernel,
)
from taac2026.infrastructure.accelerators.embedding.cuembed_runtime import (
	clear_cuembed_embedding_bag_mean_kernel_cache,
	compile_cuembed_embedding_bag_mean_kernel,
	cuembed_available,
)
from taac2026.infrastructure.accelerators.tilelang_runtime import (
	T,
	_ensure_tilelang_cuda_fp8_compatibility,
	tilelang_available,
	tilelang_dtype,
)
from taac2026.infrastructure.accelerators.triton_runtime import (
	triton_available,
	triton_supported_floating_dtype,
)
from taac2026.infrastructure.accelerators.tensor_validation import require_cuda_tensors, require_same_device


EmbeddingBagMeanKernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
EmbeddingBagMeanBackend = Literal["torch", "tilelang", "triton", "cuembed"]
_TILELANG_FORWARD_BLOCK_ROWS = 8

_embedding_bag_mean_kernel: EmbeddingBagMeanKernel | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingBagMeanKernelKey:
	batch: int
	bag_size: int
	num_embeddings: int
	emb_dim: int
	dtype: torch.dtype
	block_rows: int
	block_cols: int


_embedding_bag_mean_forward_kernel_cache: dict[
	EmbeddingBagMeanKernelKey, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
] = {}
_embedding_bag_mean_backward_kernel_cache: dict[
	EmbeddingBagMeanKernelKey, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
] = {}
_triton_embedding_bag_mean_forward_kernel_cache: dict[
	EmbeddingBagMeanKernelKey, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
] = {}
_triton_embedding_bag_mean_backward_kernel_cache: dict[
	EmbeddingBagMeanKernelKey, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
] = {}


def clear_embedding_bag_mean_kernel_cache() -> None:
	_embedding_bag_mean_forward_kernel_cache.clear()
	_embedding_bag_mean_backward_kernel_cache.clear()
	_triton_embedding_bag_mean_forward_kernel_cache.clear()
	_triton_embedding_bag_mean_backward_kernel_cache.clear()
	clear_cuembed_embedding_bag_mean_kernel_cache()


def register_embedding_bag_mean_kernel(kernel: EmbeddingBagMeanKernel) -> None:
	global _embedding_bag_mean_kernel
	_embedding_bag_mean_kernel = kernel


def _torch_embedding_bag_mean(embedding_weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
	return F.embedding_bag(values, embedding_weight, mode="mean", padding_idx=0)


def _normalize_embedding_bag_mean_inputs(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
	if embedding_weight.ndim != 2:
		raise ValueError("embedding_bag_mean weight must be a 2D tensor")
	if values.ndim != 2:
		raise ValueError("embedding_bag_mean values must be a 2D tensor")
	require_same_device("embedding_bag_mean", embedding_weight, values)
	original_shape = values.shape
	normalized_weight = embedding_weight.contiguous()
	normalized_values = values.to(device=normalized_weight.device).contiguous()
	return normalized_weight, normalized_values, original_shape


def _embedding_bag_mean_cache_key(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
	block_rows: int | None,
	block_cols: int | None,
) -> EmbeddingBagMeanKernelKey:
	return EmbeddingBagMeanKernelKey(
		batch=values.shape[0],
		bag_size=values.shape[1],
		num_embeddings=embedding_weight.shape[0],
		emb_dim=embedding_weight.shape[1],
		dtype=embedding_weight.dtype,
		block_rows=max(1, int(block_rows or _TILELANG_FORWARD_BLOCK_ROWS)),
		block_cols=max(1, int(block_cols or min(64, embedding_weight.shape[1]))),
	)


def _resolve_embedding_bag_mean_backend(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
	backend: EmbeddingBagMeanBackend,
) -> Literal["torch", "tilelang", "triton", "cuembed"]:
	if backend == "torch":
		return "torch"
	if backend == "cuembed":
		if not cuembed_available():
			raise RuntimeError("cuembed backend requested but CUDA runtime or CUDA toolkit is not available")
		require_cuda_tensors("cuembed embedding_bag_mean", embedding_weight)
		if embedding_weight.dtype not in {torch.float16, torch.float32}:
			raise RuntimeError(f"cuembed embedding_bag_mean does not support dtype {embedding_weight.dtype}")
		if embedding_weight.requires_grad:
			raise RuntimeError("cuembed embedding_bag_mean is currently forward-only; use torch or tilelang for training")
		if values.dtype not in {torch.int32, torch.int64}:
			raise RuntimeError(f"cuembed embedding_bag_mean requires int32 or int64 values, got {values.dtype}")
		return "cuembed"
	if backend == "triton":
		if not triton_available():
			raise RuntimeError("triton backend requested but triton is not installed")
		require_cuda_tensors("triton embedding_bag_mean", embedding_weight)
		if not triton_supported_floating_dtype(embedding_weight.dtype):
			raise RuntimeError(f"triton embedding_bag_mean does not support dtype {embedding_weight.dtype}")
		if values.dtype not in {torch.int32, torch.int64}:
			raise RuntimeError(f"triton embedding_bag_mean requires int32 or int64 values, got {values.dtype}")
		return "triton"
	if backend != "tilelang":
		raise ValueError(f"unsupported embedding_bag_mean backend: {backend}")
	if not tilelang_available():
		raise RuntimeError("tilelang backend requested but tilelang is not installed")
	require_cuda_tensors("tilelang embedding_bag_mean", embedding_weight)
	if embedding_weight.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
		raise RuntimeError(f"tilelang embedding_bag_mean does not support dtype {embedding_weight.dtype}")
	return "tilelang"


def resolved_embedding_bag_mean_backend(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
	backend: EmbeddingBagMeanBackend = "torch",
	*,
	block_rows: int | None = None,
	block_cols: int | None = None,
) -> Literal["torch", "tilelang", "triton", "cuembed"]:
	del block_rows, block_cols
	normalized_weight, normalized_values, _original_shape = _normalize_embedding_bag_mean_inputs(embedding_weight, values)
	return _resolve_embedding_bag_mean_backend(normalized_weight, normalized_values, backend)


def _tilelang_embedding_bag_values(values: torch.Tensor) -> torch.Tensor:
	if values.dtype not in {torch.int32, torch.int64}:
		raise RuntimeError(f"tilelang embedding_bag_mean requires int32 or int64 values, got {values.dtype}")
	return values.to(dtype=torch.int32).contiguous()


def _triton_embedding_bag_values(values: torch.Tensor) -> torch.Tensor:
	if values.dtype not in {torch.int32, torch.int64}:
		raise RuntimeError(f"triton embedding_bag_mean requires int32 or int64 values, got {values.dtype}")
	return values.to(dtype=torch.int64).contiguous()


def _run_torch_embedding_bag_mean(embedding_weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
	if _embedding_bag_mean_kernel is not None:
		return _embedding_bag_mean_kernel(embedding_weight, values)
	return _torch_embedding_bag_mean(embedding_weight, values)


def _compile_tilelang_embedding_bag_mean_forward_kernel(
	key: EmbeddingBagMeanKernelKey,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
	if not tilelang_available():
		raise RuntimeError("tilelang is not installed")
	if key in _embedding_bag_mean_forward_kernel_cache:
		return _embedding_bag_mean_forward_kernel_cache[key]

	_ensure_tilelang_cuda_fp8_compatibility()
	tl_dtype = tilelang_dtype(key.dtype)
	accum_dtype = T.float32
	compiled = build_embedding_bag_mean_forward_kernel(
		key.batch,
		key.bag_size,
		key.num_embeddings,
		key.emb_dim,
		block_rows=key.block_rows,
		block_cols=key.block_cols,
		tl_dtype=tl_dtype,
		accum_dtype=accum_dtype,
	)

	def runner(embedding_weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
		require_same_device("tilelang embedding_bag_mean", embedding_weight, values)
		result = compiled(embedding_weight, _tilelang_embedding_bag_values(values))
		if isinstance(result, tuple):
			return result[0]
		return result

	_embedding_bag_mean_forward_kernel_cache[key] = runner
	return runner


def _compile_tilelang_embedding_bag_mean_backward_kernel(
	key: EmbeddingBagMeanKernelKey,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
	if not tilelang_available():
		raise RuntimeError("tilelang is not installed")
	if key in _embedding_bag_mean_backward_kernel_cache:
		return _embedding_bag_mean_backward_kernel_cache[key]

	_ensure_tilelang_cuda_fp8_compatibility()
	tl_dtype = tilelang_dtype(key.dtype)
	accum_dtype = T.float32
	compiled = build_embedding_bag_mean_backward_kernel(
		key.batch,
		key.bag_size,
		key.num_embeddings,
		key.emb_dim,
		block_rows=key.block_rows,
		block_cols=key.block_cols,
		tl_dtype=tl_dtype,
		accum_dtype=accum_dtype,
	)

	def runner(values: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
		grad_weight = torch.zeros(
			(key.num_embeddings, key.emb_dim),
			dtype=torch.float32,
			device=grad_out.device,
		)
		compiled(values, grad_out, grad_weight)
		return grad_weight

	_embedding_bag_mean_backward_kernel_cache[key] = runner
	return runner


def _compile_triton_embedding_bag_mean_forward_kernel(
	key: EmbeddingBagMeanKernelKey,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
	if not triton_available():
		raise RuntimeError("triton is not installed")
	if key in _triton_embedding_bag_mean_forward_kernel_cache:
		return _triton_embedding_bag_mean_forward_kernel_cache[key]

	compiled = build_triton_embedding_bag_mean_forward_kernel(
		key.batch,
		key.bag_size,
		key.num_embeddings,
		key.emb_dim,
		block_rows=key.block_rows,
		block_cols=key.block_cols,
	)

	def runner(embedding_weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
		require_same_device("triton embedding_bag_mean", embedding_weight, values)
		return compiled(embedding_weight, _triton_embedding_bag_values(values))

	_triton_embedding_bag_mean_forward_kernel_cache[key] = runner
	return runner


def _compile_triton_embedding_bag_mean_backward_kernel(
	key: EmbeddingBagMeanKernelKey,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
	if not triton_available():
		raise RuntimeError("triton is not installed")
	if key in _triton_embedding_bag_mean_backward_kernel_cache:
		return _triton_embedding_bag_mean_backward_kernel_cache[key]

	compiled = build_triton_embedding_bag_mean_backward_kernel(
		key.batch,
		key.bag_size,
		key.num_embeddings,
		key.emb_dim,
		block_rows=key.block_rows,
		block_cols=key.block_cols,
	)

	def runner(values: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
		return compiled(_triton_embedding_bag_values(values), grad_out)

	_triton_embedding_bag_mean_backward_kernel_cache[key] = runner
	return runner


def compile_embedding_bag_mean_kernel(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
	*,
	block_rows: int | None = None,
	block_cols: int | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
	normalized_weight, normalized_values, _original_shape = _normalize_embedding_bag_mean_inputs(embedding_weight, values)
	key = _embedding_bag_mean_cache_key(normalized_weight, normalized_values, block_rows, block_cols)
	return _compile_tilelang_embedding_bag_mean_forward_kernel(key)


def compile_triton_embedding_bag_mean_kernel(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
	*,
	block_rows: int | None = None,
	block_cols: int | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
	normalized_weight, normalized_values, _original_shape = _normalize_embedding_bag_mean_inputs(embedding_weight, values)
	key = _embedding_bag_mean_cache_key(normalized_weight, normalized_values, block_rows, block_cols)
	return _compile_triton_embedding_bag_mean_forward_kernel(key)


class _TilelangEmbeddingBagMeanFunction(torch.autograd.Function):
	@staticmethod
	def forward(
		ctx,
		embedding_weight: torch.Tensor,
		values: torch.Tensor,
		block_rows: int,
		block_cols: int,
	) -> torch.Tensor:
		key = _embedding_bag_mean_cache_key(embedding_weight, values, block_rows, block_cols)
		forward_kernel = _compile_tilelang_embedding_bag_mean_forward_kernel(key)
		out = forward_kernel(embedding_weight, values)
		ctx.save_for_backward(values)
		ctx.key = key
		ctx.weight_dtype = embedding_weight.dtype
		return out

	@staticmethod
	def backward(ctx, grad_out: torch.Tensor):
		if not ctx.needs_input_grad[0]:
			return None, None, None, None
		(values,) = ctx.saved_tensors
		backward_kernel = _compile_tilelang_embedding_bag_mean_backward_kernel(ctx.key)
		grad_weight = backward_kernel(values, grad_out.contiguous().to(dtype=ctx.key.dtype))
		return grad_weight.to(ctx.weight_dtype), None, None, None


class _TritonEmbeddingBagMeanFunction(torch.autograd.Function):
	@staticmethod
	def forward(
		ctx,
		embedding_weight: torch.Tensor,
		values: torch.Tensor,
		block_rows: int,
		block_cols: int,
	) -> torch.Tensor:
		key = _embedding_bag_mean_cache_key(embedding_weight, values, block_rows, block_cols)
		forward_kernel = _compile_triton_embedding_bag_mean_forward_kernel(key)
		out = forward_kernel(embedding_weight, values)
		ctx.save_for_backward(values)
		ctx.key = key
		ctx.weight_dtype = embedding_weight.dtype
		return out

	@staticmethod
	def backward(ctx, grad_out: torch.Tensor):
		if not ctx.needs_input_grad[0]:
			return None, None, None, None
		(values,) = ctx.saved_tensors
		backward_kernel = _compile_triton_embedding_bag_mean_backward_kernel(ctx.key)
		grad_weight = backward_kernel(values, grad_out.contiguous().to(dtype=ctx.key.dtype))
		return grad_weight.to(ctx.weight_dtype), None, None, None


def _run_tilelang_embedding_bag_mean(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
	*,
	block_rows: int | None,
	block_cols: int | None,
) -> torch.Tensor:
	default_block_rows = 1 if embedding_weight.requires_grad else _TILELANG_FORWARD_BLOCK_ROWS
	resolved_block_rows = max(1, int(block_rows or default_block_rows))
	resolved_block_cols = max(1, int(block_cols or min(64, embedding_weight.shape[1])))
	if embedding_weight.requires_grad:
		return _TilelangEmbeddingBagMeanFunction.apply(
			embedding_weight,
			values,
			resolved_block_rows,
			resolved_block_cols,
		)
	kernel = compile_embedding_bag_mean_kernel(
		embedding_weight,
		values,
		block_rows=resolved_block_rows,
		block_cols=resolved_block_cols,
	)
	return kernel(embedding_weight, values)


def _run_triton_embedding_bag_mean(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
	*,
	block_rows: int | None,
	block_cols: int | None,
) -> torch.Tensor:
	default_block_rows = 1 if embedding_weight.requires_grad else _TILELANG_FORWARD_BLOCK_ROWS
	resolved_block_rows = max(1, int(block_rows or default_block_rows))
	resolved_block_cols = max(1, int(block_cols or min(64, embedding_weight.shape[1])))
	if embedding_weight.requires_grad:
		return _TritonEmbeddingBagMeanFunction.apply(
			embedding_weight,
			values,
			resolved_block_rows,
			resolved_block_cols,
		)
	kernel = compile_triton_embedding_bag_mean_kernel(
		embedding_weight,
		values,
		block_rows=resolved_block_rows,
		block_cols=resolved_block_cols,
	)
	return kernel(embedding_weight, values)


def _run_cuembed_embedding_bag_mean(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
) -> torch.Tensor:
	kernel = compile_cuembed_embedding_bag_mean_kernel(embedding_weight, values)
	return kernel(embedding_weight, values)


def embedding_bag_mean(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
	*,
	backend: EmbeddingBagMeanBackend = "torch",
	block_rows: int | None = None,
	block_cols: int | None = None,
) -> torch.Tensor:
	normalized_weight, normalized_values, _original_shape = _normalize_embedding_bag_mean_inputs(embedding_weight, values)
	resolved_backend = _resolve_embedding_bag_mean_backend(normalized_weight, normalized_values, backend)
	if resolved_backend == "torch":
		return _run_torch_embedding_bag_mean(normalized_weight, normalized_values)
	if resolved_backend == "cuembed":
		return _run_cuembed_embedding_bag_mean(normalized_weight, normalized_values)
	if resolved_backend == "triton":
		triton_values = _triton_embedding_bag_values(normalized_values)
		return _run_triton_embedding_bag_mean(
			normalized_weight,
			triton_values,
			block_rows=block_rows,
			block_cols=block_cols,
		)
	tilelang_values = _tilelang_embedding_bag_values(normalized_values)
	return _run_tilelang_embedding_bag_mean(
		normalized_weight,
		tilelang_values,
		block_rows=block_rows,
		block_cols=block_cols,
	)


__all__ = [
	"EmbeddingBagMeanBackend",
	"EmbeddingBagMeanKernel",
	"EmbeddingBagMeanKernelKey",
	"clear_embedding_bag_mean_kernel_cache",
	"compile_embedding_bag_mean_kernel",
	"compile_triton_embedding_bag_mean_kernel",
	"embedding_bag_mean",
	"register_embedding_bag_mean_kernel",
	"resolved_embedding_bag_mean_backend",
]

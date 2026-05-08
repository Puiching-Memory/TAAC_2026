"""Runtime loader for cuEmbed-style fixed-hotness embedding lookup.

The CUDA source loaded by this module is a narrow, project-local rewrite of the
fixed-hotness embedding forward pattern used by NVIDIA cuEmbed. It keeps the
same important execution boundary for TAAC: runtime JIT compilation through
``torch.utils.cpp_extension`` and explicit opt-in from the embedding backend.

Source reference: https://github.com/NVIDIA/cuEmbed at commit
3bb39fd4ccaca831cf55d9ff4fea2998dc65359f. cuEmbed is Apache-2.0 licensed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Lock

import torch

from taac2026.infrastructure.accelerators.tensor_validation import require_cuda_tensors, require_same_device


CuEmbedEmbeddingBagMeanKernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_CUEMBED_EXTENSION = None
_CUEMBED_EXTENSION_LOCK = Lock()
_CUEMBED_SOURCE_PATH = Path(__file__).resolve().parent / "kernels" / "cuembed_embedding_bag_mean.cu"


def clear_cuembed_embedding_bag_mean_kernel_cache() -> None:
	global _CUEMBED_EXTENSION
	_CUEMBED_EXTENSION = None


def cuembed_available() -> bool:
	if not torch.cuda.is_available():
		return False
	try:
		from torch.utils.cpp_extension import CUDA_HOME
	except Exception:  # pragma: no cover - environment dependent
		return False
	return CUDA_HOME is not None


def _load_cuembed_extension():
	global _CUEMBED_EXTENSION
	if _CUEMBED_EXTENSION is not None:
		return _CUEMBED_EXTENSION
	with _CUEMBED_EXTENSION_LOCK:
		if _CUEMBED_EXTENSION is not None:
			return _CUEMBED_EXTENSION
		if not torch.cuda.is_available():
			raise RuntimeError("cuembed backend requested but CUDA is not available")
		try:
			from torch.utils.cpp_extension import CUDA_HOME, load
		except Exception as error:  # pragma: no cover - environment dependent
			raise RuntimeError("cuembed backend requires torch C++ extension support") from error
		if CUDA_HOME is None:
			raise RuntimeError("cuembed backend requires CUDA_HOME or a discoverable CUDA toolkit")
		if not _CUEMBED_SOURCE_PATH.is_file():
			raise RuntimeError(f"cuembed extension source is missing: {_CUEMBED_SOURCE_PATH}")
		_CUEMBED_EXTENSION = load(
			name="taac2026_cuembed_embedding_bag_mean",
			sources=[str(_CUEMBED_SOURCE_PATH)],
			extra_cflags=["-O3"],
			extra_cuda_cflags=["-O3", "--use_fast_math"],
			verbose=False,
		)
		return _CUEMBED_EXTENSION


def _cuembed_values(values: torch.Tensor) -> torch.Tensor:
	if values.dtype not in {torch.int32, torch.int64}:
		raise RuntimeError(f"cuembed embedding_bag_mean requires int32 or int64 values, got {values.dtype}")
	return values.contiguous()


def compile_cuembed_embedding_bag_mean_kernel(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
) -> CuEmbedEmbeddingBagMeanKernel:
	if embedding_weight.ndim != 2:
		raise ValueError("cuembed embedding_bag_mean weight must be a 2D tensor")
	if values.ndim != 2:
		raise ValueError("cuembed embedding_bag_mean values must be a 2D tensor")
	require_same_device("cuembed embedding_bag_mean", embedding_weight, values)
	require_cuda_tensors("cuembed embedding_bag_mean", embedding_weight, values)
	if embedding_weight.dtype not in {torch.float16, torch.float32}:
		raise RuntimeError(f"cuembed embedding_bag_mean does not support dtype {embedding_weight.dtype}")
	if embedding_weight.requires_grad:
		raise RuntimeError("cuembed embedding_bag_mean is currently forward-only; use torch or tilelang for training")
	module = _load_cuembed_extension()

	def runner(weight: torch.Tensor, lookup_values: torch.Tensor) -> torch.Tensor:
		require_same_device("cuembed embedding_bag_mean", weight, lookup_values)
		if weight.requires_grad:
			raise RuntimeError("cuembed embedding_bag_mean is currently forward-only; use torch or tilelang for training")
		return module.embedding_bag_mean_forward(weight.contiguous(), _cuembed_values(lookup_values))

	return runner


__all__ = [
	"CuEmbedEmbeddingBagMeanKernel",
	"clear_cuembed_embedding_bag_mean_kernel_cache",
	"compile_cuembed_embedding_bag_mean_kernel",
	"cuembed_available",
]

"""Accelerator-backed operator boundaries for PCVR kernels."""

from __future__ import annotations

from taac2026.infrastructure.accelerators.attention.flash_attention import (
    FlashAttentionBackend,
    FlashAttentionKernel,
    FlashAttentionKernelKey,
    FlashAttentionMaskPlan,
    _plan_tilelang_flash_attention_mask,
    _resolve_flash_attention_backend,
    clear_flash_attention_kernel_cache,
    compile_flash_attention_kernel,
    flash_attention,
    register_flash_attention_kernel,
    resolved_flash_attention_backend,
)
from taac2026.infrastructure.accelerators.attention.gated_delta_rule import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_available,
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule_fwd,
)
from taac2026.infrastructure.accelerators.attention.mla import multi_latent_attention
from taac2026.infrastructure.accelerators.embedding.embedding_bag import (
    EmbeddingBagMeanBackend,
    EmbeddingBagMeanKernel,
    EmbeddingBagMeanKernelKey,
    clear_embedding_bag_mean_kernel_cache,
    compile_embedding_bag_mean_kernel,
    compile_triton_embedding_bag_mean_kernel,
    embedding_bag_mean,
    register_embedding_bag_mean_kernel,
    resolved_embedding_bag_mean_backend,
)
from taac2026.infrastructure.accelerators.embedding.cuembed_runtime import (
    CuEmbedEmbeddingBagMeanKernel,
    compile_cuembed_embedding_bag_mean_kernel,
    cuembed_available,
)
from taac2026.infrastructure.accelerators.normalization.rms_norm import (
    RMSNormBackend,
    RMSNormKernel,
    RMSNormKernelKey,
    _resolve_rms_norm_backend,
    clear_rms_norm_kernel_cache,
    compile_rms_norm_kernel,
    compile_triton_rms_norm_kernel,
    register_rms_norm_kernel,
    resolved_rms_norm_backend,
    rms_norm,
)
from taac2026.infrastructure.accelerators.triton_runtime import triton_available
from taac2026.infrastructure.accelerators.tilelang_runtime import (
    _TILELANG_E8M0_COMPAT_GUARD,
    _TILELANG_E8M0_ORIGINAL_GUARD,
    _ensure_tilelang_cuda_fp8_compatibility,
    cuda_multiprocessor_count,
    tilelang_available,
)
from taac2026.infrastructure.accelerators.tensor_validation import (
    require_cuda_tensors,
    require_same_device,
    require_same_dtype,
)


def clear_tilelang_kernel_cache() -> None:
    clear_embedding_bag_mean_kernel_cache()
    clear_flash_attention_kernel_cache()
    clear_rms_norm_kernel_cache()


__all__ = [
    "_TILELANG_E8M0_COMPAT_GUARD",
    "_TILELANG_E8M0_ORIGINAL_GUARD",
    "CuEmbedEmbeddingBagMeanKernel",
    "EmbeddingBagMeanBackend",
    "EmbeddingBagMeanKernel",
    "EmbeddingBagMeanKernelKey",
    "FlashAttentionBackend",
    "FlashAttentionKernel",
    "FlashAttentionKernelKey",
    "FlashAttentionMaskPlan",
    "RMSNormBackend",
    "RMSNormKernel",
    "RMSNormKernelKey",
    "_ensure_tilelang_cuda_fp8_compatibility",
    "_plan_tilelang_flash_attention_mask",
    "_resolve_flash_attention_backend",
    "_resolve_rms_norm_backend",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_available",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule_fwd",
    "clear_embedding_bag_mean_kernel_cache",
    "clear_flash_attention_kernel_cache",
    "clear_rms_norm_kernel_cache",
    "clear_tilelang_kernel_cache",
    "compile_cuembed_embedding_bag_mean_kernel",
    "compile_embedding_bag_mean_kernel",
    "compile_flash_attention_kernel",
    "compile_rms_norm_kernel",
    "compile_triton_embedding_bag_mean_kernel",
    "compile_triton_rms_norm_kernel",
    "cuda_multiprocessor_count",
    "cuembed_available",
    "embedding_bag_mean",
    "flash_attention",
    "multi_latent_attention",
    "register_embedding_bag_mean_kernel",
    "register_flash_attention_kernel",
    "register_rms_norm_kernel",
    "require_cuda_tensors",
    "require_same_device",
    "require_same_dtype",
    "resolved_embedding_bag_mean_backend",
    "resolved_flash_attention_backend",
    "resolved_rms_norm_backend",
    "rms_norm",
    "tilelang_available",
    "triton_available",
]

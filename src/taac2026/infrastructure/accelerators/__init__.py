"""TileLang-backed operator boundary for PCVR kernels."""

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
from taac2026.infrastructure.accelerators.attention.mla import multi_latent_attention
from taac2026.infrastructure.accelerators.embedding.embedding_bag import (
    EmbeddingBagMeanKernel,
    embedding_bag_mean,
    register_embedding_bag_mean_kernel,
)
from taac2026.infrastructure.accelerators.normalization.rms_norm import (
    RMSNormBackend,
    RMSNormKernel,
    RMSNormKernelKey,
    _resolve_rms_norm_backend,
    clear_rms_norm_kernel_cache,
    compile_rms_norm_kernel,
    register_rms_norm_kernel,
    resolved_rms_norm_backend,
    rms_norm,
)
from taac2026.infrastructure.accelerators.tilelang_runtime import (
    _TILELANG_E8M0_COMPAT_GUARD,
    _TILELANG_E8M0_ORIGINAL_GUARD,
    _ensure_tilelang_cuda_fp8_compatibility,
    tilelang_available,
)


def clear_tilelang_kernel_cache() -> None:
    clear_flash_attention_kernel_cache()
    clear_rms_norm_kernel_cache()


__all__ = [
    "_TILELANG_E8M0_COMPAT_GUARD",
    "_TILELANG_E8M0_ORIGINAL_GUARD",
    "EmbeddingBagMeanKernel",
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
    "clear_flash_attention_kernel_cache",
    "clear_rms_norm_kernel_cache",
    "clear_tilelang_kernel_cache",
    "compile_flash_attention_kernel",
    "compile_rms_norm_kernel",
    "embedding_bag_mean",
    "flash_attention",
    "multi_latent_attention",
    "register_embedding_bag_mean_kernel",
    "register_flash_attention_kernel",
    "register_rms_norm_kernel",
    "resolved_flash_attention_backend",
    "resolved_rms_norm_backend",
    "rms_norm",
    "tilelang_available",
]

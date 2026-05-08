"""Stable public imports for experiment packages."""

from __future__ import annotations

from taac2026.application.experiments.factory import create_pcvr_experiment
from taac2026.domain.config import (
    PCVRDataCacheConfig,
    PCVRDataConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRModelConfig,
    PCVRNSConfig,
    PCVROptimizerConfig,
    PCVRSparseOptimizerConfig,
    PCVRSequenceCropConfig,
    PCVRTrainConfig,
)
from taac2026.domain.model_contract import ModelInput
from taac2026.infrastructure import modeling as _modeling
from taac2026.infrastructure.modeling import (
    DenseTokenProjector,
    EmbeddingParameterMixin,
    FeatureEmbeddingBank,
    FlashAttentionBackend,
    NonSequentialTokenizer,
    RMSNorm,
    SequenceTokenizer,
    causal_valid_attention_mask,
    choose_num_heads,
    flash_attention_runtime_state,
    make_padding_mask,
    masked_last,
    masked_mean,
    maybe_gradient_checkpoint,
    safe_key_padding_mask,
    scaled_dot_product_attention,
    sinusoidal_positions,
)
from taac2026.infrastructure.runtime.execution import PCVRLossConfig, PCVRLossTermConfig, RuntimeExecutionConfig

RMS_NORM_BACKEND = _modeling.RMS_NORM_BACKEND
RMS_NORM_BLOCK_ROWS = _modeling.RMS_NORM_BLOCK_ROWS
FLASH_ATTENTION_BACKEND = _modeling.FLASH_ATTENTION_BACKEND


def configure_flash_attention_runtime(*, backend: str) -> None:
    global FLASH_ATTENTION_BACKEND
    _modeling.configure_flash_attention_runtime(backend=backend)
    FLASH_ATTENTION_BACKEND = _modeling.FLASH_ATTENTION_BACKEND


def configure_rms_norm_runtime(*, backend: str, block_rows: int) -> None:
    global RMS_NORM_BACKEND, RMS_NORM_BLOCK_ROWS
    _modeling.configure_rms_norm_runtime(backend=backend, block_rows=block_rows)
    RMS_NORM_BACKEND = _modeling.RMS_NORM_BACKEND
    RMS_NORM_BLOCK_ROWS = _modeling.RMS_NORM_BLOCK_ROWS

__all__ = [
    "FLASH_ATTENTION_BACKEND",
    "RMS_NORM_BACKEND",
    "RMS_NORM_BLOCK_ROWS",
    "DenseTokenProjector",
    "EmbeddingParameterMixin",
    "FeatureEmbeddingBank",
    "FlashAttentionBackend",
    "ModelInput",
    "NonSequentialTokenizer",
    "PCVRDataCacheConfig",
    "PCVRDataConfig",
    "PCVRDataPipelineConfig",
    "PCVRDomainDropoutConfig",
    "PCVRFeatureMaskConfig",
    "PCVRLossConfig",
    "PCVRLossTermConfig",
    "PCVRModelConfig",
    "PCVRNSConfig",
    "PCVROptimizerConfig",
    "PCVRSequenceCropConfig",
    "PCVRSparseOptimizerConfig",
    "PCVRTrainConfig",
    "RMSNorm",
    "RuntimeExecutionConfig",
    "SequenceTokenizer",
    "causal_valid_attention_mask",
    "choose_num_heads",
    "configure_flash_attention_runtime",
    "configure_rms_norm_runtime",
    "create_pcvr_experiment",
    "flash_attention_runtime_state",
    "make_padding_mask",
    "masked_last",
    "masked_mean",
    "maybe_gradient_checkpoint",
    "safe_key_padding_mask",
    "scaled_dot_product_attention",
    "sinusoidal_positions",
]

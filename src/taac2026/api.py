"""Stable public imports for experiment packages."""

from __future__ import annotations

from taac2026.application.experiments.factory import create_pcvr_experiment
from taac2026.application.training.workflow import NoopTrainReporter, TensorBoardTrainReporter, TrainReporter
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
    PCVRValidationConfig,
)
from taac2026.infrastructure.modeling.model_contract import ModelInput
from taac2026.domain.runtime_config import PCVRLossConfig, PCVRLossTermConfig, RuntimeExecutionConfig
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


def configure_flash_attention_runtime(*, backend: str) -> None:
    _modeling.configure_flash_attention_runtime(backend=backend)


def configure_rms_norm_runtime(*, backend: str, block_rows: int) -> None:
    _modeling.configure_rms_norm_runtime(backend=backend, block_rows=block_rows)


def rms_norm_runtime_state() -> tuple[str, int]:
    return _modeling.rms_norm_runtime_state()


def __getattr__(name: str):
    if name in {"FLASH_ATTENTION_BACKEND", "RMS_NORM_BACKEND", "RMS_NORM_BLOCK_ROWS"}:
        return getattr(_modeling, name)
    raise AttributeError(name)


__all__ = [
    "DenseTokenProjector",
    "EmbeddingParameterMixin",
    "FeatureEmbeddingBank",
    "FlashAttentionBackend",
    "ModelInput",
    "NonSequentialTokenizer",
    "NoopTrainReporter",
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
    "PCVRValidationConfig",
    "RMSNorm",
    "RuntimeExecutionConfig",
    "SequenceTokenizer",
    "TensorBoardTrainReporter",
    "TrainReporter",
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
    "rms_norm_runtime_state",
    "safe_key_padding_mask",
    "scaled_dot_product_attention",
    "sinusoidal_positions",
]

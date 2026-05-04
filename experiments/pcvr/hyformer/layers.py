"""Compatibility re-exports for shared PCVR model primitives."""

from taac2026.infrastructure.pcvr import modeling as _modeling
from taac2026.infrastructure.pcvr.modeling import (
    DenseTokenProjector,
    EmbeddingParameterMixin,
    FeatureEmbeddingBank,
    NonSequentialTokenizer,
    RMSNorm,
    SequenceTokenizer,
)

RMS_NORM_BACKEND = _modeling.RMS_NORM_BACKEND
RMS_NORM_BLOCK_ROWS = _modeling.RMS_NORM_BLOCK_ROWS


def configure_rms_norm_runtime(*, backend: str, block_rows: int) -> None:
    global RMS_NORM_BACKEND, RMS_NORM_BLOCK_ROWS
    _modeling.configure_rms_norm_runtime(backend=backend, block_rows=block_rows)
    RMS_NORM_BACKEND = _modeling.RMS_NORM_BACKEND
    RMS_NORM_BLOCK_ROWS = _modeling.RMS_NORM_BLOCK_ROWS


__all__ = [
    "RMS_NORM_BACKEND",
    "RMS_NORM_BLOCK_ROWS",
    "DenseTokenProjector",
    "EmbeddingParameterMixin",
    "FeatureEmbeddingBank",
    "NonSequentialTokenizer",
    "RMSNorm",
    "SequenceTokenizer",
    "configure_rms_norm_runtime",
]
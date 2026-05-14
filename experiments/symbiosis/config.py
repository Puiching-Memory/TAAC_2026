"""Symbiosis V2 configuration surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SymbiosisModelDefaults:
    v2_use_dense_tokens: bool = True
    v2_use_missing_tokens: bool = True
    v2_use_sequence_stats_tokens: bool = True
    v2_use_metadata_attention_bias: bool = True
    v2_use_candidate_readout: bool = True
    v2_tokenization_mode: str = "group"
    v2_sparse_seed: int = 20260512
    v2_recent_event_tokens: int = 16
    v2_memory_event_tokens: int = 8
    v2_user_dense_tokens: int = 2
    v2_item_dense_tokens: int = 1
    v2_user_missing_tokens: int = 2
    v2_item_missing_tokens: int = 1
    v2_high_risk_token_dropout_rate: float = 0.08
    v2_compress_large_ids: bool = True
    v2_compile_backbone: bool = True

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "symbiosis_v2_use_dense_tokens": self.v2_use_dense_tokens,
            "symbiosis_v2_use_missing_tokens": self.v2_use_missing_tokens,
            "symbiosis_v2_use_sequence_stats_tokens": self.v2_use_sequence_stats_tokens,
            "symbiosis_v2_use_metadata_attention_bias": self.v2_use_metadata_attention_bias,
            "symbiosis_v2_use_candidate_readout": self.v2_use_candidate_readout,
            "symbiosis_v2_tokenization_mode": self.v2_tokenization_mode,
            "symbiosis_v2_sparse_seed": self.v2_sparse_seed,
            "symbiosis_v2_recent_event_tokens": self.v2_recent_event_tokens,
            "symbiosis_v2_memory_event_tokens": self.v2_memory_event_tokens,
            "symbiosis_v2_user_dense_tokens": self.v2_user_dense_tokens,
            "symbiosis_v2_item_dense_tokens": self.v2_item_dense_tokens,
            "symbiosis_v2_user_missing_tokens": self.v2_user_missing_tokens,
            "symbiosis_v2_item_missing_tokens": self.v2_item_missing_tokens,
            "symbiosis_v2_high_risk_token_dropout_rate": self.v2_high_risk_token_dropout_rate,
            "symbiosis_v2_compress_large_ids": self.v2_compress_large_ids,
            "symbiosis_v2_compile_backbone": self.v2_compile_backbone,
        }


SYMBIOSIS_MODEL_DEFAULTS = SymbiosisModelDefaults()
SYMBIOSIS_MODEL_CONFIG_KEYS = tuple(SYMBIOSIS_MODEL_DEFAULTS.to_flat_dict())
SYMBIOSIS_OPTIONAL_MODEL_CONFIG_DEFAULTS: dict[str, Any] = {}
SYMBIOSIS_OPTIONAL_MODEL_CONFIG_KEYS = tuple(SYMBIOSIS_OPTIONAL_MODEL_CONFIG_DEFAULTS)


__all__ = [
    "SYMBIOSIS_MODEL_CONFIG_KEYS",
    "SYMBIOSIS_MODEL_DEFAULTS",
    "SYMBIOSIS_OPTIONAL_MODEL_CONFIG_DEFAULTS",
    "SYMBIOSIS_OPTIONAL_MODEL_CONFIG_KEYS",
    "SymbiosisModelDefaults",
]
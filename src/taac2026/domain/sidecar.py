"""Versioned PCVR train_config.json sidecar helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import field_validator

from taac2026 import __version__
from taac2026.domain.validation import TAACBoundaryModel


PCVR_TRAIN_CONFIG_FORMAT = "taac2026-pcvr-train-config"
PCVR_TRAIN_CONFIG_VERSION = 1

PCVR_TRAIN_CONFIG_METADATA_KEYS = frozenset(
    {
        "train_config_format",
        "train_config_version",
        "framework_name",
        "framework_version",
    }
)


class PCVRTrainConfigSidecar(TAACBoundaryModel):
    """Pydantic model for the current PCVR train_config payload."""

    train_config_format: str
    train_config_version: int
    framework_name: str
    framework_version: str
    train_config: dict[str, Any]

    @field_validator("train_config_format")
    @classmethod
    def _validate_format(cls, value: str) -> str:
        if value != PCVR_TRAIN_CONFIG_FORMAT:
            raise ValueError(f"unsupported PCVR train_config format: {value}")
        return value

    @field_validator("train_config_version")
    @classmethod
    def _validate_version(cls, value: int) -> int:
        if value != PCVR_TRAIN_CONFIG_VERSION:
            raise ValueError(f"unsupported PCVR train_config version: {value}")
        return value

    def to_runtime_config(self) -> dict[str, Any]:
        config = dict(self.train_config)
        metadata = self.model_dump(
            mode="python",
            include=PCVR_TRAIN_CONFIG_METADATA_KEYS,
            exclude_none=True,
        )
        config.update(metadata)
        return config

    def to_sidecar_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


def build_pcvr_train_config_sidecar(train_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the current PCVR train_config.json payload."""

    config = {} if train_config is None else dict(train_config)
    return PCVRTrainConfigSidecar.model_validate(
        {
            "train_config_format": PCVR_TRAIN_CONFIG_FORMAT,
            "train_config_version": PCVR_TRAIN_CONFIG_VERSION,
            "framework_name": "taac2026",
            "framework_version": __version__,
            "train_config": config,
        }
    ).to_sidecar_payload()


def load_pcvr_train_config_sidecar(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Load the current PCVR train_config.json payload."""

    return PCVRTrainConfigSidecar.model_validate(dict(payload)).to_runtime_config()


__all__ = [
    "PCVR_TRAIN_CONFIG_FORMAT",
    "PCVR_TRAIN_CONFIG_METADATA_KEYS",
    "PCVR_TRAIN_CONFIG_VERSION",
    "PCVRTrainConfigSidecar",
    "build_pcvr_train_config_sidecar",
    "load_pcvr_train_config_sidecar",
]

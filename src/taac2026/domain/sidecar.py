"""Versioned PCVR train_config.json sidecar helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from taac2026 import __version__


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


class PCVRTrainConfigSidecar(BaseModel):
    """Pydantic model for legacy flat and versioned PCVR train_config payloads."""

    model_config = ConfigDict(extra="allow")

    train_config_format: str | None = None
    train_config_version: int | None = None
    framework_name: str | None = None
    framework_version: str | None = None
    train_config: dict[str, Any] | None = None

    @field_validator("train_config_format")
    @classmethod
    def _validate_format(cls, value: str | None) -> str | None:
        if value is not None and value != PCVR_TRAIN_CONFIG_FORMAT:
            raise ValueError(f"unsupported PCVR train_config format: {value}")
        return value

    @field_validator("train_config_version")
    @classmethod
    def _validate_version(cls, value: int | None) -> int | None:
        if value is not None and value > PCVR_TRAIN_CONFIG_VERSION:
            raise ValueError(f"unsupported PCVR train_config version: {value}")
        return value

    def to_runtime_config(self) -> dict[str, Any]:
        if self.train_config is None:
            config = dict(self.model_extra or {})
            for key in PCVR_TRAIN_CONFIG_METADATA_KEYS:
                value = getattr(self, key)
                if value is not None:
                    config[key] = value
            return config

        config = dict(self.train_config)
        metadata = self.model_dump(
            mode="python",
            include=PCVR_TRAIN_CONFIG_METADATA_KEYS,
            exclude_none=True,
        )
        config.update(metadata)
        return config


def build_pcvr_train_config_sidecar(train_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a versioned, flat PCVR train_config.json payload."""

    config = {} if train_config is None else dict(train_config)
    return PCVRTrainConfigSidecar.model_validate(
        {
            **config,
            "train_config_format": PCVR_TRAIN_CONFIG_FORMAT,
            "train_config_version": PCVR_TRAIN_CONFIG_VERSION,
            "framework_name": "taac2026",
            "framework_version": __version__,
        }
    ).to_runtime_config()


def normalize_pcvr_train_config_sidecar(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy flat and versioned PCVR train_config.json payloads."""

    return PCVRTrainConfigSidecar.model_validate(dict(payload)).to_runtime_config()


__all__ = [
    "PCVR_TRAIN_CONFIG_FORMAT",
    "PCVR_TRAIN_CONFIG_METADATA_KEYS",
    "PCVR_TRAIN_CONFIG_VERSION",
    "PCVRTrainConfigSidecar",
    "build_pcvr_train_config_sidecar",
    "normalize_pcvr_train_config_sidecar",
]

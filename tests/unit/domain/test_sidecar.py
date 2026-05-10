from __future__ import annotations

import pytest

from taac2026.domain.sidecar import (
    PCVR_TRAIN_CONFIG_VERSION,
    build_pcvr_train_config_sidecar,
    load_pcvr_train_config_sidecar,
)


def test_load_pcvr_train_config_sidecar_accepts_current_payload() -> None:
    payload = build_pcvr_train_config_sidecar({"batch_size": 32})

    loaded = load_pcvr_train_config_sidecar(payload)

    assert loaded["batch_size"] == 32
    assert loaded["train_config_version"] == PCVR_TRAIN_CONFIG_VERSION


def test_load_pcvr_train_config_sidecar_wraps_legacy_flat_payload() -> None:
    loaded = load_pcvr_train_config_sidecar({"batch_size": 64})

    assert loaded["batch_size"] == 64
    assert loaded["train_config_version"] == PCVR_TRAIN_CONFIG_VERSION


def test_load_pcvr_train_config_sidecar_rejects_unknown_version() -> None:
    payload = build_pcvr_train_config_sidecar({"batch_size": 32})
    payload["train_config_version"] = 999

    with pytest.raises(ValueError, match="unsupported PCVR train_config version: 999"):
        load_pcvr_train_config_sidecar(payload)
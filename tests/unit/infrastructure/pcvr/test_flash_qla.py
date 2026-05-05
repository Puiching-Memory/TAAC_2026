from __future__ import annotations

from taac2026.infrastructure.pcvr import flash_qla
import pytest
import torch


def test_flash_qla_import_exports_runtime_helpers() -> None:
    assert callable(flash_qla.chunk_gated_delta_rule)
    assert isinstance(flash_qla.flash_qla_available(), bool)
    target_version = flash_qla.flash_qla_target_compute_version()
    assert target_version is None or isinstance(target_version, str)


def test_chunk_gated_delta_rule_rejects_cpu_inputs() -> None:
    q = torch.randn(2, 8, 4, 16, dtype=torch.float16)
    k = torch.randn(2, 8, 4, 16, dtype=torch.float16)
    v = torch.randn(2, 8, 4, 16, dtype=torch.float16)
    g = torch.randn(2, 8, 4, dtype=torch.float16)
    beta = torch.randn(2, 8, 4, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        flash_qla.chunk_gated_delta_rule(q, k, v, g, beta)
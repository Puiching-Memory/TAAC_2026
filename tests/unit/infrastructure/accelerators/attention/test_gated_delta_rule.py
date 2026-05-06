from __future__ import annotations

import pytest
import torch

from taac2026.infrastructure.accelerators.attention import gated_delta_rule


def test_gated_delta_rule_exports_runtime_helpers() -> None:
    assert callable(gated_delta_rule.chunk_gated_delta_rule)
    assert isinstance(gated_delta_rule.chunk_gated_delta_rule_available(), bool)


def test_chunk_gated_delta_rule_rejects_cpu_inputs() -> None:
    q = torch.randn(2, 8, 4, 16, dtype=torch.float16)
    k = torch.randn(2, 8, 4, 16, dtype=torch.float16)
    v = torch.randn(2, 8, 4, 16, dtype=torch.float16)
    g = torch.randn(2, 8, 4, dtype=torch.float16)
    beta = torch.randn(2, 8, 4, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        gated_delta_rule.chunk_gated_delta_rule(q, k, v, g, beta)


def test_chunk_gated_delta_rule_rejects_mixed_devices() -> None:
    q = torch.empty(2, 8, 4, 16, dtype=torch.float16, device="meta")
    k = torch.randn(2, 8, 4, 16, dtype=torch.float16)
    v = torch.randn(2, 8, 4, 16, dtype=torch.float16)
    g = torch.randn(2, 8, 4, dtype=torch.float16)
    beta = torch.randn(2, 8, 4, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="requires all tensors to live on the same device"):
        gated_delta_rule.chunk_gated_delta_rule(q, k, v, g, beta)

from __future__ import annotations

import pytest
import torch

from taac2026.infrastructure.nn.norms import rms_norm
from taac2026.infrastructure.nn.triton_attention import reference_attention, triton_attention
from taac2026.infrastructure.nn.triton_ffn import reference_ffn_activation, triton_ffn_activation
from taac2026.infrastructure.nn.triton_norm import triton_rms_norm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")
def test_triton_rms_norm_matches_reference() -> None:
    hidden_states = torch.randn(4, 16, 64, device="cuda", dtype=torch.float32)
    weight = torch.randn(64, device="cuda", dtype=torch.float32)

    expected = rms_norm(hidden_states, weight)
    actual = triton_rms_norm(hidden_states, weight)

    assert torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")
def test_triton_attention_matches_reference() -> None:
    query = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float32)
    key = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float32)
    value = torch.randn(2, 2, 8, 16, device="cuda", dtype=torch.float32)
    attention_mask = torch.tril(torch.ones(8, 8, device="cuda", dtype=torch.bool)).unsqueeze(0).expand(2, -1, -1)
    key_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True, False],
        ],
        device="cuda",
    )

    with torch.no_grad():
        expected = reference_attention(query, key, value, attention_mask=attention_mask, key_mask=key_mask)
        actual = triton_attention(query, key, value, attention_mask=attention_mask, key_mask=key_mask, backend="triton")

    assert torch.allclose(actual, expected, atol=1.0e-4, rtol=1.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")
def test_triton_ffn_activation_matches_reference() -> None:
    projected = torch.randn(4, 12, 32, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        expected = reference_ffn_activation(projected, "silu")
        actual = triton_ffn_activation(projected, "silu", backend="triton")

    assert torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")
def test_triton_swiglu_matches_reference() -> None:
    projected = torch.randn(3, 10, 64, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        expected = reference_ffn_activation(projected, "swiglu")
        actual = triton_ffn_activation(projected, "swiglu", backend="triton")

    assert torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-4)
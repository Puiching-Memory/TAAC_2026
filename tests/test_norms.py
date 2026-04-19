from __future__ import annotations

import torch

from taac2026.infrastructure.nn.norms import RMSNorm, rms_norm


def test_rms_norm_matches_manual_formula() -> None:
    hidden_states = torch.tensor(
        [
            [[1.0, -2.0, 3.0], [0.5, 0.5, 0.5]],
            [[4.0, 0.0, -4.0], [2.0, -1.0, 2.0]],
        ]
    )
    weight = torch.tensor([1.0, 2.0, 3.0])

    actual = rms_norm(hidden_states, weight, eps=1.0e-6)

    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    expected = hidden_states * torch.rsqrt(variance + 1.0e-6) * weight
    assert torch.allclose(actual, expected)


def test_rms_norm_module_uses_configured_epsilon_and_weight() -> None:
    module = RMSNorm(hidden_dim=4, eps=1.0e-5)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([1.0, 0.5, 2.0, 1.5]))

    hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    actual = module(hidden_states)
    expected = rms_norm(hidden_states, module.weight, eps=1.0e-5)

    assert torch.allclose(actual, expected)


def test_rms_norm_preserves_input_shape() -> None:
    module = RMSNorm(hidden_dim=6)
    hidden_states = torch.randn(3, 5, 6)

    normalized = module(hidden_states)

    assert normalized.shape == hidden_states.shape
    assert torch.isfinite(normalized).all().item()
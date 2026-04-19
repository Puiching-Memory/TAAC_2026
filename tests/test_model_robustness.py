"""Model robustness tests that go beyond shape/finite smoke checks.

These tests target the class of bugs found during PR #10 review:
  - Padding invariant violations (inter-block residual leaking into padded positions)
  - BlockAttnRes partial_sum / hidden-state consistency
  - dtype propagation under mixed-precision (AMP)
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from taac2026.infrastructure.nn.defaults import resolve_experiment_builders
from tests.support import TestWorkspace, create_test_workspace, prepare_experiment


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


def _build_unirec(test_workspace: TestWorkspace):
    """Load the UniRec experiment, build data pipeline + model, and return a batch."""
    experiment = importlib.import_module("config.unirec").EXPERIMENT
    experiment = prepare_experiment(experiment, test_workspace)
    builders = resolve_experiment_builders(experiment)
    train_loader, _, data_stats = builders.build_data_pipeline(
        experiment.data,
        experiment.model,
        experiment.train,
    )
    batch = next(iter(train_loader))
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    return model, batch, experiment


# ---------------------------------------------------------------------------
# 1. Padding invariant: padded token positions must remain zero after forward
# ---------------------------------------------------------------------------


def test_unirec_padding_positions_remain_zero(test_workspace: TestWorkspace) -> None:
    """After forward, hidden states at padding positions should not carry
    leaked signal from inter-block residuals or broadcasts.

    We hook into the final_norm layer to capture the hidden states right
    before the head, then check that padding positions are all-zero.
    """
    model, batch, experiment = _build_unirec(test_workspace)
    model.eval()

    captured: list[torch.Tensor] = []

    def hook(module, input, output):
        captured.append(input[0].detach().clone())

    # Register a forward hook on the layer normalization just before the head
    handle = model.final_norm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(batch)
    finally:
        handle.remove()

    assert len(captured) == 1, "Hook should fire exactly once"
    hidden_states = captured[0]  # (B, L, D)

    # Reconstruct the padding mask the same way forward() does
    _, padding_mask, _, _ = model._build_unified_sequence(batch)
    padded_positions = ~padding_mask  # True where padded

    if padded_positions.any():
        padded_hidden = hidden_states[padded_positions]
        max_abs = padded_hidden.abs().max().item()
        assert max_abs == 0.0, (
            f"Padding positions should be zero but found max abs value {max_abs:.6e}"
        )


# ---------------------------------------------------------------------------
# 2. BlockAttnRes consistency: partial_sum must track post-residual last token
# ---------------------------------------------------------------------------


def test_block_attn_res_partial_sum_matches_post_residual() -> None:
    """partial_sum should accumulate the *post-residual* last tokens, not
    the pre-residual ones.  When block_summaries is non-empty, the returned
    partial_sum must equal the sum of all post-residual last-tokens seen so far.
    """
    from config.unirec.model import BlockAttnRes

    dim = 16
    batch_size = 2
    seq_len = 5
    total_layers = 4

    block_res = BlockAttnRes(dim=dim, total_layers=total_layers)
    block_res.eval()

    torch.manual_seed(42)

    # Simulate two layers producing block_summaries, then a third layer
    block_summaries: list[torch.Tensor] = []
    partial_sum = torch.zeros(batch_size, dim)
    accumulated_last_tokens: list[torch.Tensor] = []

    for layer_idx in range(3):
        layer_output = torch.randn(batch_size, seq_len, dim)

        with torch.no_grad():
            combined, partial_sum, current_last = block_res(
                layer_idx, layer_output, block_summaries, partial_sum,
            )

        # current_last must be the last token of combined (post-residual)
        expected_last = combined[:, -1, :]
        assert torch.allclose(current_last, expected_last, atol=1e-6), (
            f"Layer {layer_idx}: current_last doesn't match combined[:, -1, :]"
        )

        accumulated_last_tokens.append(current_last)

        # partial_sum must equal the sum of all post-residual last tokens
        expected_partial = torch.stack(accumulated_last_tokens, dim=0).sum(dim=0)
        assert torch.allclose(partial_sum, expected_partial, atol=1e-5), (
            f"Layer {layer_idx}: partial_sum deviates from accumulated post-residual last tokens"
        )

        # Every 2 layers, start a new block
        if (layer_idx + 1) % 2 == 0:
            block_summaries.append(partial_sum.clone())
            partial_sum = torch.zeros(batch_size, dim)
            accumulated_last_tokens.clear()


# ---------------------------------------------------------------------------
# 3. dtype propagation: model should not silently upcast under autocast
# ---------------------------------------------------------------------------


def test_unirec_forward_under_cpu_autocast(test_workspace: TestWorkspace) -> None:
    """Run UniRec forward under CPU bfloat16 autocast and verify:
    - Output is finite
    - No errors from dtype mismatches (the original bug: torch.zeros defaulting
      to fp32 while hidden_states is bf16 under AMP)
    """
    model, batch, experiment = _build_unirec(test_workspace)
    model.eval()

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(batch)

    assert logits.shape == batch.labels.shape
    assert torch.isfinite(logits).all().item(), "logits contain non-finite values under autocast"

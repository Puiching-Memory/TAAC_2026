from __future__ import annotations

import pytest
import torch
from torch import nn

from taac2026.infrastructure.nn.heads import ClassificationHead
from taac2026.infrastructure.nn.pooling import TargetAwarePool, masked_mean


def test_masked_mean_supports_single_and_multi_axis_reduction() -> None:
    tokens = torch.tensor(
        [
            [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]],
            [[2.0, 20.0], [4.0, 40.0], [6.0, 60.0]],
        ]
    )
    mask = torch.tensor(
        [
            [True, False, True],
            [False, False, False],
        ]
    )

    pooled = masked_mean(tokens, mask)
    assert torch.allclose(pooled[0], torch.tensor([3.0, 30.0]))
    assert torch.allclose(pooled[1], torch.tensor([0.0, 0.0]))

    grid_tokens = tokens.unsqueeze(1).repeat(1, 2, 1, 1)
    grid_mask = mask.unsqueeze(1).repeat(1, 2, 1)
    pooled_grid = masked_mean(grid_tokens, grid_mask, dim=(1, 2))
    assert torch.allclose(pooled_grid, pooled)


def test_target_aware_pool_matches_reference_attention_formula() -> None:
    pool = TargetAwarePool(hidden_dim=2, activation="relu", dropout=0.0)
    query = torch.tensor([[0.5, -0.5]])
    keys = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [4.0, 4.0]]])
    key_mask = torch.tensor([[True, True, False]])

    expanded_query = query.unsqueeze(1).expand_as(keys)
    attention_inputs = torch.cat(
        [
            expanded_query,
            keys,
            expanded_query - keys,
            expanded_query * keys,
        ],
        dim=-1,
    )
    expected_scores = pool.scorer(attention_inputs).squeeze(-1).masked_fill(~key_mask, -1.0e4)
    expected_weights = torch.softmax(expected_scores, dim=-1)
    expected_weights = expected_weights * key_mask.float()
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    expected = torch.bmm(expected_weights.unsqueeze(1), keys).squeeze(1)

    actual = pool(query, keys, key_mask)
    assert torch.allclose(actual, expected)


def test_classification_head_matches_equivalent_sequential_stack() -> None:
    head = ClassificationHead(
        input_dim=5,
        hidden_dims=[7, 3],
        activation="prelu",
        dropout=[0.0, 0.0],
    )
    reference = nn.Sequential(
        nn.LayerNorm(5),
        nn.Linear(5, 7),
        nn.PReLU(),
        nn.Dropout(0.0),
        nn.Linear(7, 3),
        nn.PReLU(),
        nn.Dropout(0.0),
        nn.Linear(3, 1),
    )
    reference.load_state_dict(head.layers.state_dict())

    inputs = torch.randn(4, 5)
    assert torch.allclose(head(inputs), reference(inputs))


def test_classification_head_validates_dropout_schedule_length() -> None:
    with pytest.raises(ValueError, match="dropout schedule length"):
        ClassificationHead(input_dim=4, hidden_dims=[8, 4], dropout=[0.1])
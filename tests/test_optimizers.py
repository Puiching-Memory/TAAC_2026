from __future__ import annotations

import torch
from torch import nn
from torchrec.optim import RowWiseAdagrad

from taac2026.domain.config import TrainConfig
from taac2026.infrastructure.nn.defaults import default_build_optimizer
from taac2026.infrastructure.nn.optimizers import CombinedOptimizer, Muon


def test_muon_updates_matrix_and_vector_parameters() -> None:
    matrix = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [0.5, 2.0]]))
    vector = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
    optimizer = Muon([matrix, vector], lr=1.0e-2, momentum=0.9, ns_steps=2)

    matrix_before = matrix.detach().clone()
    vector_before = vector.detach().clone()
    matrix.grad = torch.tensor([[0.2, -0.1], [0.05, 0.3]])
    vector.grad = torch.tensor([0.1, -0.2])

    optimizer.step()

    assert not torch.allclose(matrix, matrix_before)
    assert not torch.allclose(vector, vector_before)
    assert torch.isfinite(matrix).all().item()
    assert torch.isfinite(vector).all().item()
def test_combined_optimizer_round_trips_nested_state_dicts() -> None:
    left = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    right = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    optimizer = CombinedOptimizer(
        [
            torch.optim.SGD([left], lr=1.0e-1, momentum=0.9),
            torch.optim.AdamW([right], lr=1.0e-2),
        ]
    )

    left.grad = torch.tensor([0.5, -0.25])
    right.grad = torch.tensor([0.1, -0.2])
    optimizer.step()
    state_dict = optimizer.state_dict()

    new_left = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    new_right = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    new_optimizer = CombinedOptimizer(
        [
            torch.optim.SGD([new_left], lr=1.0e-1, momentum=0.9),
            torch.optim.AdamW([new_right], lr=1.0e-2),
        ]
    )
    new_optimizer.load_state_dict(state_dict)

    assert isinstance(state_dict, list)
    assert len(state_dict) == 2
    assert len(new_optimizer.param_groups) == 2
    assert new_optimizer.state_dict() == state_dict


def test_default_build_optimizer_routes_embeddings_matrices_and_no_decay_weights() -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(32, 8)
            self.projection = nn.Linear(8, 4)
            self.norm = nn.LayerNorm(4)
            self.extra_bias = nn.Parameter(torch.ones(4))

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            hidden_states = self.embedding(token_ids).mean(dim=1)
            return self.norm(self.projection(hidden_states)) + self.extra_bias

    model = TinyModel()
    optimizer = default_build_optimizer(
        model,
        TrainConfig(learning_rate=1.0e-3, weight_decay=1.0e-2),
    )

    assert isinstance(optimizer, CombinedOptimizer)
    assert any(isinstance(component, RowWiseAdagrad) for component in optimizer.optimizers)
    assert any(isinstance(component, Muon) for component in optimizer.optimizers)

    adamw = next(component for component in optimizer.optimizers if isinstance(component, torch.optim.AdamW))
    assert [group["weight_decay"] for group in adamw.param_groups] == [1.0e-2, 0.0]
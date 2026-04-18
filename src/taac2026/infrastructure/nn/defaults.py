from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from ...domain.experiment import ExperimentSpec
from ...domain.features import FeatureSchema
from ..io.default_data_pipeline import load_dataloaders
from .optimizers import build_hybrid_optimizer


class DisabledAuxiliaryLoss:
    enabled = False
    requires_aux = False


class _PairwiseAUCLoss(nn.Module):
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        positive_mask = labels > 0.5
        negative_mask = ~positive_mask
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return logits.new_tensor(0.0)
        positive_scores = logits[positive_mask]
        negative_scores = logits[negative_mask]
        margins = positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0)
        return F.softplus(-margins).mean()


class RankingLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, pairwise_weight: float) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.pairwise = _PairwiseAUCLoss()
        self.pairwise_weight = min(max(pairwise_weight, 0.0), 1.0)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, labels)
        pairwise_loss = self.pairwise(logits, labels)
        return (1.0 - self.pairwise_weight) * bce_loss + self.pairwise_weight * pairwise_loss


def default_build_data_pipeline(
    feature_schema: FeatureSchema,
    data_config,
    model_config,
    train_config,
):
    del feature_schema
    return load_dataloaders(
        config=data_config,
        vocab_size=model_config.vocab_size,
        batch_size=train_config.batch_size,
        eval_batch_size=train_config.resolved_eval_batch_size,
        num_workers=train_config.num_workers,
        seed=train_config.seed,
    )


def default_build_loss_stack(data_config, model_config, train_config, data_stats, device):
    del data_config
    del model_config
    pos_weight = torch.tensor([data_stats.pos_weight], dtype=torch.float32, device=device)
    return RankingLoss(pos_weight=pos_weight, pairwise_weight=train_config.pairwise_weight), DisabledAuxiliaryLoss()


def default_build_optimizer(model, train_config):
    return build_hybrid_optimizer(model, train_config)


@dataclass(frozen=True, slots=True)
class ExperimentBuilders:
    build_data_pipeline: Callable[..., Any]
    build_loss_stack: Callable[..., Any]
    build_optimizer_component: Callable[..., Any]


def resolve_experiment_builders(experiment: ExperimentSpec) -> ExperimentBuilders:
    experiment.refresh_feature_schema()
    feature_schema = experiment.feature_schema
    if feature_schema is None:
        raise RuntimeError(f"Experiment '{experiment.name}' has no resolved feature schema")

    if experiment.build_data_pipeline is None:
        def build_data_pipeline(data_config, model_config, train_config):
            return default_build_data_pipeline(feature_schema, data_config, model_config, train_config)
    else:
        build_data_pipeline = experiment.build_data_pipeline

    return ExperimentBuilders(
        build_data_pipeline=build_data_pipeline,
        build_loss_stack=experiment.build_loss_stack or default_build_loss_stack,
        build_optimizer_component=experiment.build_optimizer_component or default_build_optimizer,
    )


__all__ = [
    "DisabledAuxiliaryLoss",
    "ExperimentBuilders",
    "RankingLoss",
    "default_build_data_pipeline",
    "default_build_loss_stack",
    "default_build_optimizer",
    "resolve_experiment_builders",
]
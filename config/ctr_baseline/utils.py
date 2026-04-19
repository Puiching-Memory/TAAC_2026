from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from taac2026.infrastructure.nn.pooling import masked_mean


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


def build_loss_stack(data_config, model_config, train_config, data_stats, device):
	del data_config
	del model_config
	pos_weight = torch.tensor([data_stats.pos_weight], dtype=torch.float32, device=device)
	return RankingLoss(pos_weight=pos_weight, pairwise_weight=train_config.pairwise_weight), DisabledAuxiliaryLoss()


def build_optimizer_component(model, train_config):
	return torch.optim.AdamW(
		model.parameters(),
		lr=train_config.learning_rate,
		weight_decay=train_config.weight_decay,
	)


__all__ = ["build_loss_stack", "build_optimizer_component", "masked_mean"]
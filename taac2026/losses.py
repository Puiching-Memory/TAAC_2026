from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PairwiseAUCLoss(nn.Module):
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        positive_mask = labels > 0.5
        negative_mask = ~positive_mask

        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return logits.new_tensor(0.0)

        positive_scores = logits[positive_mask]
        negative_scores = logits[negative_mask]
        margins = positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0)
        return F.softplus(-margins).mean()


class BCEPairwiseLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, pairwise_weight: float) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.pairwise = PairwiseAUCLoss()
        self.pairwise_weight = min(max(pairwise_weight, 0.0), 1.0)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, labels)
        pairwise_loss = self.pairwise(logits, labels)
        return (1.0 - self.pairwise_weight) * bce_loss + self.pairwise_weight * pairwise_loss


def build_criterion(loss_name: str, pos_weight: torch.Tensor, pairwise_weight: float) -> nn.Module:
    normalized = loss_name.lower()
    if normalized == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if normalized in {"bce_pairwise", "combined_auc", "omnigenrec"}:
        return BCEPairwiseLoss(pos_weight=pos_weight, pairwise_weight=pairwise_weight)
    raise ValueError(f"Unsupported loss_name: {loss_name}")


__all__ = ["BCEPairwiseLoss", "PairwiseAUCLoss", "build_criterion"]
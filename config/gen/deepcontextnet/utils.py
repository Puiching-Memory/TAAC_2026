from __future__ import annotations

import torch
from torch import nn
from taac2026.infrastructure.nn.defaults import DisabledAuxiliaryLoss
from taac2026.infrastructure.nn.optimizers import build_hybrid_optimizer


def build_loss_stack(data_config, model_config, train_config, data_stats, device):
    del data_config
    del model_config
    del train_config
    pos_weight = torch.tensor([data_stats.pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight), DisabledAuxiliaryLoss()


def build_optimizer_component(model, train_config):
    return build_hybrid_optimizer(model, train_config)


__all__ = ["build_loss_stack", "build_optimizer_component"]
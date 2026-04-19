from __future__ import annotations

import torch
from torch import nn
from taac2026.infrastructure.nn.pooling import masked_mean


class DisabledAuxiliaryLoss:
	enabled = False
	requires_aux = False


def build_loss_stack(data_config, model_config, train_config, data_stats, device):
	del data_config
	del model_config
	del train_config
	pos_weight = torch.tensor([data_stats.pos_weight], dtype=torch.float32, device=device)
	return nn.BCEWithLogitsLoss(pos_weight=pos_weight), DisabledAuxiliaryLoss()


def build_optimizer_component(model, train_config):
	return torch.optim.AdamW(
		model.parameters(),
		lr=train_config.learning_rate,
		weight_decay=train_config.weight_decay,
	)
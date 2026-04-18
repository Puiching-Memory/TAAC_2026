from __future__ import annotations

import torch
from taac2026.infrastructure.nn.optimizers import build_hybrid_optimizer


def build_optimizer_component(model, train_config):
	return build_hybrid_optimizer(
		model,
		train_config,
		muon_lr=max(0.02, train_config.learning_rate),
	)
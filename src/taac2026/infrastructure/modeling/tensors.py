"""Tensor conversion helpers for PCVR runtime."""

from __future__ import annotations

import numpy as np
import torch


def sigmoid_probabilities_numpy(logits: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(logits.detach().float()).cpu().numpy()

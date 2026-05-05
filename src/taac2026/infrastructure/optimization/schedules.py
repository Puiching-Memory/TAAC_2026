"""Learning-rate schedule helpers."""

from __future__ import annotations

import math


def dense_lr_multiplier(
    step: int,
    *,
    scheduler_type: str,
    max_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if step <= 0:
        return 1.0
    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps
    if scheduler_type == "none" or max_steps <= 0:
        return 1.0

    decay_steps = max(1, max_steps - warmup_steps)
    decay_progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    if scheduler_type == "linear":
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - decay_progress)
    if scheduler_type == "cosine":
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_scale
    return 1.0


__all__ = ["dense_lr_multiplier"]
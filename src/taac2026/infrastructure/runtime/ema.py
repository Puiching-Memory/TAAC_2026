"""Exponential moving average helpers for model state dicts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn


def _clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state_dict.items()}


@dataclass(slots=True)
class ExponentialMovingAverage:
    decay: float
    start_step: int = 0
    update_every_n_steps: int = 1
    shadow: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("ema decay must be in [0.0, 1.0)")
        if self.start_step < 0:
            raise ValueError("ema start_step must be non-negative")
        if self.update_every_n_steps < 1:
            raise ValueError("ema update_every_n_steps must be positive")

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        decay: float,
        start_step: int = 0,
        update_every_n_steps: int = 1,
    ) -> ExponentialMovingAverage:
        ema = cls(
            decay=decay,
            start_step=start_step,
            update_every_n_steps=update_every_n_steps,
        )
        ema.copy_from(model)
        return ema

    def copy_from(self, model: nn.Module) -> None:
        self.shadow = _clone_state_dict(model.state_dict())

    def update(self, model: nn.Module, *, step: int) -> bool:
        if step < self.start_step:
            self.copy_from(model)
            return False
        if (step - self.start_step) % self.update_every_n_steps != 0:
            return False

        current_state = model.state_dict()
        if set(self.shadow) != set(current_state):
            self.copy_from(model)
            return True

        with torch.no_grad():
            for key, current_value in current_state.items():
                current = current_value.detach()
                shadow = self.shadow[key]
                if (
                    shadow.shape != current.shape
                    or shadow.dtype != current.dtype
                    or shadow.device != current.device
                ):
                    self.shadow[key] = current.clone()
                    continue
                if torch.is_floating_point(current):
                    shadow.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
                else:
                    shadow.copy_(current)
        return True

    def state_dict(self) -> dict[str, torch.Tensor]:
        return _clone_state_dict(self.shadow)

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        original_state = _clone_state_dict(model.state_dict())
        model.load_state_dict(self.state_dict(), strict=True)
        try:
            yield
        finally:
            model.load_state_dict(original_state, strict=True)


__all__ = ["ExponentialMovingAverage"]

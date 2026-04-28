"""Experiment plugin contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taac2026.domain.config import EvalRequest, InferRequest, TrainRequest


TrainFn = Callable[[TrainRequest], Mapping[str, Any] | None]
EvalFn = Callable[[EvalRequest], Mapping[str, Any]]
InferFn = Callable[[InferRequest], Mapping[str, Any]]


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    package_dir: Path | None = None
    train_fn: TrainFn | None = None
    evaluate_fn: EvalFn | None = None
    infer_fn: InferFn | None = None
    train_defaults: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def train(self, request: TrainRequest) -> Mapping[str, Any] | None:
        if self.train_fn is None:
            raise NotImplementedError(f"experiment {self.name!r} does not implement training")
        return self.train_fn(request)

    def evaluate(self, request: EvalRequest) -> Mapping[str, Any]:
        if self.evaluate_fn is None:
            raise NotImplementedError(f"experiment {self.name!r} does not implement evaluation")
        return self.evaluate_fn(request)

    def infer(self, request: InferRequest) -> Mapping[str, Any]:
        if self.infer_fn is None:
            raise NotImplementedError(f"experiment {self.name!r} does not implement inference")
        return self.infer_fn(request)

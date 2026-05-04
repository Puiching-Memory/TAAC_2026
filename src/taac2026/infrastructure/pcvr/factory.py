"""Factory helpers for PCVR experiment packages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from taac2026.infrastructure.pcvr.config import PCVRTrainConfig
from taac2026.infrastructure.pcvr.experiment import PCVRExperiment
from taac2026.infrastructure.pcvr.prediction_stack import (
    PCVRPredictionHooks,
    build_pcvr_prediction_hooks,
)
from taac2026.infrastructure.pcvr.runtime_stack import (
    PCVRRuntimeHooks,
    build_pcvr_runtime_hooks,
)
from taac2026.infrastructure.pcvr.train_stack import (
    PCVRTrainHooks,
    build_pcvr_train_hooks,
)
from taac2026.infrastructure.pcvr.training import parse_pcvr_train_args


def create_pcvr_experiment(
    *,
    name: str,
    package_dir: Path,
    model_class_name: str,
    train_defaults: PCVRTrainConfig,
    train_arg_parser: Callable[[Sequence[str] | None], Any] = parse_pcvr_train_args,
    train_hook_overrides: Mapping[str, Any] | None = None,
    prediction_hook_overrides: Mapping[str, Any] | None = None,
    runtime_hook_overrides: Mapping[str, Any] | None = None,
    train_hooks: PCVRTrainHooks | None = None,
    prediction_hooks: PCVRPredictionHooks | None = None,
    runtime_hooks: PCVRRuntimeHooks | None = None,
) -> PCVRExperiment:
    """Create a PCVR experiment with default hooks and optional overrides."""

    resolved_train_hooks = train_hooks or build_pcvr_train_hooks(**dict(train_hook_overrides or {}))
    resolved_prediction_hooks = prediction_hooks or build_pcvr_prediction_hooks(**dict(prediction_hook_overrides or {}))
    resolved_runtime_hooks = runtime_hooks or build_pcvr_runtime_hooks(**dict(runtime_hook_overrides or {}))
    return PCVRExperiment(
        name=name,
        package_dir=package_dir,
        model_class_name=model_class_name,
        train_defaults=train_defaults,
        train_arg_parser=train_arg_parser,
        train_hooks=resolved_train_hooks,
        prediction_hooks=resolved_prediction_hooks,
        runtime_hooks=resolved_runtime_hooks,
    )


__all__ = ["create_pcvr_experiment"]

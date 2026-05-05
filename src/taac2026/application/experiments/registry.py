"""Load experiment packages from module names or filesystem paths."""

from __future__ import annotations

import importlib
from pathlib import Path

from taac2026.domain.experiment import ExperimentSpec
from taac2026.infrastructure.experiments.module_loader import load_module_from_path
from taac2026.infrastructure.io.files import repo_root


def _coerce_experiment(value: object, source: str) -> ExperimentSpec:
    if isinstance(value, ExperimentSpec):
        return value
    required = ("name", "train", "evaluate", "infer")
    if all(hasattr(value, attribute) for attribute in required):
        return ExperimentSpec(
            name=str(value.name),
            package_dir=getattr(value, "package_dir", None),
            train_fn=value.train,
            evaluate_fn=value.evaluate,
            infer_fn=value.infer,
            train_defaults=getattr(value, "train_defaults", None),
            metadata=dict(getattr(value, "metadata", {})),
        )
    raise TypeError(f"EXPERIMENT in {source} is not a supported experiment object")


def _path_from_user_value(value: str) -> Path | None:
    candidate = Path(value)
    if candidate.exists():
        return candidate
    root_candidate = repo_root() / value
    if root_candidate.exists():
        return root_candidate
    return None


def load_experiment_package(value: str | Path) -> ExperimentSpec:
    source = str(value)
    if isinstance(value, Path):
        module = load_module_from_path(value)
    else:
        path = _path_from_user_value(value)
        if path is not None or "/" in value or value.startswith("."):
            if path is None:
                raise FileNotFoundError(f"experiment package path not found: {value}")
            module = load_module_from_path(path)
        else:
            module_name = value.replace("/", ".")
            module = importlib.import_module(module_name)

    if not hasattr(module, "EXPERIMENT"):
        raise AttributeError(f"experiment package {source!r} does not define EXPERIMENT")
    return _coerce_experiment(module.EXPERIMENT, source)

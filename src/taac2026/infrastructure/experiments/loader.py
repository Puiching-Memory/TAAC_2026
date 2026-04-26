"""Load experiment packages from module names or filesystem paths."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from taac2026.domain.experiment import ExperimentSpec
from taac2026.infrastructure.io.files import repo_root, stable_hash64


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
            default_train_args=tuple(getattr(value, "default_train_args", ())),
            metadata=dict(getattr(value, "metadata", {})),
        )
    raise TypeError(f"EXPERIMENT in {source} is not a supported experiment object")


def _load_path_module(path: Path) -> ModuleType:
    resolved_path = path.expanduser().resolve()
    if resolved_path.is_dir():
        init_path = resolved_path / "__init__.py"
        if not init_path.exists():
            raise FileNotFoundError(f"experiment directory lacks __init__.py: {resolved_path}")
        module_name = f"taac2026_dynamic_experiment_{stable_hash64(str(resolved_path))}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_path,
            submodule_search_locations=[str(resolved_path)],
        )
    else:
        module_name = f"taac2026_dynamic_experiment_{stable_hash64(str(resolved_path))}"
        spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experiment package from {resolved_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
        module = _load_path_module(value)
    else:
        path = _path_from_user_value(value)
        if path is not None or "/" in value or value.startswith("."):
            if path is None:
                raise FileNotFoundError(f"experiment package path not found: {value}")
            module = _load_path_module(path)
        else:
            module_name = value.replace("/", ".")
            module = importlib.import_module(module_name)

    if not hasattr(module, "EXPERIMENT"):
        raise AttributeError(f"experiment package {source!r} does not define EXPERIMENT")
    return _coerce_experiment(module.EXPERIMENT, source)

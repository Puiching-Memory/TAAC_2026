from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from taac2026.infrastructure.experiments.discovery import discover_experiment_paths
from taac2026.infrastructure.experiments.loader import load_experiment_package


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "pcvr"


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    path: str
    module: str
    name: str
    model_class: str
    package_dir: Path


def build_pcvr_experiment_cases(experiment_root: Path) -> tuple[ExperimentCase, ...]:
    root = experiment_root.expanduser().resolve()
    repo_root = root.parent.parent if root.parent.name == "experiments" else root.parent
    cases: list[ExperimentCase] = []
    for experiment_path in discover_experiment_paths(root):
        experiment = load_experiment_package(repo_root / experiment_path)
        if experiment.metadata.get("kind") != "pcvr":
            continue
        model_class = experiment.metadata.get("model_class")
        if not isinstance(model_class, str) or not model_class:
            raise AssertionError(f"experiment {experiment_path!r} is missing metadata['model_class']")
        package_dir = experiment.package_dir.resolve() if experiment.package_dir is not None else (repo_root / experiment_path).resolve()
        cases.append(
            ExperimentCase(
                path=experiment_path,
                module=experiment_path.replace("/", "."),
                name=experiment.name,
                model_class=model_class,
                package_dir=package_dir,
            )
        )
    return tuple(cases)


@lru_cache(maxsize=1)
def discover_pcvr_experiment_cases() -> tuple[ExperimentCase, ...]:
    return build_pcvr_experiment_cases(EXPERIMENT_ROOT)


def discover_nonbaseline_pcvr_experiment_paths() -> tuple[str, ...]:
    return tuple(case.path for case in discover_pcvr_experiment_cases() if case.path != "experiments/pcvr/baseline")


def get_experiment_case(experiment_path: str) -> ExperimentCase:
    for case in discover_pcvr_experiment_cases():
        if case.path == experiment_path:
            return case
    raise KeyError(f"unknown experiment path: {experiment_path}")


def load_model_module(experiment_case: ExperimentCase):
    model_path = experiment_case.package_dir / "model.py"
    spec = importlib.util.spec_from_file_location(experiment_case.module + ".model", model_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

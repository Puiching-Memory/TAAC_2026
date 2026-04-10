from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from taac2026.infrastructure.experiments.loader import load_experiment_package
from tests.support import TestWorkspace, create_test_workspace, prepare_experiment

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


@pytest.mark.parametrize(
    "module_path",
    [
        "config.gen.baseline",
        "config.gen.grok",
        "config.gen.ctr_baseline",
        "config.gen.deepcontextnet",
        "config.gen.interformer",
        "config.gen.onetrans",
        "config.gen.hyformer",
        "config.gen.unirec",
        "config.gen.uniscaleformer",
        "config.gen.oo",
    ],
)
def test_experiment_package_builds_and_runs_forward(module_path: str, test_workspace: TestWorkspace) -> None:
    experiment = importlib.import_module(module_path).EXPERIMENT
    experiment = prepare_experiment(experiment, test_workspace)

    train_loader, _, data_stats = experiment.build_data_pipeline(
        experiment.data,
        experiment.model,
        experiment.train,
    )
    batch = next(iter(train_loader))
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    logits = model(batch)

    assert logits.shape == batch.labels.shape
    assert torch.isfinite(logits).all().item()


@pytest.mark.parametrize(
    "module_path",
    [
        "config.gen.baseline",
        "config.gen.grok",
        "config.gen.ctr_baseline",
        "config.gen.deepcontextnet",
        "config.gen.interformer",
        "config.gen.onetrans",
        "config.gen.hyformer",
        "config.gen.unirec",
        "config.gen.uniscaleformer",
        "config.gen.oo",
    ],
)
def test_experiment_package_owns_its_data_pipeline(module_path: str) -> None:
    experiment = importlib.import_module(module_path).EXPERIMENT

    assert experiment.build_data_pipeline.__module__ == f"{module_path}.data"


@pytest.mark.parametrize(
    "experiment_path",
    [
        "config/gen/baseline",
        "config/gen/grok",
        "config/gen/ctr_baseline",
        "config/gen/deepcontextnet",
        "config/gen/unirec",
        "config/gen/uniscaleformer",
    ],
)
def test_experiment_package_directory_path_loads_namespace_relative_imports(experiment_path: str) -> None:
    experiment = load_experiment_package(experiment_path)

    assert experiment.name

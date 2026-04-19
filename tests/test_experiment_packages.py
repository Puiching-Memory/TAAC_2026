from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.io.datasets import iter_dataset_rows, resolve_parquet_dataset_path
from taac2026.infrastructure.nn.defaults import resolve_experiment_builders
from tests.support import TestWorkspace, create_test_workspace, prepare_experiment


LEGACY_SEQUENCE_FIELD_NAMES = (
    "history_tokens",
    "history_mask",
    "history_post_tokens",
    "history_author_tokens",
    "history_action_tokens",
    "history_time_gap",
    "history_group_ids",
    "sequence_tokens",
    "sequence_mask",
)


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


@pytest.mark.parametrize(
    "module_path",
    [
        "config.baseline",
        "config.grok",
        "config.ctr_baseline",
        "config.deepcontextnet",
        "config.interformer",
        "config.onetrans",
        "config.hyformer",
        "config.unirec",
        "config.uniscaleformer",
        "config.oo",
    ],
)
def test_experiment_package_builds_and_runs_forward(module_path: str, test_workspace: TestWorkspace) -> None:
    experiment = importlib.import_module(module_path).EXPERIMENT
    experiment = prepare_experiment(experiment, test_workspace)
    builders = resolve_experiment_builders(experiment)

    train_loader, _, data_stats = builders.build_data_pipeline(
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
        "config.baseline",
        "config.ctr_baseline",
        "config.onetrans",
        "config.hyformer",
        "config.deepcontextnet",
        "config.grok",
        "config.interformer",
        "config.uniscaleformer",
        "config.oo",
        "config.unirec",
    ],
)
def test_models_with_pooled_sparse_branches_prefer_torchrec_sparse_features(
    module_path: str,
    test_workspace: TestWorkspace,
) -> None:
    experiment = importlib.import_module(module_path).EXPERIMENT
    experiment = prepare_experiment(experiment, test_workspace)
    builders = resolve_experiment_builders(experiment)

    train_loader, _, data_stats = builders.build_data_pipeline(
        experiment.data,
        experiment.model,
        experiment.train,
    )
    batch = next(iter(train_loader))
    assert batch.sparse_features is not None

    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    model.eval()

    with torch.inference_mode():
        actual_logits = model(batch)

    assert not hasattr(batch, "candidate_tokens")
    assert not hasattr(batch, "context_tokens")
    assert not hasattr(batch, "user_tokens")
    assert actual_logits.shape == batch.labels.shape


def test_interformer_uses_shared_lookup_modules_without_legacy_nn_embedding(test_workspace: TestWorkspace) -> None:
    experiment = importlib.import_module("config.interformer").EXPERIMENT
    experiment = prepare_experiment(experiment, test_workspace)
    builders = resolve_experiment_builders(experiment)

    _, _, data_stats = builders.build_data_pipeline(
        experiment.data,
        experiment.model,
        experiment.train,
    )
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)

    assert not hasattr(model, "token_embedding")
    assert not hasattr(model, "time_gap_embedding")
    assert not hasattr(model, "history_group_embedding")
    assert hasattr(model, "sparse_embedding")
    assert hasattr(model, "sequence_embedding")


@pytest.mark.parametrize(
    "module_path",
    [
        "config.baseline",
        "config.ctr_baseline",
        "config.deepcontextnet",
        "config.grok",
        "config.interformer",
        "config.oo",
        "config.onetrans",
        "config.hyformer",
        "config.unirec",
        "config.uniscaleformer",
    ],
)
def test_models_with_sequence_branches_prefer_torchrec_sequence_features(
    module_path: str,
    test_workspace: TestWorkspace,
) -> None:
    experiment = importlib.import_module(module_path).EXPERIMENT
    experiment = prepare_experiment(experiment, test_workspace)
    builders = resolve_experiment_builders(experiment)

    train_loader, _, data_stats = builders.build_data_pipeline(
        experiment.data,
        experiment.model,
        experiment.train,
    )
    batch = next(iter(train_loader))
    assert batch.sequence_features is not None
    for field_name in LEGACY_SEQUENCE_FIELD_NAMES:
        assert not hasattr(batch, field_name)

    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    model.eval()

    with torch.inference_mode():
        actual_logits = model(batch)

    assert actual_logits.shape == batch.labels.shape
    assert torch.isfinite(actual_logits).all().item()


@pytest.mark.parametrize(
    "module_path",
    [
        "config.baseline",
        "config.grok",
        "config.ctr_baseline",
        "config.deepcontextnet",
        "config.interformer",
        "config.onetrans",
        "config.hyformer",
        "config.unirec",
        "config.uniscaleformer",
        "config.oo",
    ],
)
def test_experiment_package_owns_its_data_pipeline(module_path: str) -> None:
    experiment = importlib.import_module(module_path).EXPERIMENT

    assert experiment.build_data_pipeline is None


@pytest.mark.parametrize(
    ("module_path", "uses_default_optimizer"),
    [
        ("config.baseline", True),
        ("config.grok", True),
        ("config.ctr_baseline", True),
        ("config.deepcontextnet", True),
        ("config.interformer", True),
        ("config.onetrans", True),
        ("config.hyformer", True),
        ("config.unirec", False),
        ("config.uniscaleformer", True),
        ("config.oo", True),
    ],
)
def test_experiment_package_uses_framework_defaults_when_builder_is_omitted(
    module_path: str,
    uses_default_optimizer: bool,
) -> None:
    experiment = importlib.import_module(module_path).EXPERIMENT
    builders = resolve_experiment_builders(experiment)

    assert builders.build_loss_stack.__module__ == "taac2026.infrastructure.nn.defaults"
    if uses_default_optimizer:
        assert builders.build_optimizer_component.__module__ == "taac2026.infrastructure.nn.defaults"
    else:
        assert builders.build_optimizer_component.__module__ == f"{module_path}.utils"


@pytest.mark.parametrize(
    "module_path",
    [
        "config.baseline",
        "config.grok",
        "config.ctr_baseline",
        "config.deepcontextnet",
        "config.interformer",
        "config.onetrans",
        "config.hyformer",
        "config.unirec",
        "config.uniscaleformer",
        "config.oo",
    ],
)
def test_experiment_package_exposes_feature_schema(module_path: str) -> None:
    experiment = importlib.import_module(module_path).EXPERIMENT

    assert experiment.feature_schema is not None
    assert experiment.feature_schema.dense_dim == experiment.data.dense_feature_dim
    assert experiment.feature_schema.sequence_names == experiment.data.sequence_names
    assert "candidate_tokens" in experiment.feature_schema.table_names


@pytest.mark.parametrize(
    "experiment_path",
    [
        "config/baseline",
        "config/grok",
        "config/ctr_baseline",
        "config/deepcontextnet",
        "config/unirec",
        "config/uniscaleformer",
    ],
)
def test_experiment_package_directory_path_loads_namespace_relative_imports(experiment_path: str) -> None:
    experiment = load_experiment_package(experiment_path)

    assert experiment.name


@pytest.mark.parametrize(
    "module_path",
    [
        "config.baseline",
        "config.grok",
        "config.ctr_baseline",
        "config.deepcontextnet",
        "config.interformer",
        "config.onetrans",
        "config.hyformer",
        "config.unirec",
        "config.uniscaleformer",
        "config.oo",
    ],
)
def test_experiment_package_default_dataset_points_to_hf_cache_root(module_path: str) -> None:
    experiment = importlib.import_module(module_path).EXPERIMENT
    dataset_path = Path(experiment.data.dataset_path)

    assert dataset_path.name == "datasets--TAAC2026--data_sample_1000"
    assert "snapshots" not in dataset_path.parts


def test_resolve_parquet_dataset_path_prefers_hf_main_ref(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets--TAAC2026--data_sample_1000"
    old_snapshot = dataset_root / "snapshots" / "old-revision"
    new_snapshot = dataset_root / "snapshots" / "new-revision"
    (dataset_root / "refs").mkdir(parents=True)
    old_snapshot.mkdir(parents=True)
    new_snapshot.mkdir(parents=True)
    (dataset_root / "refs" / "main").write_text("new-revision\n", encoding="utf-8")
    (old_snapshot / "sample_data.parquet").touch()
    preferred = new_snapshot / "sample_data.parquet"
    preferred.touch()

    assert resolve_parquet_dataset_path(dataset_root) == preferred


def test_resolve_parquet_dataset_path_falls_back_to_recursive_directory_search(tmp_path: Path) -> None:
    dataset_root = tmp_path / "custom_dataset"
    dataset_root.mkdir(parents=True)
    candidate = dataset_root / "nested" / "sample_data.parquet"
    candidate.parent.mkdir(parents=True)
    candidate.touch()

    assert resolve_parquet_dataset_path(dataset_root) == candidate


def test_iter_dataset_rows_missing_cache_root_downloads_from_hf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "datasets--TAAC2026--data_sample_1000"
    captured: dict[str, str | None] = {}

    def fake_load_dataset(path: str, *, split: str, cache_dir: str | None = None, data_files: str | None = None):
        captured["path"] = path
        captured["split"] = split
        captured["cache_dir"] = cache_dir
        captured["data_files"] = data_files
        return [{"ok": True}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    rows = iter_dataset_rows(dataset_root)

    assert rows == [{"ok": True}]
    assert captured == {
        "path": "TAAC2026/data_sample_1000",
        "split": "train",
        "cache_dir": str(dataset_root.parent),
        "data_files": None,
    }

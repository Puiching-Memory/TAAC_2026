from __future__ import annotations

from pathlib import Path

import pytest

from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.experiments.payload import apply_serialized_experiment, serialize_experiment
from tests.support import TestWorkspace, create_test_workspace


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


def test_payload_round_trip_restores_sequence_names_and_null_switches(test_workspace: TestWorkspace) -> None:
    experiment = load_experiment_package(test_workspace.write_experiment_package())
    experiment.data.sequence_names = ("action_seq", "bonus_seq")
    experiment.refresh_feature_schema()
    payload = serialize_experiment(experiment)
    payload["data"]["sequence_names"] = list(payload["data"]["sequence_names"])
    payload["train"]["switches"] = None
    payload["switches"] = None

    restored = apply_serialized_experiment(experiment, payload)

    assert restored.data.sequence_names == ("action_seq", "bonus_seq")
    assert restored.train.switches == {}
    assert restored.switches == {}
    assert restored.feature_schema is not None
    assert restored.feature_schema.sequence_names == ("action_seq", "bonus_seq")


def test_apply_serialized_experiment_requires_all_sections(test_workspace: TestWorkspace) -> None:
    experiment = load_experiment_package(test_workspace.write_experiment_package())
    payload = serialize_experiment(experiment)
    del payload["model"]

    with pytest.raises(KeyError):
        apply_serialized_experiment(experiment, payload)


def test_apply_serialized_experiment_rejects_unknown_fields(test_workspace: TestWorkspace) -> None:
    experiment = load_experiment_package(test_workspace.write_experiment_package())
    payload = serialize_experiment(experiment)
    payload["data"]["unexpected"] = 123

    with pytest.raises(TypeError):
        apply_serialized_experiment(experiment, payload)


def test_load_experiment_package_requires_experiment_symbol(tmp_path: Path) -> None:
    package_path = tmp_path / "missing_experiment"
    package_path.mkdir()
    (package_path / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(AttributeError, match="does not define EXPERIMENT"):
        load_experiment_package(package_path)


def test_load_experiment_package_propagates_import_failures(tmp_path: Path) -> None:
    package_path = tmp_path / "broken_experiment"
    package_path.mkdir()
    (package_path / "__init__.py").write_text("raise RuntimeError('boom from import')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom from import"):
        load_experiment_package(package_path)


def test_load_experiment_package_rejects_unknown_module_path() -> None:
    with pytest.raises(ModuleNotFoundError):
        load_experiment_package("config.this_package_does_not_exist")

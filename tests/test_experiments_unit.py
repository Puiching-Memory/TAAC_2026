from __future__ import annotations

from pathlib import Path

import pytest

from taac2026.domain.config import DataConfig, ModelConfig, SearchConfig, TrainConfig
from taac2026.domain.experiment import ExperimentSpec
from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.experiments.payload import apply_serialized_experiment, serialize_experiment


pytestmark = pytest.mark.unit


def _base_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        name="base",
        data=DataConfig(dataset_path="sample.parquet", sequence_names=("action_seq", "item_seq")),
        model=ModelConfig(name="base", vocab_size=64, embedding_dim=8, hidden_dim=8),
        train=TrainConfig(output_dir="outputs/base", switches={"logging": False}),
        search=SearchConfig(metric_name="metrics.auc"),
        build_data_pipeline=lambda *_: None,
        build_model_component=lambda *_: None,
        build_loss_stack=lambda *_: None,
        build_optimizer_component=lambda *_: None,
        switches={"visualization": True},
    )


def test_serialize_and_apply_experiment_round_trip_restores_tuple_fields() -> None:
    experiment = _base_experiment()

    payload = serialize_experiment(experiment)
    payload["data"]["sequence_names"] = list(payload["data"]["sequence_names"])
    restored = apply_serialized_experiment(_base_experiment(), payload)

    assert restored.name == experiment.name
    assert restored.data.sequence_names == ("action_seq", "item_seq")
    assert restored.train.switches == {"logging": False}
    assert restored.switches == {"visualization": True}
    assert restored.search.metric_name == "metrics.auc"


@pytest.mark.fault
def test_apply_serialized_experiment_rejects_missing_sections() -> None:
    payload = serialize_experiment(_base_experiment())
    del payload["model"]

    with pytest.raises(KeyError):
        apply_serialized_experiment(_base_experiment(), payload)


@pytest.mark.fault
def test_load_experiment_package_requires_exported_experiment(tmp_path: Path) -> None:
    package_path = tmp_path / "broken_package"
    package_path.mkdir()
    (package_path / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(AttributeError, match="does not define EXPERIMENT"):
        load_experiment_package(package_path)


@pytest.mark.fault
def test_load_experiment_package_bubbles_import_errors_for_missing_modules() -> None:
    with pytest.raises(ModuleNotFoundError):
        load_experiment_package("config.gen.this_package_does_not_exist")

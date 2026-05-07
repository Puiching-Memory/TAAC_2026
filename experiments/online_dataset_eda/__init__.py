"""Experiment package for bundle-friendly online dataset EDA."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from taac2026.domain.requests import TrainRequest
from taac2026.domain.experiment import ExperimentSpec

from .runner import OnlineDatasetEDAConfig, run_online_dataset_eda


PACKAGE_DIR = Path(__file__).resolve().parent
ONLINE_DATASET_EDA_CONFIG = OnlineDatasetEDAConfig()


def _resolved_online_dataset_eda_config(request: TrainRequest) -> OnlineDatasetEDAConfig:
    if request.dataset_path is None:
        raise ValueError("online_dataset_eda experiment requires dataset_path")
    return replace(
        ONLINE_DATASET_EDA_CONFIG,
        dataset_path=request.dataset_path.expanduser().resolve(),
        schema_path=request.schema_path.expanduser().resolve() if request.schema_path is not None else None,
    )


def _train(request: TrainRequest) -> dict[str, object]:
    if request.dataset_path is None:
        raise ValueError("online_dataset_eda experiment requires dataset_path")
    run_dir = request.run_dir.expanduser().resolve()
    if request.extra_args:
        raise ValueError(
            "online_dataset_eda experiment does not accept extra_args; edit ONLINE_DATASET_EDA_CONFIG in experiments/online_dataset_eda/__init__.py"
        )
    report = run_online_dataset_eda(_resolved_online_dataset_eda_config(request))
    return {
        "experiment_name": "online_dataset_eda",
        "run_dir": str(run_dir),
        "dataset_role": report.get("dataset_role"),
        "row_count": report.get("row_count"),
        "sampled": report.get("sampled"),
    }


EXPERIMENT = ExperimentSpec(
    name="online_dataset_eda",
    package_dir=PACKAGE_DIR,
    train_fn=_train,
    metadata={
        "kind": "maintenance",
        "task": "online_dataset_eda",
        "requires_dataset": True,
    },
)

__all__ = ["EXPERIMENT"]

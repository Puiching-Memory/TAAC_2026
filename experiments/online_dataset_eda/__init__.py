"""Experiment package for bundle-friendly online dataset EDA."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from taac2026.domain.requests import InferRequest
from taac2026.domain.requests import TrainRequest
from taac2026.domain.experiment import ExperimentSpec

from .runner import OnlineDatasetEDAConfig, run_online_dataset_eda


PACKAGE_DIR = Path(__file__).resolve().parent
ONLINE_DATASET_EDA_CONFIG = OnlineDatasetEDAConfig()


def _resolved_reference_profile_path(checkpoint_path: Path | None) -> Path | None:
    if checkpoint_path is None:
        return None
    return checkpoint_path.expanduser().resolve()


def _resolved_online_dataset_eda_config(
    *,
    dataset_path: Path | None,
    schema_path: Path | None,
    output_dir: Path,
    dataset_role: str,
    reference_profile_path: Path | None = None,
) -> OnlineDatasetEDAConfig:
    if dataset_path is None:
        raise ValueError("online_dataset_eda experiment requires dataset_path")
    return replace(
        ONLINE_DATASET_EDA_CONFIG,
        dataset_path=dataset_path.expanduser().resolve(),
        schema_path=schema_path.expanduser().resolve() if schema_path is not None else None,
        output_dir=output_dir.expanduser().resolve(),
        dataset_role=dataset_role,
        reference_profile_path=reference_profile_path,
    )


def _train(request: TrainRequest) -> dict[str, object]:
    if request.dataset_path is None:
        raise ValueError("online_dataset_eda experiment requires dataset_path")
    run_dir = request.run_dir.expanduser().resolve()
    if request.extra_args:
        raise ValueError(
            "online_dataset_eda experiment does not accept extra_args; edit ONLINE_DATASET_EDA_CONFIG in experiments/online_dataset_eda/__init__.py"
        )
    report = run_online_dataset_eda(
        _resolved_online_dataset_eda_config(
            dataset_path=request.dataset_path,
            schema_path=request.schema_path,
            output_dir=run_dir,
            dataset_role="train",
        )
    )
    return {
        "experiment_name": "online_dataset_eda",
        "run_dir": str(run_dir),
        "dataset_role": report.get("dataset_role"),
        "row_count": report.get("row_count"),
        "sampled": report.get("sampled"),
        "stdout_result": True,
    }


def _infer(request: InferRequest) -> dict[str, object]:
    if request.dataset_path is None:
        raise ValueError("online_dataset_eda experiment requires dataset_path")
    result_dir = request.result_dir.expanduser().resolve()
    report = run_online_dataset_eda(
        _resolved_online_dataset_eda_config(
            dataset_path=request.dataset_path,
            schema_path=request.schema_path,
            output_dir=result_dir,
            dataset_role="infer",
            reference_profile_path=_resolved_reference_profile_path(request.checkpoint_path),
        )
    )
    return {
        "experiment_name": "online_dataset_eda",
        "result_dir": str(result_dir),
        "dataset_role": report.get("dataset_role"),
        "row_count": report.get("row_count"),
        "sampled": report.get("sampled"),
        "reference_profile_path": report.get("reference_profile_path"),
        "risk_flags": (report.get("comparison") or {}).get("risk_flags") if isinstance(report.get("comparison"), dict) else None,
        "stdout_result": True,
    }


EXPERIMENT = ExperimentSpec(
    name="online_dataset_eda",
    package_dir=PACKAGE_DIR,
    train_fn=_train,
    infer_fn=_infer,
    metadata={
        "kind": "maintenance",
        "task": "online_dataset_eda",
        "requires_dataset": True,
        "supports_infer": True,
    },
)

__all__ = ["EXPERIMENT"]

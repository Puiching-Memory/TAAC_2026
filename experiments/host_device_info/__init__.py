"""Experiment package for collecting host and device diagnostics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from taac2026.domain.requests import TrainRequest
from taac2026.domain.experiment import ExperimentSpec

from .runner import HostDeviceInfoConfig, collect_host_device_info


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent.parent
HOST_DEVICE_INFO_CONFIG = HostDeviceInfoConfig(repo_root=PROJECT_ROOT)


def _resolved_host_device_info_config() -> HostDeviceInfoConfig:
    return replace(
        HOST_DEVICE_INFO_CONFIG,
        repo_root=PROJECT_ROOT,
        site_probe_targets=dict(HOST_DEVICE_INFO_CONFIG.site_probe_targets),
    )


def _train(request: TrainRequest) -> dict[str, object]:
    run_dir = request.run_dir.expanduser().resolve()
    if request.extra_args:
        raise ValueError(
            "host_device_info experiment does not accept extra_args; edit HOST_DEVICE_INFO_CONFIG in experiments/host_device_info/__init__.py"
        )
    summary = collect_host_device_info(_resolved_host_device_info_config())
    return {
        "experiment_name": "host_device_info",
        "run_dir": str(run_dir),
        **summary,
    }


EXPERIMENT = ExperimentSpec(
    name="host_device_info",
    package_dir=PACKAGE_DIR,
    train_fn=_train,
    metadata={
        "kind": "maintenance",
        "task": "host_device_info",
        "requires_dataset": False,
    },
)

__all__ = ["EXPERIMENT"]

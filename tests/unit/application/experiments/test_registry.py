from __future__ import annotations

from taac2026.application.experiments.registry import load_experiment_package


def test_load_baseline_experiment_from_path() -> None:
    experiment = load_experiment_package("experiments/baseline")

    assert experiment.name == "pcvr_hyformer"
    assert experiment.package_dir is not None

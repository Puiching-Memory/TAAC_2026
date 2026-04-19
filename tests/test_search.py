from __future__ import annotations

from pathlib import Path

import pytest

from taac2026.application.search.cli import _format_search_report, parse_args
from taac2026.application.search.service import run_search
from taac2026.infrastructure.compute.device_scheduler import GpuDevice, launchable_devices, parse_gpu_indices
from taac2026.infrastructure.experiments.loader import load_experiment_package
from tests.support import TestWorkspace, create_test_workspace


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


def test_parse_args_accepts_budget_overrides() -> None:
    args = parse_args(
        [
            "--experiment",
            "config/baseline",
            "--study-dir",
            "outputs/search/baseline",
            "--trials",
            "12",
            "--timeout-seconds",
            "900",
            "--metric-name",
            "metrics.auc",
            "--direction",
            "maximize",
            "--max-parameter-gb",
            "3.0",
            "--max-end-to-end-inference-seconds",
            "180",
            "--seed",
            "11",
            "--scheduler",
            "auto",
            "--gpu-indices",
            "0,2,5",
            "--min-free-memory-gb",
            "18",
            "--max-jobs-per-gpu",
            "3",
            "--poll-interval-seconds",
            "2.5",
            "--json",
        ]
    )

    assert args.experiment == "config/baseline"
    assert args.study_dir == "outputs/search/baseline"
    assert args.trials == 12
    assert args.timeout_seconds == 900
    assert args.metric_name == "metrics.auc"
    assert args.direction == "maximize"
    assert args.max_parameter_gb == 3.0
    assert args.max_end_to_end_inference_seconds == 180.0
    assert args.seed == 11
    assert args.scheduler == "auto"
    assert args.gpu_indices == "0,2,5"
    assert args.min_free_memory_gb == 18.0
    assert args.max_jobs_per_gpu == 3
    assert args.poll_interval_seconds == 2.5
    assert args.json is True


def test_run_search_writes_study_artifacts(test_workspace: TestWorkspace) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)
    experiment.search.n_trials = 3
    experiment.search.timeout_seconds = 60
    experiment.search.sampler_seed = 7
    study_dir = test_workspace.root / "optuna_study"

    report = run_search(
        experiment,
        experiment_path=experiment_path,
        study_dir=study_dir,
    )

    assert report["best_trial"] is not None
    assert report["best_trial"]["final_budget_status"]["constraints_met"] is True
    assert report["trial_state_counts"]["COMPLETE"] == 3
    assert "PRUNED" not in report["trial_state_counts"]
    assert Path(report["best_trial"]["summary_path"]).exists()
    assert (study_dir / "study_summary.json").exists()
    assert (study_dir / "best_experiment.json").exists()
    for trial in report["trials"]:
        assert "dynamic value space" not in str(trial["user_attrs"].get("trial_error", ""))


def test_run_search_prunes_trials_that_violate_budget(test_workspace: TestWorkspace) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)
    experiment.search.n_trials = 1
    experiment.search.timeout_seconds = 60
    experiment.search.max_parameter_bytes = 1
    study_dir = test_workspace.root / "budget_failure"

    report = run_search(
        experiment,
        experiment_path=experiment_path,
        study_dir=study_dir,
    )

    assert report["best_trial"] is None
    assert report["trial_state_counts"]["PRUNED"] == 1
    assert (study_dir / "study_summary.json").exists()


def test_run_search_auto_falls_back_to_sequential_when_no_gpu(test_workspace: TestWorkspace, monkeypatch) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)
    experiment.search.n_trials = 1
    experiment.search.timeout_seconds = 60
    study_dir = test_workspace.root / "auto_fallback"
    monkeypatch.setattr("taac2026.application.search.service.query_gpu_devices", lambda gpu_indices=None: [])

    report = run_search(
        experiment,
        experiment_path=experiment_path,
        study_dir=study_dir,
        scheduler="auto",
    )

    assert report["scheduler"]["used_mode"] == "sequential"
    assert "no visible GPUs" in report["scheduler"]["fallback_reason"]
    assert report["trial_state_counts"]["COMPLETE"] == 1


def test_format_search_report_is_compact() -> None:
    report = {
        "experiment_name": "demo",
        "study_dir": "outputs/demo_optuna",
        "study_summary_path": "outputs/demo_optuna/study_summary.json",
        "best_experiment_path": "outputs/demo_optuna/best_experiment.json",
        "search": {"metric_name": "best_val_auc", "direction": "maximize"},
        "trial_state_counts": {"COMPLETE": 2, "PRUNED": 1},
        "trial_count": 3,
        "best_trial": {
            "number": 2,
            "value": 0.812345,
            "trial_dir": "outputs/demo_optuna/trial_0002",
            "summary_path": "outputs/demo_optuna/trial_0002/summary.json",
            "params": {"model.hidden_dim": 128, "train.learning_rate": 0.001},
            "final_budget_status": {
                "parameter_bytes": 1024 * 1024 * 64,
                "max_parameter_gib": 3.0,
                "estimated_end_to_end_inference_seconds": 12.5,
                "max_end_to_end_inference_seconds": 180.0,
            },
        },
        "trials": [
            {"state": "PRUNED", "user_attrs": {"prune_reason": "trial exceeds search budget before training"}},
        ],
    }

    rendered = _format_search_report(report)

    assert "Search complete" in rendered
    assert "best trial: #2" in rendered
    assert "model.hidden_dim = 128" in rendered
    assert "trial exceeds search budget before training" in rendered
    assert "'trials':" not in rendered


def test_parse_gpu_indices_handles_empty_values() -> None:
    assert parse_gpu_indices(None) is None
    assert parse_gpu_indices("") is None
    assert parse_gpu_indices("0, 2,5") == {0, 2, 5}


def test_launchable_devices_respects_memory_slots_and_running_jobs() -> None:
    devices = [
        GpuDevice(index=0, name="gpu0", total_memory_mb=81920, used_memory_mb=40960, free_memory_mb=40960),
        GpuDevice(index=1, name="gpu1", total_memory_mb=81920, used_memory_mb=66560, free_memory_mb=15360),
    ]

    launchable = launchable_devices(
        devices,
        {0: 1, 1: 0},
        min_free_memory_mb=12 * 1024,
        max_jobs_per_gpu=3,
    )

    assert [device.index for device in launchable] == [0, 0, 1]

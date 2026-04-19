from __future__ import annotations

from pathlib import Path

import pytest

from taac2026.application.search.trial import budget_status, execute_search_trial, resolve_metric
from taac2026.infrastructure.experiments.loader import load_experiment_package
from tests.support import TestWorkspace, create_test_workspace


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


def test_resolve_metric_reads_nested_paths() -> None:
    summary = {"metrics": {"gauc": {"value": 0.78}}}

    assert resolve_metric(summary, "metrics.gauc.value") == pytest.approx(0.78)


def test_resolve_metric_raises_for_missing_paths() -> None:
    with pytest.raises(KeyError, match="not present"):
        resolve_metric({"metrics": {"auc": 0.9}}, "metrics.gauc.value")


def test_budget_status_tracks_parameter_and_latency_limits() -> None:
    status = budget_status(
        {"parameter_size_mb": 128.0},
        {"estimated_end_to_end_inference_seconds": 12.5},
        load_experiment_package("config/baseline").search,
    )

    assert status["constraints_met"] is True
    assert status["parameter_bytes"] == pytest.approx(128.0 * 1024.0 * 1024.0)
    assert status["estimated_end_to_end_inference_minutes"] == pytest.approx(12.5 / 60.0)


def test_execute_search_trial_prunes_before_training(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_experiment_package(test_workspace.write_experiment_package())
    called = {"training": False}

    monkeypatch.setattr(
        "taac2026.application.search.trial.profile_trial_budget",
        lambda experiment: {
            "model_profile": {},
            "latency": {},
            "inference_profile": {},
            "budget_status": {"constraints_met": False},
        },
    )
    monkeypatch.setattr(
        "taac2026.application.search.trial.run_training",
        lambda experiment: called.__setitem__("training", True),
    )

    result = execute_search_trial(experiment)

    assert result["status"] == "pruned"
    assert "before training" in str(result["prune_reason"])
    assert called["training"] is False


def test_execute_search_trial_prunes_after_training_when_final_budget_fails(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_experiment_package(test_workspace.write_experiment_package())
    experiment.train.output_dir = str(test_workspace.root / "trial_budget")

    monkeypatch.setattr(
        "taac2026.application.search.trial.profile_trial_budget",
        lambda experiment: {
            "model_profile": {},
            "latency": {},
            "inference_profile": {},
            "budget_status": {"constraints_met": True},
        },
    )
    monkeypatch.setattr(
        "taac2026.application.search.trial.budget_status",
        lambda model_profile, inference_profile, search_config: {
            "constraints_met": False,
            "estimated_end_to_end_inference_seconds": 999.0,
        },
    )
    monkeypatch.setattr(
        "taac2026.application.search.trial.run_training",
        lambda experiment: {
            "model_profile": {"parameter_size_mb": 1.0},
            "inference_profile": {"estimated_end_to_end_inference_seconds": 999.0},
            "metrics": {"auc": 0.81},
        },
    )

    result = execute_search_trial(experiment)

    assert result["status"] == "pruned"
    assert "after training" in str(result["prune_reason"])
    assert result["final_budget_status"]["constraints_met"] is False
    assert result["summary_path"] == str(Path(experiment.train.output_dir) / "summary.json")


def test_execute_search_trial_surfaces_missing_metric_paths(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_experiment_package(test_workspace.write_experiment_package())
    experiment.search.metric_name = "metrics.gauc.value"

    monkeypatch.setattr(
        "taac2026.application.search.trial.profile_trial_budget",
        lambda experiment: {
            "model_profile": {},
            "latency": {},
            "inference_profile": {},
            "budget_status": {"constraints_met": True},
        },
    )
    monkeypatch.setattr(
        "taac2026.application.search.trial.budget_status",
        lambda model_profile, inference_profile, search_config: {"constraints_met": True},
    )
    monkeypatch.setattr(
        "taac2026.application.search.trial.run_training",
        lambda experiment: {
            "model_profile": {"parameter_size_mb": 1.0},
            "inference_profile": {"estimated_end_to_end_inference_seconds": 1.0},
            "metrics": {"auc": 0.81},
        },
    )

    with pytest.raises(KeyError, match="not present"):
        execute_search_trial(experiment)

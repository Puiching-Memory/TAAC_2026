from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from taac2026.application.search import service as search_service
from taac2026.application.search import trial as search_trial
from taac2026.application.search import worker as search_worker
from taac2026.application.search.service import (
    SearchWorkerProcess,
    _collect_worker_result,
    _finalize_parallel_trial,
    _run_search_auto,
)
from taac2026.application.search.trial import budget_status, execute_search_trial, resolve_metric
from taac2026.infrastructure.compute.device_scheduler import GpuDevice
from taac2026.infrastructure.experiments.loader import load_experiment_package
from tests.support import create_test_workspace


pytestmark = pytest.mark.unit


class _FakeTrial:
    def __init__(self, number: int) -> None:
        self.number = number
        self.params: dict[str, object] = {}
        self.user_attrs: dict[str, object] = {}
        self.state = SimpleNamespace(name="RUNNING")
        self.value = None

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value


class _FakeStudy:
    def __init__(self) -> None:
        self.trials: list[_FakeTrial] = []
        self.best_trial: _FakeTrial | None = None
        self.best_value: float | None = None

    def ask(self) -> _FakeTrial:
        trial = _FakeTrial(len(self.trials))
        self.trials.append(trial)
        return trial

    def tell(self, trial: _FakeTrial, value=None, state=None) -> None:
        state_name = state if isinstance(state, str) else getattr(state, "name", str(state))
        trial.state = SimpleNamespace(name=state_name)
        trial.value = value
        if state_name == "COMPLETE" and value is not None and (self.best_value is None or value > self.best_value):
            self.best_value = float(value)
            self.best_trial = trial


class _FakeProcess:
    def __init__(self, returncode: int = 0, *, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    def poll(self):
        return self.returncode

    def communicate(self):
        return self._stdout, self._stderr


def test_resolve_metric_rejects_missing_nested_keys() -> None:
    with pytest.raises(KeyError, match="metrics.auc"):
        resolve_metric({"metrics": {}}, "metrics.auc")


def test_budget_status_combines_parameter_and_latency_constraints() -> None:
    result = budget_status(
        {"parameter_size_mb": 64.0},
        {"estimated_end_to_end_inference_seconds": 12.0},
        SimpleNamespace(max_parameter_bytes=128 * 1024 * 1024, max_end_to_end_inference_seconds=20.0),
    )

    assert result["parameter_budget_met"] is True
    assert result["inference_budget_met"] is True
    assert result["constraints_met"] is True


def test_execute_search_trial_prunes_before_training(monkeypatch) -> None:
    monkeypatch.setattr(
        search_trial,
        "profile_trial_budget",
        lambda experiment: {"budget_status": {"constraints_met": False}},
    )

    result = execute_search_trial(SimpleNamespace())

    assert result["status"] == "pruned"
    assert result["prune_reason"] == "trial exceeds search budget before training"


def test_execute_search_trial_prunes_after_training_if_final_budget_exceeded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        search_trial,
        "profile_trial_budget",
        lambda experiment: {"budget_status": {"constraints_met": True}},
    )
    monkeypatch.setattr(
        search_trial,
        "run_training",
        lambda experiment: {
            "model_profile": {"parameter_size_mb": 64.0},
            "inference_profile": {"estimated_end_to_end_inference_seconds": 99.0},
            "best_val_auc": 0.81,
        },
    )
    monkeypatch.setattr(
        search_trial,
        "budget_status",
        lambda model_profile, inference_profile, search: {"constraints_met": False},
    )
    experiment = SimpleNamespace(
        train=SimpleNamespace(output_dir=str(tmp_path)),
        search=SimpleNamespace(metric_name="best_val_auc"),
    )

    result = execute_search_trial(experiment)

    assert result["status"] == "pruned"
    assert result["prune_reason"] == "trial exceeds search budget after training"
    assert result["summary_path"] == str(tmp_path / "summary.json")


def test_collect_worker_result_uses_process_output_when_result_file_is_missing(tmp_path: Path) -> None:
    worker = SearchWorkerProcess(
        trial=None,
        process=_FakeProcess(returncode=1, stdout="stdout", stderr="stderr"),
        result_path=tmp_path / "missing.json",
        physical_gpu_index=None,
    )

    result = _collect_worker_result(worker)

    assert result["status"] == "fail"
    assert result["trial_error"] == "stderr"


def test_finalize_parallel_trial_marks_missing_objective_as_failure() -> None:
    study = _FakeStudy()
    trial = study.ask()
    optuna = SimpleNamespace(trial=SimpleNamespace(TrialState=SimpleNamespace(COMPLETE="COMPLETE", PRUNED="PRUNED", FAIL="FAIL")))

    state = _finalize_parallel_trial(study, trial, {"status": "complete"}, optuna)

    assert state == "FAIL"
    assert trial.state.name == "FAIL"
    assert "objective value" in str(trial.user_attrs["trial_error"])


@pytest.mark.fault
def test_worker_main_writes_failure_payload(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "worker_experiment.json"
    result_path = tmp_path / "worker_result.json"
    config_path.write_text(json.dumps({}), encoding="utf-8")

    monkeypatch.setattr(search_worker, "load_experiment_package", lambda _path: (_ for _ in ()).throw(RuntimeError("boom")))

    exit_code = search_worker.worker_main(
        [
            "--experiment",
            "config/gen/baseline",
            "--config-path",
            str(config_path),
            "--result-path",
            str(result_path),
        ]
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["status"] == "fail"
    assert payload["trial_error"] == "boom"


def test_run_search_auto_collects_completed_worker_results(monkeypatch, tmp_path: Path) -> None:
    workspace = create_test_workspace(tmp_path)
    experiment_path = workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)
    experiment.search.n_trials = 1
    experiment.search.timeout_seconds = 5
    experiment.build_search_experiment = lambda base_experiment, trial: base_experiment.clone()

    study = _FakeStudy()
    optuna = SimpleNamespace(trial=SimpleNamespace(TrialState=SimpleNamespace(COMPLETE="COMPLETE", PRUNED="PRUNED", FAIL="FAIL")))

    monkeypatch.setattr(search_service, "_require_optuna", lambda: optuna)
    monkeypatch.setattr(search_service, "query_gpu_devices", lambda gpu_indices=None: [GpuDevice(index=0, name="gpu0", total_memory_mb=1024, used_memory_mb=0, free_memory_mb=1024)])
    monkeypatch.setattr(search_service, "launchable_devices", lambda devices, running_jobs_by_gpu, **kwargs: devices if not running_jobs_by_gpu else [])
    monkeypatch.setattr(
        search_service,
        "_launch_search_worker",
        lambda **kwargs: SearchWorkerProcess(
            trial=None,
            process=_FakeProcess(returncode=0),
            result_path=Path(kwargs["trial_dir"]) / "worker_result.json",
            physical_gpu_index=kwargs["physical_gpu_index"],
        ),
    )
    monkeypatch.setattr(
        search_service,
        "_collect_worker_result",
        lambda worker: {
            "status": "complete",
            "objective_value": 0.91,
            "summary_path": str(Path(worker.result_path).with_name("summary.json")),
            "final_budget_status": {"constraints_met": True},
            "budget_probe": {"budget_status": {"constraints_met": True}},
        },
    )

    report = _run_search_auto(
        study=study,
        experiment=experiment,
        experiment_path=experiment_path,
        study_root=tmp_path / "study",
        show_progress=False,
        gpu_indices=None,
        min_free_memory_gb=0.5,
        max_jobs_per_gpu=1,
        poll_interval_seconds=0.01,
        scheduler_info={"requested_mode": "auto", "used_mode": "auto"},
    )

    assert report["trial_state_counts"] == {"COMPLETE": 1}
    assert report["best_trial"] is not None
    assert report["best_trial"]["value"] == pytest.approx(0.91)
    assert report["trials"][0]["user_attrs"]["assigned_gpu_index"] == 0

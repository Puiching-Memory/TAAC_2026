from __future__ import annotations

import json
from pathlib import Path

import pytest

from taac2026.application.search.service import SearchWorkerProcess, run_search
from taac2026.infrastructure.compute.device_scheduler import GpuDevice
from taac2026.infrastructure.experiments.loader import load_experiment_package
from tests.support import TestWorkspace, create_test_workspace


class _CompletedProcess:
    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    def poll(self) -> int:
        return self.returncode

    def communicate(self) -> tuple[str, str]:
        return self._stdout, self._stderr


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


def test_run_search_auto_converges_complete_and_failed_workers(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)
    experiment.search.n_trials = 2
    experiment.search.timeout_seconds = 30
    experiment.search.sampler_seed = 11
    study_dir = test_workspace.root / "auto_workers"

    monkeypatch.setattr(
        "taac2026.application.search.service.query_gpu_devices",
        lambda gpu_indices=None: [
            GpuDevice(index=3, name="gpu3", total_memory_mb=81_920, used_memory_mb=8_192, free_memory_mb=73_728)
        ],
    )

    def fake_launch_worker(experiment_path, serialized_experiment, trial_dir, physical_gpu_index):
        result_path = trial_dir / "worker_result.json"
        summary_path = trial_dir / "summary.json"
        if trial_dir.name.endswith("0000"):
            summary_path.write_text(json.dumps({"best_val_auc": 0.81}), encoding="utf-8")
            result_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "objective_value": 0.81,
                        "budget_probe": {"budget_status": {"constraints_met": True}},
                        "summary_path": str(summary_path),
                        "final_budget_status": {"constraints_met": True},
                    }
                ),
                encoding="utf-8",
            )
            process = _CompletedProcess(returncode=0)
        else:
            result_path.write_text(
                json.dumps(
                    {
                        "status": "fail",
                        "trial_error": "worker exited unexpectedly",
                        "budget_probe": {"budget_status": {"constraints_met": True}},
                    }
                ),
                encoding="utf-8",
            )
            process = _CompletedProcess(returncode=1, stderr="worker exited unexpectedly")
        return SearchWorkerProcess(
            trial=None,
            process=process,
            result_path=result_path,
            physical_gpu_index=physical_gpu_index,
        )

    monkeypatch.setattr("taac2026.application.search.service._launch_search_worker", fake_launch_worker)

    report = run_search(
        experiment,
        experiment_path=experiment_path,
        study_dir=study_dir,
        scheduler="auto",
        max_jobs_per_gpu=1,
        poll_interval_seconds=0.01,
    )

    assert report["scheduler"]["used_mode"] == "auto"
    assert report["trial_state_counts"]["COMPLETE"] == 1
    assert report["trial_state_counts"]["FAIL"] == 1

    completed_trial = next(trial for trial in report["trials"] if trial["state"] == "COMPLETE")
    failed_trial = next(trial for trial in report["trials"] if trial["state"] == "FAIL")

    assert completed_trial["user_attrs"]["objective_value"] == pytest.approx(0.81)
    assert completed_trial["user_attrs"]["summary_path"].endswith("summary.json")
    assert completed_trial["user_attrs"]["assigned_gpu_index"] == 3
    assert failed_trial["user_attrs"]["trial_error"] == "worker exited unexpectedly"
    assert failed_trial["user_attrs"]["assigned_gpu_index"] == 3

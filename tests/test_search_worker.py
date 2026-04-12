from __future__ import annotations

import json
from pathlib import Path

import pytest

from taac2026.application.search.service import SearchWorkerProcess, _collect_worker_result
from taac2026.application.search.worker import worker_main
from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.experiments.payload import serialize_experiment
from tests.support import TestWorkspace, create_test_workspace


class _CompletedProcess:
    def __init__(self, *, returncode: int, stdout: str = "", stderr: str = "") -> None:
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


def test_collect_worker_result_returns_stderr_when_result_is_missing(tmp_path: Path) -> None:
    worker = SearchWorkerProcess(
        trial=None,
        process=_CompletedProcess(returncode=2, stderr="worker blew up"),
        result_path=tmp_path / "missing.json",
        physical_gpu_index=None,
    )

    result = _collect_worker_result(worker)

    assert result["status"] == "fail"
    assert result["trial_error"] == "worker blew up"


def test_collect_worker_result_handles_invalid_json_fault(tmp_path: Path) -> None:
    result_path = tmp_path / "worker_result.json"
    result_path.write_text("{not-valid-json", encoding="utf-8")
    worker = SearchWorkerProcess(
        trial=None,
        process=_CompletedProcess(returncode=1, stderr="json parse failure"),
        result_path=result_path,
        physical_gpu_index=None,
    )

    result = _collect_worker_result(worker)

    assert result["status"] == "fail"
    assert "invalid worker result payload" in result["trial_error"]


def test_worker_main_writes_failure_payload_when_trial_execution_raises(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_experiment_package(test_workspace.write_experiment_package())
    config_path = test_workspace.root / "worker_experiment.json"
    result_path = test_workspace.root / "worker_result.json"
    config_path.write_text(json.dumps(serialize_experiment(experiment)), encoding="utf-8")

    monkeypatch.setattr(
        "taac2026.application.search.worker.execute_search_trial",
        lambda experiment: (_ for _ in ()).throw(RuntimeError("trial exploded")),
    )

    exit_code = worker_main(
        [
            "--experiment",
            str(test_workspace.write_experiment_package()),
            "--config-path",
            str(config_path),
            "--result-path",
            str(result_path),
        ]
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "fail"
    assert "trial exploded" in payload["trial_error"]
    assert "RuntimeError" in payload["traceback"]


def test_worker_main_applies_device_override_on_success(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)
    config_path = test_workspace.root / "worker_success_experiment.json"
    result_path = test_workspace.root / "worker_success_result.json"
    config_path.write_text(json.dumps(serialize_experiment(experiment)), encoding="utf-8")
    captured: dict[str, str | None] = {}

    def fake_execute_search_trial(input_experiment):
        captured["device"] = input_experiment.train.device
        return {"status": "complete", "objective_value": 0.82}

    monkeypatch.setattr("taac2026.application.search.worker.execute_search_trial", fake_execute_search_trial)

    exit_code = worker_main(
        [
            "--experiment",
            str(experiment_path),
            "--config-path",
            str(config_path),
            "--result-path",
            str(result_path),
            "--device",
            "cuda:0",
        ]
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured["device"] == "cuda:0"
    assert payload["status"] == "complete"
    assert payload["objective_value"] == pytest.approx(0.82)

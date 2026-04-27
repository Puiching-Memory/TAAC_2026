from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from taac2026.domain.config import InferRequest
from taac2026.infrastructure.pcvr.experiment import PCVRExperiment, _log_prediction_progress
from taac2026.infrastructure.training.runtime import RuntimeExecutionConfig


def _make_experiment(tmp_path: Path, *, default_train_args: tuple[str, ...] = ()) -> PCVRExperiment:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    return PCVRExperiment(
        name="pcvr_symbiosis",
        package_dir=package_dir,
        model_class_name="DummyModel",
        default_train_args=default_train_args,
    )


def test_resolve_infer_runtime_settings_falls_back_to_experiment_defaults(tmp_path: Path) -> None:
    experiment = _make_experiment(
        tmp_path,
        default_train_args=("--batch_size", "128", "--num_workers", "8"),
    )
    request = InferRequest(
        experiment="config/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    resolved = experiment._resolve_infer_runtime_settings(request, {})

    assert resolved == (128, "experiment_default_train_args", 8, "experiment_default_train_args")


def test_resolve_infer_runtime_settings_preserves_explicit_request_values(tmp_path: Path) -> None:
    experiment = _make_experiment(
        tmp_path,
        default_train_args=("--batch_size", "128", "--num_workers", "8"),
    )
    request = InferRequest(
        experiment="config/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
        batch_size=64,
        num_workers=2,
    )

    resolved = experiment._resolve_infer_runtime_settings(request, {"batch_size": 32, "num_workers": 6})

    assert resolved == (64, "request", 2, "request")


def test_infer_uses_train_config_runtime_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(
        tmp_path,
        default_train_args=("--batch_size", "128", "--num_workers", "8"),
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").write_bytes(b"checkpoint")
    (checkpoint_dir / "train_config.json").write_text(
        json.dumps({"batch_size": 96, "num_workers": 4}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_prediction_loop(**kwargs):
        captured.update(kwargs)
        return {"records": [{"user_id": "u1", "score": 0.5, "target": 0.0, "timestamp": None}]}

    def fake_bound_run_prediction_loop(self, **kwargs):
        del self
        return fake_run_prediction_loop(**kwargs)

    monkeypatch.setenv("MODEL_OUTPUT_PATH", str(checkpoint_dir))
    monkeypatch.setattr(PCVRExperiment, "_run_prediction_loop", fake_bound_run_prediction_loop)

    request = InferRequest(
        experiment="config/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    payload = experiment.infer(request)

    assert captured["batch_size"] == 96
    assert captured["num_workers"] == 4
    assert payload["batch_size"] == 96
    assert payload["num_workers"] == 4
    assert json.loads((tmp_path / "results" / "predictions.json").read_text(encoding="utf-8")) == {
        "predictions": {"u1": 0.5},
    }


def test_resolve_prediction_runtime_execution_falls_back_to_experiment_defaults(tmp_path: Path) -> None:
    experiment = _make_experiment(
        tmp_path,
        default_train_args=("--amp", "--amp-dtype", "float16", "--compile"),
    )
    request = InferRequest(
        experiment="config/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    runtime_execution, amp_source, amp_dtype_source, compile_source = experiment._resolve_prediction_runtime_execution(
        request,
        {},
    )

    assert runtime_execution == RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=True)
    assert amp_source == "experiment_default_train_args"
    assert amp_dtype_source == "experiment_default_train_args"
    assert compile_source == "experiment_default_train_args"


def test_infer_uses_train_config_runtime_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").write_bytes(b"checkpoint")
    (checkpoint_dir / "train_config.json").write_text(
        json.dumps({"amp": True, "amp_dtype": "float16", "compile": True}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_bound_run_prediction_loop(self, **kwargs):
        del self
        captured.update(kwargs)
        return {"records": [{"user_id": "u1", "score": 0.5, "target": 0.0, "timestamp": None}]}

    monkeypatch.setenv("MODEL_OUTPUT_PATH", str(checkpoint_dir))
    monkeypatch.setattr(PCVRExperiment, "_run_prediction_loop", fake_bound_run_prediction_loop)

    request = InferRequest(
        experiment="config/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    experiment.infer(request)

    runtime_execution = captured["runtime_execution"]
    assert runtime_execution == RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=True)


def test_infer_request_runtime_settings_override_train_config(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path)
    request = InferRequest(
        experiment="config/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
        amp=False,
        amp_dtype="bfloat16",
        compile=False,
    )

    runtime_execution, amp_source, amp_dtype_source, compile_source = experiment._resolve_prediction_runtime_execution(
        request,
        {"amp": True, "amp_dtype": "float16", "compile": True},
    )

    assert runtime_execution == RuntimeExecutionConfig(amp=False, amp_dtype="bfloat16", compile=False)
    assert amp_source == "request"
    assert amp_dtype_source == "request"
    assert compile_source == "request"


def test_log_prediction_progress_reports_rows_and_batches(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        _log_prediction_progress(
            mode="inference",
            processed_rows=50_000,
            total_rows=310_000,
            batch_index=200,
            total_batches=1_211,
            elapsed_seconds=12.3,
        )

    assert "PCVR inference progress: 50000/310000 rows (16.1%), batch 200/1211, elapsed=12.3s" in caplog.text
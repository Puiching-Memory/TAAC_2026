from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType

import pytest

from taac2026.domain.config import EvalRequest, InferRequest
from taac2026.infrastructure.io.json_utils import dumps, loads
from taac2026.infrastructure.pcvr.config import PCVRDataConfig, PCVRTrainConfig
import taac2026.infrastructure.pcvr.experiment as experiment_module
from taac2026.infrastructure.pcvr.experiment import PCVRExperiment, _log_prediction_progress
from taac2026.infrastructure.training.runtime import RuntimeExecutionConfig


def _make_experiment(tmp_path: Path, *, train_defaults: PCVRTrainConfig | None = None) -> PCVRExperiment:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    return PCVRExperiment(
        name="pcvr_symbiosis",
        package_dir=package_dir,
        model_class_name="DummyModel",
        train_defaults=train_defaults or PCVRTrainConfig(),
    )


def _write_train_config(checkpoint_dir: Path, overrides: dict[str, object] | None = None) -> None:
    config = PCVRTrainConfig().to_flat_dict()
    if overrides:
        config.update(overrides)
    (checkpoint_dir / "train_config.json").write_text(dumps(config), encoding="utf-8")


def test_resolve_infer_runtime_settings_requires_train_config_values(tmp_path: Path) -> None:
    experiment = _make_experiment(
        tmp_path,
        train_defaults=PCVRTrainConfig(data=PCVRDataConfig(batch_size=128, num_workers=8)),
    )
    request = InferRequest(
        experiment="config/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    with pytest.raises(KeyError, match="batch_size"):
        experiment._resolve_infer_runtime_settings(request, {})


def test_resolve_infer_runtime_settings_preserves_explicit_request_values(tmp_path: Path) -> None:
    experiment = _make_experiment(
        tmp_path,
        train_defaults=PCVRTrainConfig(data=PCVRDataConfig(batch_size=128, num_workers=8)),
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
        train_defaults=PCVRTrainConfig(data=PCVRDataConfig(batch_size=128, num_workers=8)),
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"checkpoint")
    schema_payload = {"features": [{"name": "user_id"}]}
    (checkpoint_dir / "schema.json").write_text(dumps(schema_payload), encoding="utf-8")
    _write_train_config(checkpoint_dir, {"batch_size": 96, "num_workers": 4})
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
    assert payload["schema_path"] == str((checkpoint_dir / "schema.json").resolve())
    assert payload["schema"] == schema_payload
    assert loads((tmp_path / "results" / "predictions.json").read_bytes()) == {
        "predictions": {"u1": 0.5},
    }


def test_resolve_prediction_runtime_execution_requires_train_config_values(tmp_path: Path) -> None:
    experiment = _make_experiment(
        tmp_path,
        train_defaults=PCVRTrainConfig(runtime=RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=True)),
    )
    request = InferRequest(
        experiment="config/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    with pytest.raises(KeyError, match="amp"):
        experiment._resolve_prediction_runtime_execution(request, {})


def test_load_train_config_requires_sidecar(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=r"train_config\.json"):
        experiment._load_train_config(checkpoint_dir)


def test_load_train_config_requires_complete_sidecar(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "train_config.json").write_text(dumps({"batch_size": 128}), encoding="utf-8")

    with pytest.raises(KeyError, match="amp"):
        experiment._load_train_config(checkpoint_dir)


def test_infer_uses_train_config_runtime_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"checkpoint")
    (checkpoint_dir / "schema.json").write_text(dumps({"features": []}), encoding="utf-8")
    _write_train_config(checkpoint_dir, {"amp": True, "amp_dtype": "float16", "compile": True})
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


def test_run_prediction_loop_loads_safetensors_via_checkpoint_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "model.safetensors"
    checkpoint_path.write_bytes(b"checkpoint")
    loaded_state_dict = {"weight": object()}

    class FakeDataset:
        num_rows = 0

    class FakeModel:
        def __init__(self) -> None:
            self.loaded_state_dict = None

        def to(self, device) -> FakeModel:
            del device
            return self

        def load_state_dict(self, state_dict) -> None:
            self.loaded_state_dict = state_dict

        def eval(self) -> None:
            return None

        def predict(self, model_input):
            del model_input
            raise AssertionError("predict should not run when loader is empty")

    fake_model = FakeModel()
    monkeypatch.setitem(__import__("sys").modules, "model", ModuleType("model"))
    monkeypatch.setattr(
        PCVRExperiment,
        "_resolve_schema_path",
        lambda self, dataset_path, schema_path, checkpoint_dir: checkpoint_dir / "schema.json",
    )
    monkeypatch.setattr(experiment_module, "parse_seq_max_lens", lambda value: {})
    monkeypatch.setattr(experiment_module.pcvr_data, "PCVRParquetDataset", lambda **kwargs: FakeDataset())
    monkeypatch.setattr(experiment_module, "DataLoader", lambda dataset, batch_size, num_workers, pin_memory: [])
    monkeypatch.setattr(experiment_module, "build_pcvr_model", lambda **kwargs: fake_model)
    monkeypatch.setattr(experiment_module, "load_checkpoint_state_dict", lambda path, map_location=None: loaded_state_dict)
    monkeypatch.setattr(
        experiment_module.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("torch.load should not be used for safetensors checkpoints")),
    )

    payload = experiment._run_prediction_loop(
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=checkpoint_path,
        batch_size=32,
        num_workers=0,
        device="cpu",
        is_training_data=False,
        config={"seq_max_lens": "{}"},
    )

    assert payload == {"labels": [], "probabilities": [], "records": []}
    assert fake_model.loaded_state_dict is loaded_state_dict


def test_evaluate_writes_score_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "model.safetensors"
    checkpoint_path.write_bytes(b"checkpoint")
    schema_payload = {"features": [{"name": "label"}]}
    (checkpoint_dir / "schema.json").write_text(dumps(schema_payload), encoding="utf-8")
    _write_train_config(checkpoint_dir)

    def fake_bound_run_prediction_loop(self, **kwargs):
        del self, kwargs
        return {
            "labels": [0.0, 1.0, 1.0, 0.0],
            "probabilities": [0.1, 0.9, 0.8, 0.2],
            "records": [
                {"user_id": "u0", "score": 0.1, "target": 0.0, "timestamp": None},
                {"user_id": "u1", "score": 0.9, "target": 1.0, "timestamp": None},
            ],
        }

    monkeypatch.setattr(PCVRExperiment, "_run_prediction_loop", fake_bound_run_prediction_loop)
    monkeypatch.setattr(
        "taac2026.infrastructure.pcvr.experiment.pcvr_data.collect_pcvr_row_groups",
        lambda _path: [(str(tmp_path / "eval.parquet"), 0, 2), (str(tmp_path / "eval.parquet"), 1, 2)],
    )
    output_path = tmp_path / "evaluation.json"
    request = EvalRequest(
        experiment="config/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        run_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        predictions_path=tmp_path / "predictions.jsonl",
    )

    payload = experiment.evaluate(request)

    diagnostics = payload["metrics"]["score_diagnostics"]
    assert diagnostics["positive_count"] == 2
    assert diagnostics["negative_count"] == 2
    assert diagnostics["score_margin_mean"] == pytest.approx(0.7)
    assert payload["metrics"]["auc_ci"]["low"] <= payload["metrics"]["auc"] <= payload["metrics"]["auc_ci"]["high"]
    assert payload["data_diagnostics"]["row_group_split"]["is_l1_ready"] is True
    assert payload["schema_path"] == str((checkpoint_dir / "schema.json").resolve())
    assert payload["schema"] == schema_payload
    saved_payload = loads(output_path.read_bytes())
    assert saved_payload["metrics"]["score_diagnostics"] == diagnostics
    assert saved_payload["data_diagnostics"] == payload["data_diagnostics"]
    assert saved_payload["schema"] == schema_payload
    predictions_payload = (tmp_path / "predictions.jsonl").read_bytes()
    assert predictions_payload.endswith(b"\n")
    assert [loads(line) for line in predictions_payload.splitlines()] == [
        {"user_id": "u0", "score": 0.1, "target": 0.0, "timestamp": None},
        {"user_id": "u1", "score": 0.9, "target": 1.0, "timestamp": None},
    ]


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
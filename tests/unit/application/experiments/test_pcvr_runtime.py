from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taac2026.domain.requests import EvalRequest, InferRequest, TrainRequest
from taac2026.infrastructure.io.json import dumps, loads
from taac2026.domain.config import PCVRDataConfig, PCVRTrainConfig
from taac2026.domain.sidecar import build_pcvr_train_config_sidecar
import taac2026.application.experiments.experiment as experiment_module
from taac2026.application.experiments.experiment import PCVRExperiment, _log_prediction_progress
from taac2026.application.evaluation.workflow import (
    PCVRPredictionDataBundle,
    PCVRPredictionRunner,
    build_pcvr_prediction_hooks,
)
from taac2026.application.evaluation.runtime import build_pcvr_runtime_hooks
from taac2026.application.training.workflow import build_pcvr_train_hooks
from taac2026.application.training.args import parse_pcvr_train_args
from taac2026.infrastructure.runtime.execution import RuntimeExecutionConfig


def _make_experiment(
    tmp_path: Path,
    *,
    train_defaults: PCVRTrainConfig | None = None,
    prediction_hooks=None,
    runtime_hooks=None,
) -> PCVRExperiment:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "model.py").write_text(
        "class ModelInput:\n    pass\n\nclass DummyModel:\n    pass\n",
        encoding="utf-8",
    )
    return PCVRExperiment(
        name="pcvr_symbiosis",
        package_dir=package_dir,
        model_class_name="DummyModel",
        train_defaults=train_defaults or PCVRTrainConfig(),
        train_arg_parser=parse_pcvr_train_args,
        train_hooks=build_pcvr_train_hooks(),
        prediction_hooks=prediction_hooks or build_pcvr_prediction_hooks(),
        runtime_hooks=runtime_hooks or build_pcvr_runtime_hooks(),
    )


def _write_observed_schema_fixture(schema_path: Path, parquet_path: Path) -> None:
    payload = {
        "user_int": [[1, 10, 1], [2, 20, 4]],
        "item_int": [[3, 20, 1]],
        "user_dense": [[4, 4]],
        "seq": {
            "seq_a": {
                "prefix": "domain_a_seq",
                "ts_fid": 10,
                "features": [[10, 0], [11, 20]],
            }
        },
    }
    schema_path.write_text(dumps(payload), encoding="utf-8")
    pq.write_table(
        pa.table(
            {
                "user_int_feats_1": [1, 2],
                "user_int_feats_2": [[1, 2], [2, 3, 4]],
                "item_int_feats_3": [10, 11],
                "user_dense_feats_4": [[0.1, 0.2], [0.3]],
                "domain_a_seq_10": [[100, 101], [103]],
                "domain_a_seq_11": [[5, 6], [6, 7, 7]],
                "timestamp": [100, 200],
            }
        ),
        parquet_path,
        row_group_size=1,
    )


def _write_train_config(checkpoint_dir: Path, overrides: dict[str, object] | None = None) -> None:
    config = PCVRTrainConfig().to_flat_dict()
    if overrides:
        config.update(overrides)
    (checkpoint_dir / "train_config.json").write_text(dumps(build_pcvr_train_config_sidecar(config)), encoding="utf-8")


def test_resolve_infer_runtime_settings_requires_train_config_values(tmp_path: Path) -> None:
    experiment = _make_experiment(
        tmp_path,
        train_defaults=PCVRTrainConfig(data=PCVRDataConfig(batch_size=128, num_workers=8)),
    )
    request = InferRequest(
        experiment="experiments/symbiosis",
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
        experiment="experiments/symbiosis",
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
        experiment="experiments/symbiosis",
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
    assert payload["telemetry"]["label"] == "inference"
    assert payload["telemetry"]["rows"] == 1
    assert (tmp_path / "results" / "inference_telemetry.json").exists()
    assert loads((tmp_path / "results" / "predictions.json").read_bytes()) == {
        "predictions": {"u1": 0.5},
    }


def test_train_writes_split_observed_schema_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _make_experiment(tmp_path)
    dataset_path = tmp_path / "train.parquet"
    schema_path = tmp_path / "schema.json"
    run_dir = tmp_path / "outputs"
    _write_observed_schema_fixture(schema_path, dataset_path)

    monkeypatch.setattr(
        experiment_module,
        "train_pcvr_model",
        lambda **kwargs: {
            "run_dir": str(run_dir.resolve()),
            "checkpoint_root": str(run_dir.resolve()),
            "schema_path": str(schema_path.resolve()),
            "train_ratio": 1.0,
            "valid_ratio": 0.1,
        },
    )

    payload = experiment.train(
        TrainRequest(
            experiment="experiments/symbiosis",
            dataset_path=dataset_path,
            schema_path=schema_path,
            run_dir=run_dir,
        )
    )

    observed_paths = payload["observed_schema_paths"]
    assert set(observed_paths) == {"train_split", "valid_split"}
    train_report = loads(Path(observed_paths["train_split"]).read_bytes())
    valid_report = loads(Path(observed_paths["valid_split"]).read_bytes())
    assert train_report["dataset_role"] == "train_split"
    assert valid_report["dataset_role"] == "valid_split"
    assert payload["row_group_split"]["train_row_group_range"] == [0, 1]
    assert payload["row_group_split"]["valid_row_group_range"] == [1, 2]
    assert train_report["schema"]["user_int"] == [[1, 1, 1], [2, 2, 2]]
    assert valid_report["schema"]["user_int"] == [[1, 1, 1], [2, 3, 3]]


def test_train_writes_timestamp_split_observed_schema_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _make_experiment(tmp_path)
    dataset_path = tmp_path / "train.parquet"
    schema_path = tmp_path / "schema.json"
    run_dir = tmp_path / "outputs"
    _write_observed_schema_fixture(schema_path, dataset_path)

    monkeypatch.setattr(
        experiment_module,
        "train_pcvr_model",
        lambda **kwargs: {
            "run_dir": str(run_dir.resolve()),
            "checkpoint_root": str(run_dir.resolve()),
            "schema_path": str(schema_path.resolve()),
            "train_ratio": 1.0,
            "valid_ratio": 0.1,
            "split_strategy": "timestamp_range",
            "train_timestamp_start": 0,
            "train_timestamp_end": 150,
            "valid_timestamp_start": 150,
            "valid_timestamp_end": 250,
        },
    )

    payload = experiment.train(
        TrainRequest(
            experiment="experiments/symbiosis",
            dataset_path=dataset_path,
            schema_path=schema_path,
            run_dir=run_dir,
        )
    )

    observed_paths = payload["observed_schema_paths"]
    train_report = loads(Path(observed_paths["train_split"]).read_bytes())
    valid_report = loads(Path(observed_paths["valid_split"]).read_bytes())
    assert payload["data_split"] == {
        "split_strategy": "timestamp_range",
        "train_timestamp_range": {"start": None, "end": 150},
        "valid_timestamp_range": {"start": 150, "end": 250},
    }
    assert payload["row_group_split"]["train_row_group_range"] == [0, 2]
    assert payload["row_group_split"]["valid_row_group_range"] == [0, 2]
    assert payload["row_group_split"]["train_rows"] == 1
    assert payload["row_group_split"]["valid_rows"] == 1
    assert train_report["timestamp_range"] == {"start": None, "end": 150}
    assert valid_report["timestamp_range"] == {"start": 150, "end": 250}
    assert train_report["schema"]["user_int"] == [[1, 1, 1], [2, 2, 2]]
    assert valid_report["schema"]["user_int"] == [[1, 1, 1], [2, 3, 3]]


def test_train_defaults_missing_dataset_to_hf_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(tmp_path)
    run_dir = tmp_path / "outputs"
    resolved_dataset_path = tmp_path / "hf_cache" / "demo_1000.parquet"
    resolved_schema_path = tmp_path / "hf_cache" / "schema.json"
    resolved_dataset_path.parent.mkdir(parents=True)
    _write_observed_schema_fixture(resolved_schema_path, resolved_dataset_path)
    captured_argv: dict[str, object] = {}

    def fake_train_pcvr_model(**kwargs):
        captured_argv["argv"] = kwargs["argv"]
        return {
            "run_dir": str(run_dir.resolve()),
            "checkpoint_root": str(run_dir.resolve()),
            "schema_path": str(resolved_schema_path.resolve()),
            "train_ratio": 1.0,
            "valid_ratio": 0.1,
        }

    monkeypatch.setattr(
        experiment_module,
        "resolve_default_pcvr_sample_paths",
        lambda dataset_path, schema_path: (resolved_dataset_path, resolved_schema_path),
    )
    monkeypatch.setattr(experiment_module, "train_pcvr_model", fake_train_pcvr_model)

    payload = experiment.train(
        TrainRequest(
            experiment="experiments/symbiosis",
            dataset_path=None,
            schema_path=None,
            run_dir=run_dir,
        )
    )

    assert "--data_dir" in captured_argv["argv"]
    assert str(resolved_dataset_path) in captured_argv["argv"]
    assert "--schema_path" in captured_argv["argv"]
    assert str(resolved_schema_path) in captured_argv["argv"]
    assert set(payload["observed_schema_paths"]) == {"train_split", "valid_split"}


def test_infer_defaults_missing_dataset_to_hf_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"checkpoint")
    _write_train_config(checkpoint_dir, {"batch_size": 32, "num_workers": 1})
    resolved_dataset_path = tmp_path / "hf_cache" / "demo_1000.parquet"
    resolved_dataset_path.parent.mkdir(parents=True)
    resolved_dataset_path.write_bytes(b"parquet")
    schema_payload = {"features": [{"name": "user_id"}]}
    resolved_schema_path = tmp_path / "hf_cache" / "schema.json"
    resolved_schema_path.write_text(dumps(schema_payload), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_bound_run_prediction_loop(self, **kwargs):
        del self
        captured.update(kwargs)
        return {"records": [{"user_id": "u1", "score": 0.5, "target": 0.0, "timestamp": None}]}

    monkeypatch.setenv("MODEL_OUTPUT_PATH", str(checkpoint_dir))
    monkeypatch.setattr(
        experiment_module,
        "resolve_default_pcvr_sample_paths",
        lambda dataset_path, schema_path: (resolved_dataset_path, resolved_schema_path),
    )
    monkeypatch.setattr(PCVRExperiment, "_run_prediction_loop", fake_bound_run_prediction_loop)

    request = InferRequest(
        experiment="experiments/symbiosis",
        dataset_path=None,
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    payload = experiment.infer(request)

    assert captured["dataset_path"] == resolved_dataset_path
    assert payload["schema_path"] == str(resolved_schema_path.resolve())


def test_resolve_prediction_runtime_execution_requires_train_config_values(tmp_path: Path) -> None:
    experiment = _make_experiment(
        tmp_path,
        train_defaults=PCVRTrainConfig(runtime=RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=True)),
    )
    request = InferRequest(
        experiment="experiments/symbiosis",
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
    incomplete_config = PCVRTrainConfig().to_flat_dict()
    incomplete_config.pop("amp")
    (checkpoint_dir / "train_config.json").write_text(dumps(build_pcvr_train_config_sidecar(incomplete_config)), encoding="utf-8")

    with pytest.raises(KeyError, match="amp"):
        experiment._load_train_config(checkpoint_dir)


def test_load_train_config_raises_on_missing_data_split_keys(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    config = PCVRTrainConfig().to_flat_dict()
    for key in (
        "split_strategy",
        "train_timestamp_start",
        "train_timestamp_end",
        "valid_timestamp_start",
        "valid_timestamp_end",
        "deterministic",
    ):
        config.pop(key)
    (checkpoint_dir / "train_config.json").write_text(
        dumps(build_pcvr_train_config_sidecar(config)),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="split_strategy"):
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
        experiment="experiments/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    experiment.infer(request)

    runtime_execution = captured["runtime_execution"]
    assert runtime_execution == RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=True)


def test_infer_uses_injected_runtime_hooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_dir = tmp_path / "hook_checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "model.safetensors"
    checkpoint_path.write_bytes(b"checkpoint")
    schema_path = checkpoint_dir / "custom_schema.json"
    schema_payload = {"features": ["runtime_probe"]}
    schema_path.write_text(dumps(schema_payload), encoding="utf-8")
    events: list[tuple[str, object]] = []
    captured: dict[str, object] = {}

    def resolve_inference_checkpoint(experiment, request):
        del experiment
        events.append(("checkpoint", request.dataset_path))
        return checkpoint_path

    def load_train_config(experiment, checkpoint_dir_arg):
        del experiment
        events.append(("config", checkpoint_dir_arg))
        return {
            **PCVRTrainConfig().to_flat_dict(),
            "amp": True,
            "amp_dtype": "float16",
            "compile": True,
        }

    def load_runtime_schema(experiment, *, dataset_path, schema_path, checkpoint_dir, mode):
        del experiment, schema_path
        events.append(("schema", (dataset_path, checkpoint_dir, mode)))
        return schema_path_override.resolve(), schema_payload

    schema_path_override = schema_path
    experiment = _make_experiment(
        tmp_path,
        runtime_hooks=build_pcvr_runtime_hooks(
            resolve_inference_checkpoint=resolve_inference_checkpoint,
            load_train_config=load_train_config,
            load_runtime_schema=load_runtime_schema,
        ),
    )

    def fake_bound_run_prediction_loop(self, **kwargs):
        del self
        captured.update(kwargs)
        return {"records": [{"user_id": "u1", "score": 0.5, "target": 0.0, "timestamp": None}]}

    monkeypatch.setattr(PCVRExperiment, "_run_prediction_loop", fake_bound_run_prediction_loop)

    request = InferRequest(
        experiment="experiments/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    payload = experiment.infer(request)

    assert payload["checkpoint_path"] == str(checkpoint_path)
    assert payload["schema_path"] == str(schema_path.resolve())
    assert payload["schema"] == schema_payload
    assert captured["runtime_execution"] == RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=True)
    assert events == [
        ("checkpoint", request.dataset_path),
        ("config", checkpoint_dir),
        ("schema", (request.dataset_path, checkpoint_dir, "inference")),
    ]


def test_run_prediction_loop_uses_injected_prediction_hooks(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    schema_path = checkpoint_dir / "schema.json"
    schema_path.write_text(dumps({"features": []}), encoding="utf-8")
    checkpoint_path = checkpoint_dir / "model.safetensors"
    events: list[tuple[str, object]] = []

    def build_data(context):
        events.append(("data", context.schema_path))
        return PCVRPredictionDataBundle(
            dataset=SimpleNamespace(num_rows=2),
            loader=[{"probe": 1}],
            data_module="custom_data_module",
        )

    def build_model(context, data_bundle):
        del context
        events.append(("model", data_bundle.data_module))
        return {"checkpoint": str(checkpoint_path)}

    def prepare_predictor(context, data_bundle, model):
        del context, data_bundle
        events.append(("predictor", model))

        def predict_fn(payload):
            return {"seen": payload}

        return PCVRPredictionRunner(model=model, predict_fn=predict_fn)

    def run_loop(context, data_bundle, runner):
        events.append(("loop", (context.mode, list(data_bundle.loader))))
        return {
            "labels": [0.0],
            "probabilities": [0.5],
            "records": [runner.predict_fn("probe")],
        }

    experiment = _make_experiment(
        tmp_path,
        prediction_hooks=build_pcvr_prediction_hooks(
            build_data=build_data,
            build_model=build_model,
            prepare_predictor=prepare_predictor,
            run_loop=run_loop,
        ),
    )

    with experiment._module_context():
        payload = experiment._run_prediction_loop(
            dataset_path=tmp_path / "eval.parquet",
            schema_path=None,
            checkpoint_path=checkpoint_path,
            batch_size=32,
            num_workers=0,
            device="cpu",
            is_training_data=False,
            dataset_role="inference",
            config={"seq_max_lens": "{}"},
        )

    assert payload == {
        "labels": [0.0],
        "probabilities": [0.5],
        "records": [{"seen": "probe"}],
    }
    assert events == [
        ("data", schema_path.resolve()),
        ("model", "custom_data_module"),
        ("predictor", {"checkpoint": str(checkpoint_path)}),
        ("loop", ("inference", [{"probe": 1}])),
    ]

def test_evaluate_writes_score_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "model.safetensors"
    checkpoint_path.write_bytes(b"checkpoint")
    schema_payload = {
        "user_int": [[1, 10, 1], [2, 20, 4]],
        "item_int": [[3, 20, 1]],
        "user_dense": [[4, 4]],
        "seq": {
            "seq_a": {
                "prefix": "domain_a_seq",
                "ts_fid": 10,
                "features": [[10, 0], [11, 20]],
            }
        },
    }
    (checkpoint_dir / "schema.json").write_text(dumps(schema_payload), encoding="utf-8")
    _write_train_config(checkpoint_dir)
    dataset_path = tmp_path / "eval.parquet"
    _write_observed_schema_fixture(checkpoint_dir / "schema.json", dataset_path)

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
    output_path = tmp_path / "evaluation.json"
    request = EvalRequest(
        experiment="experiments/symbiosis",
        dataset_path=dataset_path,
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
    assert payload["telemetry"]["label"] == "evaluation"
    assert payload["telemetry"]["rows"] == 4
    assert payload["data_diagnostics"]["row_group_split"]["is_l1_ready"] is True
    assert payload["schema_path"] == str((checkpoint_dir / "schema.json").resolve())
    assert payload["schema"] == schema_payload
    observed_schema_path = Path(payload["observed_schema_paths"]["eval"])
    assert observed_schema_path.exists()
    observed_schema_payload = loads(observed_schema_path.read_bytes())
    assert observed_schema_payload["dataset_role"] == "eval"
    assert observed_schema_payload["schema"]["user_int"] == [[1, 2, 1], [2, 4, 3]]
    saved_payload = loads(output_path.read_bytes())
    assert saved_payload["metrics"]["score_diagnostics"] == diagnostics
    assert saved_payload["telemetry"]["label"] == "evaluation"
    assert saved_payload["data_diagnostics"] == payload["data_diagnostics"]
    assert saved_payload["schema"] == schema_payload
    assert saved_payload["observed_schema_paths"] == payload["observed_schema_paths"]
    predictions_payload = (tmp_path / "predictions.jsonl").read_bytes()
    assert predictions_payload.endswith(b"\n")
    assert [loads(line) for line in predictions_payload.splitlines()] == [
        {"user_id": "u0", "score": 0.1, "target": 0.0, "timestamp": None},
        {"user_id": "u1", "score": 0.9, "target": 1.0, "timestamp": None},
    ]
    assert (tmp_path / "evaluation_telemetry.json").exists()


def test_infer_request_runtime_settings_override_train_config(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path)
    request = InferRequest(
        experiment="experiments/symbiosis",
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


def test_log_prediction_progress_reports_rows_and_batches(log_capture) -> None:
    with log_capture.at_level(logging.INFO):
        _log_prediction_progress(
            mode="inference",
            processed_rows=50_000,
            total_rows=310_000,
            batch_index=200,
            total_batches=1_211,
            elapsed_seconds=12.3,
        )

    assert "PCVR inference progress:" in log_capture.text
    assert "50000/310000 rows" in log_capture.text
    assert "batch 200/1211" in log_capture.text
    assert "elapsed=12.3s" in log_capture.text

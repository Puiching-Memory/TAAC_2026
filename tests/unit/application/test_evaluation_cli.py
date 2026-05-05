from __future__ import annotations

from types import SimpleNamespace

import pytest

import taac2026.application.evaluation.cli as evaluation_cli
from taac2026.application.evaluation.cli import parse_eval_args
from taac2026.infrastructure.io.json_utils import loads


def test_parse_eval_args_accepts_runtime_flags() -> None:
    args = parse_eval_args(
        [
            "infer",
            "--experiment",
            "experiments/pcvr/baseline",
            "--dataset-path",
            "/tmp/eval.parquet",
            "--result-dir",
            "/tmp/results",
            "--amp",
            "--amp-dtype",
            "bfloat16",
            "--compile",
        ]
    )

    assert args.command == "infer"
    assert args.amp is True
    assert args.amp_dtype == "bfloat16"
    assert args.compile is True


def test_main_output_is_compact_single_line(monkeypatch, capsys) -> None:
    payload = {
        "checkpoint_path": "/tmp/model.safetensors",
        "schema_path": "/tmp/schema.json",
        "schema": {"features": [{"name": "user_id"}]},
    }

    class FakeExperiment:
        def infer(self, request):
            del request
            return payload

    monkeypatch.setattr(evaluation_cli, "load_experiment_package", lambda _experiment: FakeExperiment())

    exit_code = evaluation_cli.main(
        [
            "infer",
            "--experiment",
            "experiments/pcvr/baseline",
            "--dataset-path",
            "/tmp/eval.parquet",
            "--result-dir",
            "/tmp/results",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "\n" not in captured.out.strip()
    assert '"schema":{"features":[{"name":"user_id"}]}' in captured.out.strip()
    assert loads(captured.out) == payload


def test_evaluation_main_allows_missing_dataset_for_pcvr_kind_experiment(monkeypatch, capsys) -> None:
    payload = {
        "checkpoint_path": "/tmp/model.safetensors",
        "schema_path": "/tmp/schema.json",
        "schema": {},
    }
    captured_request = {}

    class FakeExperiment:
        metadata = {"requires_dataset": True, "kind": "pcvr"}

        def infer(self, request):
            captured_request["dataset_path"] = request.dataset_path
            return payload

    monkeypatch.setattr(evaluation_cli, "load_experiment_package", lambda _experiment: FakeExperiment())

    exit_code = evaluation_cli.main(
        [
            "infer",
            "--experiment",
            "experiments/pcvr/baseline",
            "--result-dir",
            "/tmp/results",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured_request["dataset_path"] is None
    assert loads(captured.out) == payload


def test_evaluation_main_rejects_explicit_dataset_for_local_pcvr_kind_experiment(monkeypatch) -> None:
    class FakeExperiment:
        metadata = {"requires_dataset": True, "kind": "pcvr"}

        def infer(self, request):
            del request
            return {}

    monkeypatch.setattr(evaluation_cli, "load_experiment_package", lambda _experiment: FakeExperiment())

    with pytest.raises(ValueError, match="local PCVR runs no longer accept --dataset-path"):
        evaluation_cli.main(
            [
                "infer",
                "--experiment",
                "experiments/pcvr/baseline",
                "--dataset-path",
                "/tmp/custom.parquet",
                "--result-dir",
                "/tmp/results",
            ]
        )


def test_evaluation_main_allows_explicit_dataset_for_bundle_pcvr_kind_experiment(monkeypatch, capsys) -> None:
    payload = {
        "checkpoint_path": "/tmp/model.safetensors",
        "schema_path": "/tmp/schema.json",
        "schema": {},
    }
    captured_request = {}

    class FakeExperiment:
        metadata = {"requires_dataset": True, "kind": "pcvr"}

        def infer(self, request):
            captured_request["dataset_path"] = request.dataset_path
            return payload

    monkeypatch.setenv("TAAC_BUNDLE_MODE", "1")
    monkeypatch.setattr(evaluation_cli, "load_experiment_package", lambda _experiment: FakeExperiment())

    exit_code = evaluation_cli.main(
        [
            "infer",
            "--experiment",
            "experiments/pcvr/baseline",
            "--dataset-path",
            "/tmp/custom.parquet",
            "--result-dir",
            "/tmp/results",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(captured_request["dataset_path"]) == "/tmp/custom.parquet"
    assert loads(captured.out) == payload


def test_evaluation_main_rejects_missing_dataset_for_non_pcvr_experiment(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation_cli,
        "load_experiment_package",
        lambda _experiment: SimpleNamespace(metadata={"requires_dataset": True, "kind": "maintenance"}),
    )

    with pytest.raises(ValueError, match="requires --dataset-path"):
        evaluation_cli.main(
            [
                "infer",
                "--experiment",
                "experiments/maintenance/example",
                "--result-dir",
                "/tmp/results",
            ]
        )
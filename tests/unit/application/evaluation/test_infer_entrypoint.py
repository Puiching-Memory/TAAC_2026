from __future__ import annotations

from collections.abc import Sequence

import pytest

import taac2026.application.evaluation.infer as infer_entrypoint


def test_infer_entrypoint_passes_taac_experiment_to_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_DATA_PATH", "/tmp/eval.parquet")
    monkeypatch.setenv("EVAL_RESULT_PATH", "/tmp/results")
    monkeypatch.setenv("MODEL_OUTPUT_PATH", "/tmp/model")
    monkeypatch.setenv("TAAC_SCHEMA_PATH", "/tmp/schema.json")
    monkeypatch.setenv("TAAC_EXPERIMENT", "experiments/symbiosis")
    captured: list[Sequence[str]] = []

    def fake_evaluation_main(argv: Sequence[str]) -> None:
        captured.append(list(argv))

    monkeypatch.setattr(infer_entrypoint, "evaluation_main", fake_evaluation_main)

    infer_entrypoint.main()

    assert captured == [[
        "infer",
        "--experiment",
        "experiments/symbiosis",
        "--dataset-path",
        "/tmp/eval.parquet",
        "--result-dir",
        "/tmp/results",
        "--checkpoint",
        "/tmp/model",
        "--schema-path",
        "/tmp/schema.json",
    ]]


def test_infer_entrypoint_passes_optional_runtime_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_DATA_PATH", "/tmp/eval.parquet")
    monkeypatch.setenv("EVAL_RESULT_PATH", "/tmp/results")
    monkeypatch.setenv("TAAC_INFER_BATCH_SIZE", "128")
    monkeypatch.setenv("TAAC_INFER_NUM_WORKERS", "8")
    monkeypatch.setenv("TAAC_INFER_AMP", "1")
    monkeypatch.setenv("TAAC_INFER_AMP_DTYPE", "float16")
    monkeypatch.setenv("TAAC_INFER_COMPILE", "true")
    captured: list[Sequence[str]] = []

    def fake_evaluation_main(argv: Sequence[str]) -> None:
        captured.append(list(argv))

    monkeypatch.setattr(infer_entrypoint, "evaluation_main", fake_evaluation_main)

    infer_entrypoint.main()

    assert captured == [[
        "infer",
        "--dataset-path",
        "/tmp/eval.parquet",
        "--result-dir",
        "/tmp/results",
        "--batch-size",
        "128",
        "--num-workers",
        "8",
        "--amp",
        "--amp-dtype",
        "float16",
        "--compile",
    ]]


def test_infer_entrypoint_passes_explicit_runtime_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_DATA_PATH", "/tmp/eval.parquet")
    monkeypatch.setenv("EVAL_RESULT_PATH", "/tmp/results")
    monkeypatch.setenv("TAAC_INFER_AMP", "0")
    monkeypatch.setenv("TAAC_INFER_COMPILE", "off")
    captured: list[Sequence[str]] = []

    def fake_evaluation_main(argv: Sequence[str]) -> None:
        captured.append(list(argv))

    monkeypatch.setattr(infer_entrypoint, "evaluation_main", fake_evaluation_main)

    infer_entrypoint.main()

    assert captured == [[
        "infer",
        "--dataset-path",
        "/tmp/eval.parquet",
        "--result-dir",
        "/tmp/results",
        "--no-amp",
        "--no-compile",
    ]]

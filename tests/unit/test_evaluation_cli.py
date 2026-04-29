from __future__ import annotations

import taac2026.application.evaluation.cli as evaluation_cli
from taac2026.application.evaluation.cli import parse_eval_args
from taac2026.infrastructure.io.json_utils import loads


def test_parse_eval_args_accepts_runtime_flags() -> None:
    args = parse_eval_args(
        [
            "infer",
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


def test_main_json_output_is_compact_single_line(monkeypatch, capsys) -> None:
    payload = {
        "checkpoint_path": "/tmp/model.pt",
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
            "--dataset-path",
            "/tmp/eval.parquet",
            "--result-dir",
            "/tmp/results",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "\n" not in captured.out.strip()
    assert '"schema":{"features":[{"name":"user_id"}]}' in captured.out.strip()
    assert loads(captured.out) == payload
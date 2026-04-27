from __future__ import annotations

from taac2026.application.evaluation.cli import parse_eval_args


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
from __future__ import annotations

from pathlib import Path

from taac2026.infrastructure.pcvr.training import parse_pcvr_train_args
from taac2026.application.training.cli import parse_train_args


def test_parse_train_args_forwards_experiment_specific_options() -> None:
    args, extra = parse_train_args(
        [
            "--experiment",
            "config/baseline",
            "--dataset-path",
            "/data/train",
            "--schema-path",
            "/data/schema.json",
            "--batch_size",
            "8",
        ]
    )

    assert args.experiment == "config/baseline"
    assert args.dataset_path == "/data/train"
    assert args.schema_path == "/data/schema.json"
    assert extra == ["--batch_size", "8"]


def test_parse_pcvr_train_args_accepts_runtime_flags(tmp_path: Path) -> None:
    args = parse_pcvr_train_args(
        ["--amp", "--amp-dtype", "float16", "--compile"],
        package_dir=tmp_path,
    )

    assert args.amp is True
    assert args.amp_dtype == "float16"
    assert args.compile is True

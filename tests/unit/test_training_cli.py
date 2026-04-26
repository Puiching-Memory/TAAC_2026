from __future__ import annotations

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

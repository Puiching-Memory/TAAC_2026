"""Unified training CLI used by the repository-level run.sh."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from taac2026.domain.config import TrainRequest, default_run_dir
from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.io.json_utils import dumps


def parse_train_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train a TAAC 2026 experiment package")
    parser.add_argument("--experiment", default="config/baseline", help="experiment package path or module")
    parser.add_argument("--dataset-path", default=None, help="parquet file or parquet directory; required for data-driven experiments")
    parser.add_argument("--schema-path", default=None, help="schema.json path; defaults to the dataset directory")
    parser.add_argument("--run-dir", default=None, help="checkpoint/output directory")
    return parser.parse_known_args(argv)


def _experiment_requires_dataset(experiment: object) -> bool:
    metadata = getattr(experiment, "metadata", {})
    if not isinstance(metadata, dict):
        return True
    requires_dataset = metadata.get("requires_dataset", True)
    return bool(requires_dataset) if isinstance(requires_dataset, bool) else True


def main(argv: Sequence[str] | None = None) -> int:
    args, extra_args = parse_train_args(argv)
    experiment = load_experiment_package(args.experiment)
    if args.dataset_path is None and _experiment_requires_dataset(experiment):
        raise ValueError(f"experiment {args.experiment!r} requires --dataset-path")
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.experiment)
    request = TrainRequest(
        experiment=args.experiment,
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
        schema_path=Path(args.schema_path) if args.schema_path else None,
        run_dir=run_dir,
        extra_args=tuple(extra_args),
    )
    summary = experiment.train(request) or {"run_dir": str(run_dir)}
    print(dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unified training CLI used by the repository-level run.sh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from taac2026.domain.config import TrainRequest, default_run_dir
from taac2026.infrastructure.experiments.loader import load_experiment_package


def parse_train_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, tuple[str, ...]]:
    parser = argparse.ArgumentParser(description="Train a TAAC 2026 experiment package")
    parser.add_argument("--experiment", default="config/baseline", help="experiment package path or module")
    parser.add_argument("--dataset-path", required=True, help="PCVR parquet file or parquet directory")
    parser.add_argument("--schema-path", default=None, help="schema.json path; defaults to the dataset directory")
    parser.add_argument("--run-dir", default=None, help="checkpoint/output directory")
    parser.add_argument("--json", action="store_true", help="print the training summary as JSON")
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args, extra_args = parse_train_args(argv)
    experiment = load_experiment_package(args.experiment)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.experiment)
    request = TrainRequest(
        experiment=args.experiment,
        dataset_path=Path(args.dataset_path),
        schema_path=Path(args.schema_path) if args.schema_path else None,
        run_dir=run_dir,
        extra_args=tuple(extra_args),
    )
    summary = experiment.train(request) or {"run_dir": str(run_dir)}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

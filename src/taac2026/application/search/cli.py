"""Small hyperparameter-search command facade.

The heavy search policy is intentionally left to experiment packages. This CLI
captures a reproducible study request and can be expanded without changing the
repository-level run.sh contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.io.files import write_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record or launch an experiment search request")
    parser.add_argument("--experiment", default="config/baseline")
    parser.add_argument("--study-dir", default="outputs/search/baseline")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--metric-name", default="metrics.auc")
    parser.add_argument("--direction", choices=("maximize", "minimize"), default="maximize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment = load_experiment_package(args.experiment)
    study_dir = Path(args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_name": experiment.name,
        "experiment": args.experiment,
        "trials": args.trials,
        "timeout_seconds": args.timeout_seconds,
        "metric_name": args.metric_name,
        "direction": args.direction,
        "seed": args.seed,
        "status": "recorded",
    }
    summary_path = write_json(study_dir / "study_request.json", payload)
    payload["study_request_path"] = str(summary_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

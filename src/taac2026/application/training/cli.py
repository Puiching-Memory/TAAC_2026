"""Unified training CLI used by the repository-level run.sh."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from taac2026.domain.requests import TrainRequest, default_run_dir
from taac2026.application.experiments.registry import load_experiment_package
from taac2026.infrastructure.io.json import dumps
from taac2026.infrastructure.io.rich_output import print_rich_summary
from taac2026.infrastructure.io.streams import write_stdout_line


@dataclass(frozen=True, slots=True)
class TrainCLIArgs:
    experiment: str
    dataset_path: str | None = None
    schema_path: str | None = None
    run_dir: str | None = None
    json: bool = False


def parse_train_args(argv: Sequence[str] | None = None) -> tuple[TrainCLIArgs, list[str]]:
    return tyro.cli(
        TrainCLIArgs,
        description="Train a TAAC 2026 experiment package",
        args=argv,
        return_unknown_args=True,
    )


def _experiment_requires_dataset(experiment: object) -> bool:
    metadata = getattr(experiment, "metadata", {})
    if not isinstance(metadata, dict):
        return True
    requires_dataset = metadata.get("requires_dataset", True)
    return bool(requires_dataset) if isinstance(requires_dataset, bool) else True


def _experiment_kind(experiment: object) -> str | None:
    metadata = getattr(experiment, "metadata", {})
    if not isinstance(metadata, dict):
        return None
    kind = metadata.get("kind")
    return kind if isinstance(kind, str) else None


def _is_bundle_mode() -> bool:
    return os.environ.get("TAAC_BUNDLE_MODE") == "1"


def _format_train_summary(summary: dict[str, Any], *, experiment: str) -> None:
    skip = {"telemetry", "training_telemetry", "evaluation_telemetry"}
    fields = [("Experiment", experiment)]
    for key, value in summary.items():
        if key in skip or isinstance(value, (dict, list)):
            continue
        label = key.replace("_", " ").title()
        fields.append((label, str(value)))
    telemetry = summary.get("telemetry") or summary.get("training_telemetry") or {}
    sections = []
    if isinstance(telemetry, dict):
        env_fields = []
        for env_key in ("device", "model_parameters", "elapsed_sec", "steps"):
            if env_key in telemetry:
                env_fields.append((env_key.replace("_", " "), str(telemetry[env_key])))
        if env_fields:
            sections.append(("Telemetry", env_fields))
    print_rich_summary("Training complete", fields, sections=sections, border_style="green")


def main(argv: Sequence[str] | None = None) -> int:
    args, extra_args = parse_train_args(argv)
    experiment = load_experiment_package(args.experiment)
    if _experiment_kind(experiment) == "pcvr" and not _is_bundle_mode() and args.dataset_path is not None:
        raise ValueError("local PCVR runs no longer accept --dataset-path; demo data is managed automatically")
    if args.dataset_path is None and _experiment_requires_dataset(experiment) and (_experiment_kind(experiment) != "pcvr" or _is_bundle_mode()):
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
    if args.json:
        write_stdout_line(dumps(summary))
    else:
        _format_train_summary(summary, experiment=args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

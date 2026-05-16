"""Unified local validation and platform inference CLI."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
import tyro

from taac2026.domain.requests import EvalRequest, InferRequest, default_run_dir
from taac2026.application.shared import experiment_kind, experiment_requires_dataset, is_bundle_mode
from taac2026.application.experiments.registry import load_experiment_package
from taac2026.infrastructure.io.json import dumps
from taac2026.infrastructure.io.rich_output import print_rich_summary
from taac2026.infrastructure.io.streams import write_stdout_line
from taac2026.infrastructure.logging import configure_logging


AMPDType = Literal["bfloat16", "float16"]


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, slots=True)
class EvalSingleCLIArgs:
    command: Literal["single"] = field(init=False, default="single")
    experiment: str
    dataset_path: str | None = None
    schema_path: str | None = None
    run_dir: str | None = None
    checkpoint: str | None = None
    output: str | None = None
    predictions_path: str | None = None
    batch_size: int = 256
    num_workers: int = 0
    device: str = field(default_factory=_default_device)
    amp: bool | None = None
    amp_dtype: AMPDType | None = None
    compile: bool | None = None
    json: bool = False


@dataclass(frozen=True, slots=True)
class EvalInferCLIArgs:
    command: Literal["infer"] = field(init=False, default="infer")
    experiment: str
    result_dir: str
    dataset_path: str | None = None
    schema_path: str | None = None
    checkpoint: str | None = None
    batch_size: int = 256
    num_workers: int = 0
    device: str = field(default_factory=_default_device)
    amp: bool | None = None
    amp_dtype: AMPDType | None = None
    compile: bool | None = None
    json: bool = False


EvalCLIArgs = (
    Annotated[EvalSingleCLIArgs, tyro.conf.subcommand(name="single")]
    | Annotated[EvalInferCLIArgs, tyro.conf.subcommand(name="infer")]
)


_OPTIONAL_RUNTIME_BOOL_FLAG_VALUES = {
    "--amp": "--amp=True",
    "--no-amp": "--amp=False",
    "--compile": "--compile=True",
    "--no-compile": "--compile=False",
}


def _normalize_optional_runtime_bool_args(argv: Sequence[str] | None) -> Sequence[str]:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    return tuple(_OPTIONAL_RUNTIME_BOOL_FLAG_VALUES.get(arg, arg) for arg in raw_argv)


def parse_eval_args(argv: Sequence[str] | None = None) -> EvalCLIArgs:
    return tyro.cli(
        EvalCLIArgs,
        description="Evaluate or run inference for a TAAC 2026 experiment",
        args=_normalize_optional_runtime_bool_args(argv),
    )


def _format_eval_summary(payload: dict[str, Any], *, title: str) -> None:
    skip = {"telemetry", "training_telemetry", "evaluation_telemetry", "inference_telemetry", "metrics"}
    fields = []
    for key, value in payload.items():
        if key in skip or isinstance(value, (dict, list)):
            continue
        label = key.replace("_", " ").title()
        fields.append((label, str(value)))
    metrics = payload.get("metrics") or {}
    sections = []
    if isinstance(metrics, dict):
        metric_fields = [(k, f"{v:.5f}" if isinstance(v, float) else str(v)) for k, v in metrics.items() if not isinstance(v, (dict, list))]
        if metric_fields:
            sections.append(("Metrics", metric_fields))
    print_rich_summary(title, fields, sections=sections, border_style="cyan")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_eval_args(argv)
    experiment = load_experiment_package(args.experiment)
    if args.dataset_path is None and experiment_requires_dataset(experiment) and (experiment_kind(experiment) != "pcvr" or is_bundle_mode()):
        raise ValueError(f"experiment {args.experiment!r} requires --dataset-path")
    if args.command == "single":
        run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.experiment)
        run_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(run_dir / "evaluate.log")
        request = EvalRequest(
            experiment=args.experiment,
            dataset_path=Path(args.dataset_path) if args.dataset_path else None,
            schema_path=Path(args.schema_path) if args.schema_path else None,
            run_dir=run_dir,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            output_path=Path(args.output) if args.output else None,
            predictions_path=Path(args.predictions_path) if args.predictions_path else None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            compile=args.compile,
        )
        payload = experiment.evaluate(request)
        title = "Evaluation complete"
    else:
        result_dir = Path(args.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(result_dir / "infer.log")
        request = InferRequest(
            experiment=args.experiment,
            dataset_path=Path(args.dataset_path) if args.dataset_path else None,
            schema_path=Path(args.schema_path) if args.schema_path else None,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            result_dir=result_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            compile=args.compile,
        )
        payload = experiment.infer(request)
        title = "Inference complete"

    if args.json:
        write_stdout_line(dumps(payload))
    else:
        _format_eval_summary(payload, title=title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unified local validation and platform inference CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

import torch

from taac2026.domain.config import EvalRequest, InferRequest, default_run_dir
from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.training.runtime import AMP_DTYPE_CHOICES


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _add_runtime_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp-dtype", default=None, choices=AMP_DTYPE_CHOICES)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)


def parse_eval_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or run inference for a TAAC 2026 experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="evaluate one checkpoint on a labeled parquet dataset")
    single.add_argument("--experiment", default="config/baseline")
    single.add_argument("--dataset-path", required=True)
    single.add_argument("--schema-path", default=None)
    single.add_argument("--run-dir", default=None)
    single.add_argument("--checkpoint", default=None)
    single.add_argument("--output", default=None)
    single.add_argument("--predictions-path", default=None)
    single.add_argument("--batch-size", type=int, default=256)
    single.add_argument("--num-workers", type=int, default=0)
    single.add_argument("--device", default=_default_device())
    _add_runtime_execution_args(single)
    single.add_argument("--json", action="store_true")

    infer = subparsers.add_parser("infer", help="write platform predictions.json")
    infer.add_argument("--experiment", default="config/baseline")
    infer.add_argument("--dataset-path", required=True)
    infer.add_argument("--schema-path", default=None)
    infer.add_argument("--checkpoint", default=None)
    infer.add_argument("--result-dir", required=True)
    infer.add_argument("--batch-size", type=int, default=256)
    infer.add_argument("--num-workers", type=int, default=0)
    infer.add_argument("--device", default=_default_device())
    _add_runtime_execution_args(infer)
    infer.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_eval_args(argv)
    experiment = load_experiment_package(args.experiment)
    if args.command == "single":
        run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.experiment)
        request = EvalRequest(
            experiment=args.experiment,
            dataset_path=Path(args.dataset_path),
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
    else:
        request = InferRequest(
            experiment=args.experiment,
            dataset_path=Path(args.dataset_path),
            schema_path=Path(args.schema_path) if args.schema_path else None,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            result_dir=Path(args.result_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            compile=args.compile,
        )
        payload = experiment.infer(request)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

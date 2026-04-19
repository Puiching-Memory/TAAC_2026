from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from rich import box
from rich.table import Table

from ...infrastructure.experiments.loader import load_experiment_package
from ...infrastructure.io.console import (
    configure_logging,
    create_progress_bar,
    print_summary_table,
    stdout_console,
)
from .service import _sort_records, evaluate_checkpoint


def _apply_runtime_optimization_args(experiment, args: argparse.Namespace) -> None:
    if getattr(args, "compile", False):
        experiment.train.enable_torch_compile = True
    if getattr(args, "compile_backend", None) is not None:
        experiment.train.enable_torch_compile = True
        experiment.train.torch_compile_backend = args.compile_backend
    if getattr(args, "compile_mode", None) is not None:
        experiment.train.enable_torch_compile = True
        experiment.train.torch_compile_mode = args.compile_mode
    if getattr(args, "amp", False):
        experiment.train.enable_amp = True
        experiment.train.amp_dtype = args.amp_dtype


def _print_batch_table(records: list[dict[str, Any]]) -> None:
    table = Table(title="taac-evaluate batch", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    table.add_column("Experiment", style="white")
    table.add_column("Quantization", justify="center")
    table.add_column("AUC", justify="right")
    table.add_column("PR AUC", justify="right")
    table.add_column("Latency ms/sample", justify="right")
    for index, record in enumerate(records, start=1):
        table.add_row(
            str(index),
            str(record.get("experiment_path") or record.get("experiment") or record.get("model_name")),
            str(record.get("quantization", {}).get("mode", "none")),
            f"{float(record.get('metrics', {}).get('auc', record.get('auc', 0.0))):.6f}",
            f"{float(record.get('metrics', {}).get('pr_auc', record.get('pr_auc', 0.0))):.6f}",
            f"{float(record.get('mean_latency_ms_per_sample', 0.0)):.4f}",
        )
    stdout_console.print(table)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TAAC 2026 experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser("single")
    single_parser.add_argument("--experiment", required=True)
    single_parser.add_argument("--checkpoint")
    single_parser.add_argument("--output-path")
    single_parser.add_argument("--run-dir")
    single_parser.add_argument("--compile", action="store_true")
    single_parser.add_argument("--compile-backend")
    single_parser.add_argument("--compile-mode")
    single_parser.add_argument("--amp", action="store_true")
    single_parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    single_parser.add_argument("--quantize", choices=("none", "int8"), default="none")
    single_parser.add_argument("--export-mode", choices=("none", "torch-export"), default="none")
    single_parser.add_argument("--export-path")

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("--experiment-paths", nargs="+", required=True)
    batch_parser.add_argument("--compile", action="store_true")
    batch_parser.add_argument("--compile-backend")
    batch_parser.add_argument("--compile-mode")
    batch_parser.add_argument("--amp", action="store_true")
    batch_parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    batch_parser.add_argument("--quantize", choices=("none", "int8"), default="none")
    batch_parser.add_argument("--export-mode", choices=("none", "torch-export"), default="none")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    if args.command == "single":
        experiment = load_experiment_package(args.experiment)
        if args.run_dir:
            experiment.train.output_dir = args.run_dir
        _apply_runtime_optimization_args(experiment, args)
        checkpoint = args.checkpoint
        if checkpoint is None and args.run_dir:
            checkpoint = str(Path(args.run_dir) / "best.pt")
        report = evaluate_checkpoint(
            experiment_path=args.experiment,
            checkpoint_path=checkpoint,
            output_path=args.output_path,
            experiment=experiment,
            quantization_mode=args.quantize,
            export_mode=args.export_mode,
            export_path=args.export_path,
        )
        print_summary_table(
            "taac-evaluate single",
            [
                ("experiment", report["experiment"]),
                ("experiment_path", report["experiment_path"]),
                ("checkpoint_path", report["checkpoint_path"]),
                ("device", report["device"]),
                ("quantization", report["quantization"]["mode"]),
                ("export", report["export"]["mode"]),
                ("export_artifact", report["export"].get("artifact_path") or "-"),
                ("loss", f"{float(report['loss']):.6f}"),
                ("auc", f"{float(report['metrics'].get('auc', 0.0)):.6f}"),
                ("pr_auc", f"{float(report['metrics'].get('pr_auc', 0.0)):.6f}"),
                ("mean_latency_ms_per_sample", f"{float(report['mean_latency_ms_per_sample']):.4f}"),
                ("p95_latency_ms_per_sample", f"{float(report['p95_latency_ms_per_sample']):.4f}"),
            ],
        )
        return 0

    progress_bar = None
    if sys.stderr.isatty():
        progress_bar = create_progress_bar(
            total=len(args.experiment_paths),
            description="taac-evaluate[batch]",
        )
    reports: list[dict[str, Any]] = []
    try:
        for experiment_path in args.experiment_paths:
            experiment = load_experiment_package(experiment_path)
            _apply_runtime_optimization_args(experiment, args)
            reports.append(
                evaluate_checkpoint(
                    experiment_path=experiment_path,
                    experiment=experiment,
                    quantization_mode=args.quantize,
                    export_mode=args.export_mode,
                )
            )
            if progress_bar is not None:
                progress_bar.update()
                progress_bar.set_postfix({"last": experiment_path}, refresh=False)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    ranked = _sort_records(reports)
    _print_batch_table(ranked)
    return 0


__all__ = ["main", "parse_args"]

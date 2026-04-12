from __future__ import annotations

import argparse
import sys

from ...infrastructure.experiments.loader import load_experiment_package
from ...infrastructure.io.console import configure_logging, print_summary_table
from .service import run_training


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TAAC 2026 experiment")
    parser.add_argument("--experiment", required=True, help="Experiment package path or module path")
    parser.add_argument("--run-dir", help="Override output directory")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for model execution")
    parser.add_argument("--compile-backend", help="Override torch.compile backend")
    parser.add_argument("--compile-mode", help="Override torch.compile mode")
    parser.add_argument("--amp", action="store_true", help="Enable AMP autocast during model execution")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16", help="Autocast dtype when AMP is enabled")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_train_args(argv)
    experiment = load_experiment_package(args.experiment)
    if args.run_dir:
        experiment.train.output_dir = args.run_dir
    if args.compile:
        experiment.train.enable_torch_compile = True
    if args.compile_backend is not None:
        experiment.train.enable_torch_compile = True
        experiment.train.torch_compile_backend = args.compile_backend
    if args.compile_mode is not None:
        experiment.train.enable_torch_compile = True
        experiment.train.torch_compile_mode = args.compile_mode
    if args.amp:
        experiment.train.enable_amp = True
        experiment.train.amp_dtype = args.amp_dtype
    summary = run_training(
        experiment,
        experiment_path=args.experiment,
        show_progress=bool(sys.stderr.isatty()),
    )
    print_summary_table(
        "taac-train",
        [
            ("experiment", experiment.name),
            ("output_dir", experiment.train.output_dir),
            ("best_epoch", summary["best_epoch"]),
            ("best_val_auc", f"{float(summary['best_val_auc']):.6f}"),
            ("auc", f"{float(summary['metrics'].get('auc', 0.0)):.6f}"),
            ("pr_auc", f"{float(summary['metrics'].get('pr_auc', 0.0)):.6f}"),
            ("mean_latency_ms_per_sample", f"{float(summary['mean_latency_ms_per_sample']):.4f}"),
            ("parameter_size_mb", f"{float(summary['model_profile']['parameter_size_mb']):.2f}"),
            ("estimated_end_to_end_tflops_total", f"{float(summary['compute_profile']['estimated_end_to_end_tflops_total']):.4f}"),
            ("ncu_available", str(summary["profiling"]["external_profilers"]["tools"]["ncu"]["available"])),
            ("nsys_available", str(summary["profiling"]["external_profilers"]["tools"]["nsys"]["available"])),
        ],
    )
    return 0


__all__ = ["main", "parse_train_args"]

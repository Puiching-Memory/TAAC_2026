"""Benchmark experiment packages and render model-performance figures."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.ticker import ScalarFormatter
from tqdm.auto import tqdm

import taac2026.infrastructure.pcvr.data as pcvr_data
from taac2026.domain.config import EvalRequest, TrainRequest
from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.io.files import repo_root
from taac2026.infrastructure.pcvr.protocol import batch_to_model_input, build_pcvr_model, parse_seq_max_lens, resolve_schema_path
from taac2026.infrastructure.pcvr.training import parse_pcvr_train_args


DEFAULT_SIZE_FIGURE = Path("figures/model_performance_vs_size.svg")
DEFAULT_COMPUTE_FIGURE = Path("figures/model_performance_vs_compute.svg")
DEFAULT_REPORT_PATH = Path("outputs/reports/model_performance_smoke.json")
DEFAULT_RUN_ROOT = Path("outputs/model_performance_runs")


@dataclass(slots=True)
class BenchmarkResult:
    experiment_path: str
    experiment_name: str
    label: str
    run_dir: str
    auc: float
    logloss: float
    sample_count: int
    total_params: int
    total_params_millions: float
    estimated_step_flops: float
    estimated_training_compute_tflops: float
    batch_size: int
    train_steps_per_epoch: int
    num_epochs: int


def discover_experiment_paths(config_root: Path) -> list[str]:
    experiment_paths: list[str] = []
    root = config_root.expanduser().resolve()
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("__"):
            continue
        required = ("__init__.py", "model.py", "ns_groups.json")
        if all((child / name).exists() for name in required):
            experiment_paths.append(child.relative_to(root.parent).as_posix())
    return experiment_paths


def compute_pareto_frontier(rows: list[dict[str, Any]], *, x_key: str, y_key: str) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    best_y = float("-inf")
    for row in sorted(rows, key=lambda item: float(item[x_key])):
        current_y = float(row[y_key])
        if current_y > best_y:
            frontier.append(row)
            best_y = current_y
    return frontier


def _load_model_module(experiment_path: str) -> Any:
    model_path = (repo_root() / experiment_path / "model.py").resolve()
    spec = importlib.util.spec_from_file_location(experiment_path.replace("/", "_") + "_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load model module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _benchmark_override_args(args: argparse.Namespace) -> tuple[str, ...]:
    forwarded: list[str] = [
        "--num_epochs",
        str(args.num_epochs),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if args.batch_size is not None:
        forwarded.extend(["--batch_size", str(args.batch_size)])
    return tuple(forwarded)


def _resolved_schema_path(dataset_path: Path, schema_path: Path | None) -> Path:
    return resolve_schema_path(dataset_path, schema_path, Path.cwd())


def _build_profile_components(
    experiment_path: str,
    dataset_path: Path,
    schema_path: Path,
    run_dir: Path,
    override_args: tuple[str, ...],
) -> tuple[argparse.Namespace, Any, Any, torch.utils.data.DataLoader]:
    experiment = load_experiment_package(experiment_path)
    if experiment.package_dir is None:
        raise ValueError(f"experiment {experiment_path!r} does not declare package_dir")

    model_module = _load_model_module(experiment_path)
    forwarded_args = [
        "--data_dir",
        str(dataset_path),
        "--schema_path",
        str(schema_path),
        "--ckpt_dir",
        str(run_dir),
        "--log_dir",
        str(run_dir / "logs"),
        "--tf_events_dir",
        str(run_dir / "tensorboard"),
        *experiment.default_train_args,
        *override_args,
    ]
    parsed_args = parse_pcvr_train_args(forwarded_args, package_dir=experiment.package_dir)
    config = vars(parsed_args).copy()
    seq_max_lens = parse_seq_max_lens(str(parsed_args.seq_max_lens))
    train_loader, _valid_loader, dataset = pcvr_data.get_pcvr_data(
        data_dir=str(dataset_path),
        schema_path=str(schema_path),
        batch_size=parsed_args.batch_size,
        valid_ratio=parsed_args.valid_ratio,
        train_ratio=parsed_args.train_ratio,
        num_workers=parsed_args.num_workers,
        buffer_batches=parsed_args.buffer_batches,
        seed=parsed_args.seed,
        seq_max_lens=seq_max_lens,
    )
    model = build_pcvr_model(
        model_module=model_module,
        model_class_name=experiment.metadata["model_class"],
        data_module=pcvr_data,
        dataset=dataset,
        config=config,
        package_dir=experiment.package_dir,
        checkpoint_dir=run_dir,
    )
    return parsed_args, model_module, model, train_loader


def _sum_profiler_flops(profile: Any) -> float:
    total_flops = 0.0
    for event in profile.key_averages():
        event_flops = getattr(event, "flops", 0) or 0
        total_flops += float(event_flops)
    return total_flops


def _estimate_step_flops(
    model: torch.nn.Module,
    model_module: Any,
    batch: dict[str, Any],
    *,
    device: torch.device,
) -> float:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    model = model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    label = batch["label"].to(device).float()
    with torch.profiler.profile(activities=activities, with_flops=True) as profile:
        model_input = batch_to_model_input(batch, model_module.ModelInput, device)
        logits = model(model_input).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()
    estimated_step_flops = _sum_profiler_flops(profile)
    model.zero_grad(set_to_none=True)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if estimated_step_flops <= 0:
        batch_size = int(label.shape[0])
        estimated_step_flops = float(sum(parameter.numel() for parameter in model.parameters()) * batch_size * 6)
    return estimated_step_flops


def _subtitle(dataset_path: Path, num_epochs: int) -> str:
    return f"{dataset_path.name}, {num_epochs}-epoch smoke"


def _footer() -> str:
    return "single-row-group sample; train/valid reuse in smoke benchmark"


def _plot_offsets(count: int) -> list[tuple[int, int]]:
    base = [(10, 10), (10, -14), (-10, 10), (-10, -14), (14, 16), (14, -18), (-14, 16), (-14, -18)]
    return [base[index % len(base)] for index in range(count)]


def _render_plot(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    x_label: str,
    title: str,
    subtitle: str,
    footer: str,
    output_path: Path,
    xscale: str = "linear",
    footer_note: str,
) -> None:
    sorted_rows = sorted(rows, key=lambda item: float(item[x_key]))
    frontier = compute_pareto_frontier(sorted_rows, x_key=x_key, y_key="auc")
    frontier_labels = {row["label"] for row in frontier}

    fig, ax = plt.subplots(figsize=(11.5, 7.5), dpi=200)
    fig.patch.set_facecolor("#111418")
    ax.set_facecolor("#111418")
    ax.grid(True, color="#2b313c", linewidth=0.8, alpha=0.85)
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#111418")

    y_values = [float(row["auc"]) for row in sorted_rows]
    x_values = [float(row[x_key]) for row in sorted_rows]
    y_min = min(y_values) - 0.01
    y_max = max(y_values) + 0.01

    frontier_x = [float(row[x_key]) for row in frontier]
    frontier_y = [float(row["auc"]) for row in frontier]
    ax.fill_between(frontier_x, frontier_y, y_min, color="#2563eb", alpha=0.22)
    ax.plot(frontier_x, frontier_y, color="#3b82f6", linewidth=2.5, alpha=0.95)

    colors = ["#3b82f6" if row["label"] in frontier_labels else "#c7d0dc" for row in sorted_rows]
    edges = ["#93c5fd" if row["label"] in frontier_labels else "none" for row in sorted_rows]
    ax.scatter(x_values, y_values, s=64, c=colors, edgecolors=edges, linewidths=1.5, alpha=0.98)

    offsets = _plot_offsets(len(sorted_rows))
    for index, row in enumerate(sorted_rows):
        dx, dy = offsets[index]
        ax.annotate(
            row["label"],
            (float(row[x_key]), float(row["auc"])),
            xytext=(dx, dy),
            textcoords="offset points",
            color="#4f9cff" if row["label"] in frontier_labels else "#c7d0dc",
            fontsize=10,
            ha="left" if dx >= 0 else "right",
        )

    ax.set_xlabel(x_label, color="#e5e7eb", fontsize=12.5, labelpad=14)
    ax.set_ylabel("Validation AUC", color="#e5e7eb", fontsize=12.5, labelpad=12)
    ax.set_ylim(y_min, y_max)
    ax.set_xscale(xscale)
    if xscale == "log":
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        ax.xaxis.set_major_formatter(formatter)

    fig.suptitle(title, color="#f8fafc", fontsize=18, fontweight="bold", y=0.98)
    ax.set_title(subtitle, color="#94a3b8", fontsize=10.5, pad=8)
    fig.text(0.98, 0.035, footer_note, color="#94a3b8", ha="right", fontsize=8.5)
    fig.text(0.5, 0.012, footer, color="#94a3b8", ha="center", fontsize=8.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def benchmark_experiment(
    experiment_path: str,
    *,
    dataset_path: Path,
    schema_path: Path,
    run_root: Path,
    override_args: tuple[str, ...],
    force: bool,
) -> BenchmarkResult:
    experiment = load_experiment_package(experiment_path)
    label = Path(experiment_path).name
    run_dir = (run_root / label).resolve()
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    parsed_args, model_module, model, train_loader = _build_profile_components(
        experiment_path,
        dataset_path,
        schema_path,
        run_dir,
        override_args,
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    train_steps_per_epoch = len(train_loader)
    sample_batch = next(iter(train_loader))
    runtime_device = torch.device(parsed_args.device)
    estimated_step_flops = _estimate_step_flops(model, model_module, sample_batch, device=runtime_device)
    estimated_training_compute_tflops = estimated_step_flops * train_steps_per_epoch * parsed_args.num_epochs / 1e12

    request = TrainRequest(
        experiment=experiment_path,
        dataset_path=dataset_path,
        schema_path=schema_path,
        run_dir=run_dir,
        extra_args=override_args,
    )
    experiment.train(request)

    evaluation = experiment.evaluate(
        EvalRequest(
            experiment=experiment_path,
            dataset_path=dataset_path,
            schema_path=schema_path,
            run_dir=run_dir,
            batch_size=parsed_args.batch_size,
            num_workers=0,
            device=parsed_args.device,
        )
    )
    metrics = evaluation["metrics"]

    if runtime_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BenchmarkResult(
        experiment_path=experiment_path,
        experiment_name=experiment.name,
        label=label,
        run_dir=str(run_dir),
        auc=float(metrics["auc"]),
        logloss=float(metrics["logloss"]),
        sample_count=int(metrics["sample_count"]),
        total_params=total_params,
        total_params_millions=total_params / 1_000_000.0,
        estimated_step_flops=estimated_step_flops,
        estimated_training_compute_tflops=estimated_training_compute_tflops,
        batch_size=int(parsed_args.batch_size),
        train_steps_per_epoch=int(train_steps_per_epoch),
        num_epochs=int(parsed_args.num_epochs),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark experiment packages and render model-performance plots")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--schema-path", default=None)
    parser.add_argument("--config-root", default="config")
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--size-output", default=str(DEFAULT_SIZE_FIGURE))
    parser.add_argument("--compute-output", default=str(DEFAULT_COMPUTE_FIGURE))
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    schema_path = _resolved_schema_path(dataset_path, Path(args.schema_path).expanduser().resolve() if args.schema_path else None)
    config_root = Path(args.config_root).expanduser().resolve()
    experiment_paths = args.experiments or discover_experiment_paths(config_root)
    if not experiment_paths:
        raise ValueError(f"no experiment packages found under {config_root}")

    override_args = _benchmark_override_args(args)
    report_rows: list[dict[str, Any]] = []
    progress = tqdm(experiment_paths, desc="Benchmark experiments", unit="exp", dynamic_ncols=True)
    try:
        for experiment_path in progress:
            progress.set_postfix_str(Path(experiment_path).name)
            result = benchmark_experiment(
                experiment_path,
                dataset_path=dataset_path,
                schema_path=schema_path,
                run_root=Path(args.run_root),
                override_args=override_args,
                force=args.force,
            )
            report_rows.append(asdict(result))
    finally:
        progress.close()

    report_payload = {
        "dataset_path": str(dataset_path),
        "schema_path": str(schema_path),
        "device": args.device,
        "benchmark_override_args": list(override_args),
        "results": report_rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    subtitle = _subtitle(dataset_path, args.num_epochs)
    footer = _footer()
    _render_plot(
        report_rows,
        x_key="total_params_millions",
        x_label="Total Model Size (Million Parameters)",
        title="Model Performance VS Size",
        subtitle=subtitle,
        footer=footer,
        footer_note="highlight = pareto frontier by smaller size and higher AUC",
        output_path=Path(args.size_output),
    )
    _render_plot(
        report_rows,
        x_key="estimated_training_compute_tflops",
        x_label="Estimated End-to-End Training Compute (TFLOPs)",
        title="Model Performance VS Compute",
        subtitle=subtitle,
        footer=footer,
        footer_note="highlight = pareto frontier by lower training compute and higher AUC",
        output_path=Path(args.compute_output),
        xscale="log",
    )

    print(json.dumps({
        "report": str(report_path),
        "size_figure": str(Path(args.size_output)),
        "compute_figure": str(Path(args.compute_output)),
        "experiments": experiment_paths,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

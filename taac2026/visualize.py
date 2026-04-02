from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def _configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.frameon": False,
        }
    )


_configure_matplotlib()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_output_dir(input_path: str | Path, output_dir: str | Path | None) -> Path:
    if output_dir:
        return ensure_dir(output_dir)
    return ensure_dir(Path(input_path).parent / "plots")


def _save_figure(figure: plt.Figure, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _safe_name(value: str) -> str:
    cleaned = []
    for character in value.lower():
        if character.isalnum():
            cleaned.append(character)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "plot"


def _ci_error(records: list[dict[str, Any]], value_key: str, ci_key: str) -> np.ndarray | None:
    lowers: list[float] = []
    uppers: list[float] = []
    for record in records:
        center = float(record.get(value_key, 0.0))
        ci = record.get(ci_key)
        if not isinstance(ci, dict) or not ci.get("defined", True):
            return None
        lower = ci.get("lower")
        upper = ci.get("upper")
        if lower is None or upper is None:
            return None
        lowers.append(center - float(lower))
        uppers.append(float(upper) - center)
    return np.asarray([lowers, uppers], dtype=np.float64)


def _summary_ci_error(records: list[dict[str, Any]], key: str) -> np.ndarray | None:
    lowers: list[float] = []
    uppers: list[float] = []
    for record in records:
        interval = record.get(key)
        if not isinstance(interval, dict):
            return None
        mean_value = interval.get("mean")
        lower = interval.get("lower")
        upper = interval.get("upper")
        if mean_value is None or lower is None or upper is None:
            return None
        lowers.append(float(mean_value) - float(lower))
        uppers.append(float(upper) - float(mean_value))
    return np.asarray([lowers, uppers], dtype=np.float64)


def plot_batch_report(input_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    payload = _load_json(input_path)
    records = sorted(payload.get("records", []), key=lambda item: float(item.get("auc", 0.0)), reverse=True)
    if not records:
        raise ValueError("experiment_report.json 中没有 records，无法绘图。")

    output_root = _resolve_output_dir(input_path, output_dir)
    labels = [record["experiment_id"] for record in records]
    positions = np.arange(len(records))
    bar_height = 0.36

    auc = np.asarray([float(record["auc"]) for record in records], dtype=np.float64)
    pr_auc = np.asarray([float(record["pr_auc"]) for record in records], dtype=np.float64)
    brier = np.asarray([float(record["brier"]) for record in records], dtype=np.float64)
    logloss = np.asarray([float(record["logloss"]) for record in records], dtype=np.float64)
    mean_latency = np.asarray([float(record["mean_latency_ms_per_sample"]) for record in records], dtype=np.float64)
    p95_latency = np.asarray([float(record["p95_latency_ms_per_sample"]) for record in records], dtype=np.float64)

    figure, axes = plt.subplots(3, 1, figsize=(14, 14), sharey=True, constrained_layout=True)

    axes[0].barh(positions - bar_height / 2, auc, height=bar_height, color="#0F766E", label="AUC")
    axes[0].barh(positions + bar_height / 2, pr_auc, height=bar_height, color="#FB7185", label="PR-AUC")
    auc_error = _ci_error(records, "auc", "auc_ci95")
    if auc_error is not None:
        axes[0].errorbar(auc, positions - bar_height / 2, xerr=auc_error, fmt="none", ecolor="#134E4A", capsize=3)
    pr_auc_error = _ci_error(records, "pr_auc", "pr_auc_ci95")
    if pr_auc_error is not None:
        axes[0].errorbar(pr_auc, positions + bar_height / 2, xerr=pr_auc_error, fmt="none", ecolor="#9F1239", capsize=3)
    axes[0].set_title("Ranking Metrics")
    axes[0].set_xlabel("Score")
    axes[0].legend(loc="lower right")

    axes[1].barh(positions - bar_height / 2, brier, height=bar_height, color="#F59E0B", label="Brier")
    axes[1].barh(positions + bar_height / 2, logloss, height=bar_height, color="#7C3AED", label="Logloss")
    axes[1].set_title("Calibration Metrics")
    axes[1].set_xlabel("Lower is better")
    axes[1].legend(loc="lower right")

    axes[2].barh(positions - bar_height / 2, mean_latency, height=bar_height, color="#2563EB", label="Mean latency")
    axes[2].barh(positions + bar_height / 2, p95_latency, height=bar_height, color="#9333EA", label="P95 latency")
    axes[2].set_title("Latency Metrics")
    axes[2].set_xlabel("ms / sample")
    axes[2].legend(loc="lower right")

    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)

    dashboard_path = _save_figure(figure, output_root / "experiment_dashboard.png")

    pareto_figure, pareto_axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    scatter = pareto_axis.scatter(
        mean_latency,
        auc,
        c=pr_auc,
        cmap="viridis",
        s=180,
        edgecolors="#111827",
        linewidths=0.8,
    )
    for record, x_value, y_value in zip(records, mean_latency, auc):
        pareto_axis.annotate(record["experiment_id"], (x_value, y_value), textcoords="offset points", xytext=(6, 5))
    pareto_axis.set_title("AUC vs Latency Pareto View")
    pareto_axis.set_xlabel("Mean latency (ms / sample)")
    pareto_axis.set_ylabel("AUC")
    colorbar = pareto_figure.colorbar(scatter, ax=pareto_axis)
    colorbar.set_label("PR-AUC")
    pareto_axis.grid(alpha=0.25)
    pareto_path = _save_figure(pareto_figure, output_root / "experiment_pareto.png")
    return [dashboard_path, pareto_path]


def plot_summary(input_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    payload = _load_json(input_path)
    history = payload.get("history", [])
    if not history:
        raise ValueError("summary.json 中没有 history，无法绘图。")

    output_root = _resolve_output_dir(input_path, output_dir)
    epochs = np.asarray([int(record["epoch"]) for record in history], dtype=np.int64)
    train_loss = np.asarray([float(record["train_loss"]) for record in history], dtype=np.float64)
    val_loss = np.asarray([float(record["val_loss"]) for record in history], dtype=np.float64)
    val_auc = np.asarray([float(record["val_auc"]) for record in history], dtype=np.float64)
    best_epoch = int(payload.get("best_epoch", epochs[int(np.argmax(val_auc))]))

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(epochs, train_loss, marker="o", color="#0F766E", label="Train loss")
    axes[0].plot(epochs, val_loss, marker="o", color="#F59E0B", label="Val loss")
    axes[0].axvline(best_epoch, color="#475569", linestyle="--", linewidth=1.2, label=f"Best epoch {best_epoch}")
    axes[0].set_title(f"Training Loss Curves: {payload.get('model_name', 'model')}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best")

    axes[1].plot(epochs, val_auc, marker="o", color="#2563EB", label="Val AUC")
    best_index = int(np.argmax(val_auc))
    axes[1].scatter([epochs[best_index]], [val_auc[best_index]], color="#BE123C", s=80, zorder=3)
    axes[1].annotate(f"best={val_auc[best_index]:.4f}", (epochs[best_index], val_auc[best_index]), textcoords="offset points", xytext=(8, 6))
    axes[1].set_title("Validation AUC by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].legend(loc="best")

    for axis in axes:
        axis.grid(alpha=0.25)

    return [_save_figure(figure, output_root / "training_history.png")]


def plot_truncation_sweep(input_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    payload = _load_json(input_path)
    summary = sorted(payload.get("summary", []), key=lambda item: int(item.get("max_seq_len", 0)))
    runs = payload.get("runs", [])
    if not summary:
        raise ValueError("report.json 中没有 summary，无法绘图。")

    output_root = _resolve_output_dir(input_path, output_dir)
    seq_lens = np.asarray([int(record["max_seq_len"]) for record in summary], dtype=np.int64)

    auc_mean = np.asarray([float(record["auc_ci95"]["mean"]) for record in summary], dtype=np.float64)
    pr_auc_mean = np.asarray([float(record["pr_auc_ci95"]["mean"]) for record in summary], dtype=np.float64)
    brier_mean = np.asarray([float(record["brier_ci95"]["mean"]) for record in summary], dtype=np.float64)
    logloss_mean = np.asarray([float(record["logloss_ci95"]["mean"]) for record in summary], dtype=np.float64)
    mean_latency = np.asarray([float(record["mean_latency_ci95"]["mean"]) for record in summary], dtype=np.float64)
    p95_latency = np.asarray([float(record["p95_latency_ci95"]["mean"]) for record in summary], dtype=np.float64)

    figure, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    axes[0].errorbar(seq_lens, auc_mean, yerr=_summary_ci_error(summary, "auc_ci95"), marker="o", color="#0F766E", capsize=4, label="AUC")
    axes[0].errorbar(seq_lens, pr_auc_mean, yerr=_summary_ci_error(summary, "pr_auc_ci95"), marker="o", color="#BE123C", capsize=4, label="PR-AUC")
    axes[0].set_title("Ranking Metrics by max_seq_len")
    axes[0].set_xlabel("max_seq_len")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="best")

    axes[1].errorbar(seq_lens, brier_mean, yerr=_summary_ci_error(summary, "brier_ci95"), marker="o", color="#F59E0B", capsize=4, label="Brier")
    axes[1].errorbar(seq_lens, logloss_mean, yerr=_summary_ci_error(summary, "logloss_ci95"), marker="o", color="#7C3AED", capsize=4, label="Logloss")
    axes[1].set_title("Calibration Metrics by max_seq_len")
    axes[1].set_xlabel("max_seq_len")
    axes[1].set_ylabel("Lower is better")
    axes[1].legend(loc="best")

    axes[2].plot(seq_lens, mean_latency, marker="o", color="#2563EB", label="Mean latency")
    axes[2].plot(seq_lens, p95_latency, marker="o", color="#9333EA", label="P95 latency")
    axes[2].set_title("Latency by max_seq_len")
    axes[2].set_xlabel("max_seq_len")
    axes[2].set_ylabel("ms / sample")
    axes[2].legend(loc="best")

    for axis in axes:
        axis.grid(alpha=0.25)

    summary_path = _save_figure(figure, output_root / "truncation_summary.png")

    raw_figure, raw_axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    for metric_name, axis, color in [
        ("best_val_auc", raw_axes[0], "#0F766E"),
        ("best_val_pr_auc", raw_axes[1], "#BE123C"),
    ]:
        for seq_len in seq_lens:
            seq_runs = [run for run in runs if int(run.get("max_seq_len", 0)) == int(seq_len)]
            values = [float(run[metric_name]) for run in seq_runs]
            if not values:
                continue
            jitter = np.linspace(-4.0, 4.0, num=len(values)) if len(values) > 1 else np.asarray([0.0])
            axis.scatter(np.full(len(values), seq_len, dtype=np.float64) + jitter, values, color=color, alpha=0.75, s=60)
        axis.set_xlabel("max_seq_len")
        axis.grid(alpha=0.25)
    raw_axes[0].set_title("Per-seed AUC")
    raw_axes[0].set_ylabel("AUC")
    raw_axes[1].set_title("Per-seed PR-AUC")
    raw_axes[1].set_ylabel("PR-AUC")
    raw_path = _save_figure(raw_figure, output_root / "truncation_by_seed.png")
    return [summary_path, raw_path]


def plot_dataset_profile(input_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    payload = _load_json(input_path)
    output_root = _resolve_output_dir(input_path, output_dir)

    sequence_profile = payload["sequence_profile"]
    time_windows = payload["temporal_drift"]["time_windows"]
    item_buckets = payload["hot_cold_profile"]["item_buckets"]
    per_group = sequence_profile["per_group"]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    event_stats = ["mean", "p50", "p95"]
    total_values = [float(sequence_profile["total_event_count"][stat]) for stat in event_stats]
    selected_values = [float(sequence_profile["selected_event_count"][stat]) for stat in event_stats]
    x_positions = np.arange(len(event_stats))
    axes[0, 0].bar(x_positions - 0.18, total_values, width=0.36, color="#0F766E", label="Total events")
    axes[0, 0].bar(x_positions + 0.18, selected_values, width=0.36, color="#F59E0B", label="Selected events")
    axes[0, 0].set_xticks(x_positions, [stat.upper() for stat in event_stats])
    axes[0, 0].set_title(f"Truncation Summary (rate={sequence_profile['truncation_rate']:.3f})")
    axes[0, 0].set_ylabel("Events")
    axes[0, 0].legend(loc="upper left")

    group_names = list(per_group.keys())
    group_means = [float(per_group[group_name]["length"]["mean"]) for group_name in group_names]
    axes[0, 1].bar(group_names, group_means, color=["#2563EB", "#7C3AED", "#BE123C"][: len(group_names)])
    axes[0, 1].set_title("Mean Sequence Length by Group")
    axes[0, 1].set_ylabel("Mean length")

    window_labels = [bucket["bucket"] for bucket in time_windows]
    positive_rates = [float(bucket["positive_rate"]) for bucket in time_windows]
    mean_event_counts = [float(bucket["mean_total_event_count"]) for bucket in time_windows]
    axes[1, 0].plot(window_labels, positive_rates, marker="o", color="#BE123C", label="Positive rate")
    time_axis = axes[1, 0].twinx()
    time_axis.plot(window_labels, mean_event_counts, marker="s", color="#0F766E", label="Mean total events")
    axes[1, 0].set_title("Temporal Drift by Time Window")
    axes[1, 0].set_ylabel("Positive rate")
    time_axis.set_ylabel("Mean total events")
    lines, labels = axes[1, 0].get_legend_handles_labels()
    lines2, labels2 = time_axis.get_legend_handles_labels()
    axes[1, 0].legend(lines + lines2, labels + labels2, loc="upper left")

    bucket_labels = [bucket["bucket"] for bucket in item_buckets]
    bucket_row_rate = [float(bucket["row_rate"]) for bucket in item_buckets]
    bucket_positive_rate = [float(bucket["positive_rate"]) for bucket in item_buckets]
    axes[1, 1].bar(bucket_labels, bucket_row_rate, color="#2563EB", label="Row rate")
    bucket_axis = axes[1, 1].twinx()
    bucket_axis.plot(bucket_labels, bucket_positive_rate, marker="o", color="#F59E0B", label="Positive rate")
    axes[1, 1].set_title("Item Frequency Buckets")
    axes[1, 1].set_ylabel("Row rate")
    bucket_axis.set_ylabel("Positive rate")
    lines, labels = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = bucket_axis.get_legend_handles_labels()
    axes[1, 1].legend(lines + lines2, labels + labels2, loc="upper right")

    for axis in axes.flat:
        axis.grid(alpha=0.25)

    overview_path = _save_figure(figure, output_root / "dataset_profile_overview.png")

    hour_counts = payload["temporal_drift"]["hour_of_day_counts"]
    hour_labels = sorted(int(hour) for hour in hour_counts.keys())
    hourly_figure, hourly_axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    hourly_axis.bar([str(hour) for hour in hour_labels], [int(hour_counts[str(hour)]) for hour in hour_labels], color="#0F766E")
    hourly_axis.set_title("Hour-of-day Activity")
    hourly_axis.set_xlabel("Hour")
    hourly_axis.set_ylabel("Rows")
    hourly_axis.grid(axis="y", alpha=0.25)
    hourly_path = _save_figure(hourly_figure, output_root / "dataset_hourly_activity.png")
    return [overview_path, hourly_path]


def plot_evaluation(input_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    payload = _load_json(input_path)
    metrics = payload.get("metrics", {})
    bucket_metrics = metrics.get("bucket_metrics", {})
    if not metrics:
        raise ValueError("evaluation.json 中没有 metrics，无法绘图。")

    output_root = _resolve_output_dir(input_path, output_dir)
    global_figure, global_axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    score_labels = ["AUC", "PR-AUC", "GAUC", "Positive rate"]
    score_values = [
        float(metrics.get("auc", 0.0)),
        float(metrics.get("pr_auc", 0.0)),
        float(metrics.get("gauc", {}).get("value", 0.0)),
        float(metrics.get("positive_rate", 0.0)),
    ]
    global_axes[0].bar(score_labels, score_values, color=["#0F766E", "#BE123C", "#2563EB", "#F59E0B"])
    global_axes[0].set_title(f"Global Metrics: {payload.get('model_name', 'model')}")
    global_axes[0].set_ylim(0.0, max(score_values) * 1.2 if score_values else 1.0)

    calib_labels = ["Brier", "Logloss", "Mean pred", "Pred std"]
    calib_values = [
        float(metrics.get("brier", 0.0)),
        float(metrics.get("logloss", 0.0)),
        float(metrics.get("prediction_mean", 0.0)),
        float(metrics.get("prediction_std", 0.0)),
    ]
    global_axes[1].bar(calib_labels, calib_values, color=["#F59E0B", "#7C3AED", "#2563EB", "#475569"])
    global_axes[1].set_title("Calibration / Prediction Shape")
    for axis in global_axes:
        axis.grid(axis="y", alpha=0.25)

    paths = [_save_figure(global_figure, output_root / "evaluation_overview.png")]

    for bucket_name, bucket_rows in bucket_metrics.items():
        if not bucket_rows:
            continue
        labels = [str(row.get("bucket", "unknown")) for row in bucket_rows]
        positions = np.arange(len(labels))
        auc_values = [float(row.get("auc", 0.0)) for row in bucket_rows]
        pr_auc_values = [float(row.get("pr_auc", 0.0)) for row in bucket_rows]
        counts = [int(row.get("count", 0)) for row in bucket_rows]
        positive_rates = [float(row.get("positive_rate", 0.0)) for row in bucket_rows]

        figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
        axes[0].plot(labels, auc_values, marker="o", color="#0F766E", label="AUC")
        axes[0].plot(labels, pr_auc_values, marker="o", color="#BE123C", label="PR-AUC")
        axes[0].set_title(f"Bucket Metrics: {bucket_name}")
        axes[0].set_ylabel("Score")
        axes[0].legend(loc="best")

        axes[1].bar(labels, counts, color="#2563EB", label="Count")
        count_axis = axes[1].twinx()
        count_axis.plot(labels, positive_rates, marker="o", color="#F59E0B", label="Positive rate")
        axes[1].set_ylabel("Count")
        count_axis.set_ylabel("Positive rate")
        axes[1].set_xlabel("Bucket")
        lines, line_labels = axes[1].get_legend_handles_labels()
        lines2, line_labels2 = count_axis.get_legend_handles_labels()
        axes[1].legend(lines + lines2, line_labels + line_labels2, loc="upper right")
        for axis in axes:
            axis.grid(alpha=0.25)

        paths.append(_save_figure(figure, output_root / f"evaluation_bucket_{_safe_name(bucket_name)}.png"))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于现有 JSON 产物生成 matplotlib 可视化图表。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluation_parser = subparsers.add_parser("evaluation", help="从 evaluation.json 生成全局指标与分桶图。")
    evaluation_parser.add_argument("--input", type=str, default="outputs/creatorwyx_din_adapter/evaluation.json", help="evaluation.json 路径。")
    evaluation_parser.add_argument("--output-dir", type=str, default="", help="图片输出目录，默认写到输入文件同级 plots/。")

    batch_parser = subparsers.add_parser("batch-report", help="从 experiment_report.json 生成总览图。")
    batch_parser.add_argument("--input", type=str, default="outputs/reports/current_experiments/experiment_report.json", help="experiment_report.json 路径。")
    batch_parser.add_argument("--output-dir", type=str, default="", help="图片输出目录，默认写到输入文件同级 plots/。")

    summary_parser = subparsers.add_parser("summary", help="从 summary.json 生成训练过程曲线。")
    summary_parser.add_argument("--input", type=str, default="outputs/grok_din_readout/summary.json", help="summary.json 路径。")
    summary_parser.add_argument("--output-dir", type=str, default="", help="图片输出目录，默认写到输入文件同级 plots/。")

    truncation_parser = subparsers.add_parser("truncation-sweep", help="从 truncation sweep report.json 生成均值/方差与 seed 散点图。")
    truncation_parser.add_argument("--input", type=str, default="outputs/truncation_sweep/grok_din_readout/report.json", help="truncation sweep report.json 路径。")
    truncation_parser.add_argument("--output-dir", type=str, default="", help="图片输出目录，默认写到输入文件同级 plots/。")

    dataset_parser = subparsers.add_parser("dataset-profile", help="从 dataset_profile.json 生成数据画像图。")
    dataset_parser.add_argument("--input", type=str, default="outputs/feature_engineering/dataset_profile.json", help="dataset_profile.json 路径。")
    dataset_parser.add_argument("--output-dir", type=str, default="", help="图片输出目录，默认写到输入文件同级 plots/。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "evaluation":
        paths = plot_evaluation(args.input, args.output_dir or None)
    elif args.command == "batch-report":
        paths = plot_batch_report(args.input, args.output_dir or None)
    elif args.command == "summary":
        paths = plot_summary(args.input, args.output_dir or None)
    elif args.command == "truncation-sweep":
        paths = plot_truncation_sweep(args.input, args.output_dir or None)
    elif args.command == "dataset-profile":
        paths = plot_dataset_profile(args.input, args.output_dir or None)
    else:
        raise ValueError(f"未知命令: {args.command}")

    for path in paths:
        print(f"plot_written_to={path}")


if __name__ == "__main__":
    main()
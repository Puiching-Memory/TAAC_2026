from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

from .config import load_config
from .train import run_training
from .utils import ensure_dir, write_json


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _ci95(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "lower": 0.0, "upper": 0.0}
    mean_value = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean_value, "std": 0.0, "lower": mean_value, "upper": mean_value}
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    std_value = math.sqrt(variance)
    margin = 1.96 * std_value / math.sqrt(len(values))
    return {
        "mean": mean_value,
        "std": std_value,
        "lower": mean_value - margin,
        "upper": mean_value + margin,
    }


def _build_markdown(summary_records: list[dict[str, Any]]) -> str:
    lines = [
        "# 截断策略消融",
        "",
        "| 实验 | max_seq_len | AUC(mean±std) | PR-AUC(mean±std) | GAUC(mean±std) | 平均时延(mean) | P95 时延(mean) |",
        "| ---- | ----------: | ------------: | ---------------: | -------------: | -------------: | -------------: |",
    ]
    for record in summary_records:
        lines.append(
            "| {experiment_id} | {max_seq_len} | {auc_mean}±{auc_std} | {pr_auc_mean}±{pr_auc_std} | {gauc_mean}±{gauc_std} | {mean_latency} | {p95_latency} |".format(
                experiment_id=record["experiment_id"],
                max_seq_len=record["max_seq_len"],
                auc_mean=_format_metric(record["auc_ci95"]["mean"]),
                auc_std=_format_metric(record["auc_ci95"]["std"]),
                pr_auc_mean=_format_metric(record["pr_auc_ci95"]["mean"]),
                pr_auc_std=_format_metric(record["pr_auc_ci95"]["std"]),
                gauc_mean=_format_metric(record["gauc_ci95"]["mean"]),
                gauc_std=_format_metric(record["gauc_ci95"]["std"]),
                mean_latency=_format_metric(record["mean_latency_ci95"]["mean"]),
                p95_latency=_format_metric(record["p95_latency_ci95"]["mean"]),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对指定配置做 history truncation policy 多 seed 消融。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grok_din_readout.yaml",
        help="基础配置路径。",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[128, 256, 384],
        help="要比较的 max_seq_len 列表。",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="每个 max_seq_len 下的随机种子列表。",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/truncation_sweep/grok_din_readout",
        help="消融实验输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(Path(args.config))
    output_root = ensure_dir(args.output_root)

    raw_runs: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

    for experiment_offset, seq_len in enumerate(args.seq_lens, start=13):
        seq_len_runs: list[dict[str, Any]] = []
        for seed in args.seeds:
            config = copy.deepcopy(base_config)
            config.data.max_seq_len = int(seq_len)
            config.train.seed = int(seed)
            config.train.output_dir = str(output_root / f"len_{seq_len}" / f"seed_{seed}")
            summary = run_training(config)
            run_record = {
                "experiment_id": f"E{experiment_offset:03d}",
                "config_path": args.config,
                "max_seq_len": int(seq_len),
                "seed": int(seed),
                "output_dir": config.train.output_dir,
                "best_val_auc": summary["best_val_auc"],
                "best_val_pr_auc": summary["best_val_pr_auc"],
                "best_val_gauc": summary["best_val_gauc"],
                "best_val_brier": summary["best_val_brier"],
                "best_val_logloss": summary["best_val_logloss"],
                "mean_latency_ms_per_sample": summary["latency"]["mean_ms_per_sample"],
                "p95_latency_ms_per_sample": summary["latency"]["p95_ms_per_sample"],
            }
            raw_runs.append(run_record)
            seq_len_runs.append(run_record)

        summary_record = {
            "experiment_id": f"E{experiment_offset:03d}",
            "config_path": args.config,
            "max_seq_len": int(seq_len),
            "seeds": [int(seed) for seed in args.seeds],
            "auc_ci95": _ci95([record["best_val_auc"] for record in seq_len_runs]),
            "pr_auc_ci95": _ci95([record["best_val_pr_auc"] for record in seq_len_runs]),
            "gauc_ci95": _ci95([record["best_val_gauc"] for record in seq_len_runs]),
            "brier_ci95": _ci95([record["best_val_brier"] for record in seq_len_runs]),
            "logloss_ci95": _ci95([record["best_val_logloss"] for record in seq_len_runs]),
            "mean_latency_ci95": _ci95([record["mean_latency_ms_per_sample"] for record in seq_len_runs]),
            "p95_latency_ci95": _ci95([record["p95_latency_ms_per_sample"] for record in seq_len_runs]),
        }
        summary_records.append(summary_record)
        print(
            f"{summary_record['experiment_id']} max_seq_len={seq_len} auc_mean={summary_record['auc_ci95']['mean']:.5f} pr_auc_mean={summary_record['pr_auc_ci95']['mean']:.5f} gauc_mean={summary_record['gauc_ci95']['mean']:.5f}"
        )

    payload = {
        "base_config": args.config,
        "runs": raw_runs,
        "summary": summary_records,
    }
    write_json(output_root / "report.json", payload)
    (output_root / "report.md").write_text(_build_markdown(summary_records), encoding="utf-8")
    print(f"report_written_to={output_root / 'report.json'}")
    print(f"markdown_written_to={output_root / 'report.md'}")


if __name__ == "__main__":
    main()
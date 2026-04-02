from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .evaluate import evaluate_checkpoint
from .utils import ensure_dir, write_json


DEFAULT_CONFIG_PATHS = [
    "configs/baseline.yaml",
    "configs/creatorwyx_din_adapter.yaml",
    "configs/creatorwyx_grouped_din_adapter.yaml",
    "configs/tencent_sasrec_adapter.yaml",
    "configs/zcyeee_retrieval_adapter.yaml",
    "configs/oo_retrieval_adapter.yaml",
    "configs/omnigenrec_adapter.yaml",
    "configs/deep_context_net.yaml",
    "configs/unirec.yaml",
    "configs/uniscaleformer.yaml",
    "configs/grok_din_readout.yaml",
    "configs/unirec_din_readout.yaml",
]


def _experiment_id(index: int) -> str:
    return f"E{index:03d}"


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _build_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# 实验多指标回填汇总",
        "",
        "| 编号 | 配置 | 模型 | AUC | PR-AUC | GAUC | Brier | Logloss | 平均时延(ms/样本) | P95 时延(ms/样本) |",
        "| ---- | ---- | ---- | ---: | -----: | ---: | ----: | ------: | ----------------: | ---------------: |",
    ]
    for record in records:
        lines.append(
            "| {experiment_id} | {config_path} | {model_name} | {auc} | {pr_auc} | {gauc} | {brier} | {logloss} | {mean_latency} | {p95_latency} |".format(
                experiment_id=record["experiment_id"],
                config_path=record["config_path"],
                model_name=record["model_name"],
                auc=_format_metric(record["auc"]),
                pr_auc=_format_metric(record["pr_auc"]),
                gauc=_format_metric(record["gauc"]),
                brier=_format_metric(record["brier"]),
                logloss=_format_metric(record["logloss"]),
                mean_latency=_format_metric(record["mean_latency_ms_per_sample"]),
                p95_latency=_format_metric(record["p95_latency_ms_per_sample"]),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量回填当前实验的多指标评估，并生成总表。")
    parser.add_argument(
        "--config-paths",
        nargs="*",
        default=DEFAULT_CONFIG_PATHS,
        help="需要评估的配置路径列表。默认覆盖当前 12 个活跃实验。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reports/current_experiments",
        help="汇总报告输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    records: list[dict[str, Any]] = []

    for index, config_path in enumerate(args.config_paths, start=1):
        payload = evaluate_checkpoint(config_path=config_path)
        metrics = payload["metrics"]
        latency = payload["latency"]
        record = {
            "experiment_id": _experiment_id(index),
            "config_path": payload["config_path"],
            "checkpoint_path": payload["checkpoint_path"],
            "model_name": payload["model_name"],
            "auc": metrics["auc"],
            "pr_auc": metrics["pr_auc"],
            "gauc": metrics["gauc"]["value"],
            "brier": metrics["brier"],
            "logloss": metrics["logloss"],
            "mean_latency_ms_per_sample": latency["mean_ms_per_sample"],
            "p95_latency_ms_per_sample": latency["p95_ms_per_sample"],
            "auc_ci95": metrics["confidence_intervals"]["auc"],
            "pr_auc_ci95": metrics["confidence_intervals"]["pr_auc"],
        }
        records.append(record)
        print(
            f"{record['experiment_id']} model={record['model_name']} auc={record['auc']:.5f} pr_auc={record['pr_auc']:.5f} gauc={record['gauc']:.5f}"
        )

    report_payload = {
        "records": records,
    }
    write_json(output_dir / "experiment_report.json", report_payload)
    (output_dir / "experiment_report.md").write_text(_build_markdown(records), encoding="utf-8")
    print(f"report_written_to={output_dir / 'experiment_report.json'}")
    print(f"markdown_written_to={output_dir / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
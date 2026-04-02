from __future__ import annotations

import argparse
from pathlib import Path

from ..utils import ensure_dir, write_json
from .dataset_analysis import build_dataset_profile_artifacts, print_dataset_profile_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="输出 TAAC 2026 数据集的时序漂移、序列长度与冷热分布分析。")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/datasets--TAAC2026--data_sample_1000/snapshots/2f0ddba721a8323495e73d5229c836df5d603b39/sample_data.parquet",
        help="Parquet 数据集路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/feature_engineering",
        help="导出目录，默认写出 dataset_profile.json。",
    )
    parser.add_argument(
        "--label-action-type",
        type=int,
        default=2,
        help="正样本 action_type，默认与训练配置保持一致。",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="用于统计截断率的 max_seq_len。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = build_dataset_profile_artifacts(
        dataset_path=args.dataset_path,
        label_action_type=args.label_action_type,
        max_seq_len=args.max_seq_len,
    )
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    output_path = output_dir / "dataset_profile.json"
    write_json(output_path, profile)

    print_dataset_profile_summary(profile)
    print(f"dataset_profile_written_to={output_path}")


if __name__ == "__main__":
    main()
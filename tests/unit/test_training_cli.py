from __future__ import annotations

from pathlib import Path

import pytest

from taac2026.infrastructure.pcvr.config import (
    PCVRDataCacheConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRSequenceCropConfig,
    PCVRTrainConfig,
)
from taac2026.infrastructure.pcvr.training import parse_pcvr_train_args
from taac2026.application.training.cli import parse_train_args


def test_parse_train_args_forwards_experiment_specific_options() -> None:
    args, extra = parse_train_args(
        [
            "--experiment",
            "config/baseline",
            "--dataset-path",
            "/data/train",
            "--schema-path",
            "/data/schema.json",
            "--batch_size",
            "8",
        ]
    )

    assert args.experiment == "config/baseline"
    assert args.dataset_path == "/data/train"
    assert args.schema_path == "/data/schema.json"
    assert extra == ["--batch_size", "8"]


def test_parse_pcvr_train_args_accepts_runtime_flags(tmp_path: Path) -> None:
    args = parse_pcvr_train_args(
        ["--amp", "--amp-dtype", "float16", "--compile"],
        package_dir=tmp_path,
    )

    assert args.amp is True
    assert args.amp_dtype == "float16"
    assert args.compile is True


def test_parse_pcvr_train_args_accepts_symbiosis_ablation_flags(tmp_path: Path) -> None:
    args = parse_pcvr_train_args(
        [
            "--no-symbiosis-use-user-item-graph",
            "--no-symbiosis-use-fourier-time",
            "--no-symbiosis-use-context-exchange",
            "--no-symbiosis-use-multi-scale",
            "--symbiosis-use-domain-gate",
        ],
        package_dir=tmp_path,
    )

    assert args.symbiosis_use_user_item_graph is False
    assert args.symbiosis_use_fourier_time is False
    assert args.symbiosis_use_context_exchange is False
    assert args.symbiosis_use_multi_scale is False
    assert args.symbiosis_use_domain_gate is True


def test_parse_pcvr_train_args_rejects_data_pipeline_flags(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_pcvr_train_args(
            ["--data-pipeline-transforms", "sequence_crop,feature_mask"],
            package_dir=tmp_path,
        )


def test_parse_pcvr_train_args_uses_typed_data_pipeline_defaults(
    tmp_path: Path,
) -> None:
    defaults = PCVRTrainConfig(
        data_pipeline=PCVRDataPipelineConfig(
            cache=PCVRDataCacheConfig(mode="memory", max_batches=32),
            seed=77,
            transforms=(
                PCVRSequenceCropConfig(
                    views_per_row=2,
                    seq_window_mode="random_tail",
                    seq_window_min_len=8,
                ),
                PCVRFeatureMaskConfig(probability=0.05),
            ),
        )
    )

    args = parse_pcvr_train_args([], package_dir=tmp_path, defaults=defaults)

    assert not hasattr(args, "data_pipeline_transforms")
    assert not hasattr(args, "augmentation_mode")


def test_pcvr_train_config_serializes_structured_data_pipeline() -> None:
    defaults = PCVRTrainConfig(
        data_pipeline=PCVRDataPipelineConfig(
            cache=PCVRDataCacheConfig(mode="memory", max_batches=32),
            seed=77,
            strict_time_filter=False,
            transforms=(
                PCVRSequenceCropConfig(
                    views_per_row=2,
                    seq_window_mode="random_tail",
                    seq_window_min_len=8,
                ),
                PCVRFeatureMaskConfig(probability=0.05),
                PCVRDomainDropoutConfig(probability=0.1),
            ),
        )
    )

    flat_config = defaults.to_flat_dict()

    assert "data_pipeline_transforms" not in flat_config
    assert "augmentation_mode" not in flat_config
    assert flat_config["data_pipeline"] == {
        "cache": {"mode": "memory", "max_batches": 32},
        "seed": 77,
        "strict_time_filter": False,
        "transforms": [
            {
                "name": "sequence_crop",
                "enabled": True,
                "views_per_row": 2,
                "seq_window_mode": "random_tail",
                "seq_window_min_len": 8,
            },
            {"name": "feature_mask", "enabled": True, "probability": 0.05},
            {"name": "domain_dropout", "enabled": True, "probability": 0.1},
        ],
    }

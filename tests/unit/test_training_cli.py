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
from taac2026.application.training.cli import main, parse_train_args
from taac2026.infrastructure.io.json_utils import loads
from taac2026.infrastructure.pcvr.training import parse_pcvr_train_args


def _write_minimal_experiment(package_dir: Path, *, requires_dataset: bool) -> Path:
    package_dir.mkdir(parents=True)
    metadata = "{'requires_dataset': False}" if not requires_dataset else "{}"
    (package_dir / "__init__.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "from taac2026.domain.experiment import ExperimentSpec\n"
        "\n"
        "\n"
        "def _train(request):\n"
        "    return {\"dataset_path\": None if request.dataset_path is None else str(request.dataset_path), \"run_dir\": str(request.run_dir)}\n"
        "\n"
        "\n"
        "EXPERIMENT = ExperimentSpec(\n"
        "    name=\"minimal_experiment\",\n"
        "    package_dir=Path(__file__).resolve().parent,\n"
        "    train_fn=_train,\n"
        f"    metadata={metadata},\n"
        ")\n",
        encoding="utf-8",
    )
    return package_dir


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


def test_parse_train_args_allows_missing_dataset_path() -> None:
    args, extra = parse_train_args(["--experiment", "config/host_device_info"])

    assert args.experiment == "config/host_device_info"
    assert args.dataset_path is None
    assert extra == []


def test_training_main_allows_experiment_without_dataset_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_dir = _write_minimal_experiment(tmp_path / "config" / "maintenance_exp", requires_dataset=False)

    exit_code = main([
        "--experiment",
        str(experiment_dir),
        "--run-dir",
        str(tmp_path / "outputs"),
    ])

    captured = capsys.readouterr()
    payload = loads(captured.out)
    assert exit_code == 0
    assert "\n" not in captured.out.strip()
    assert payload["dataset_path"] is None
    assert payload["run_dir"] == str(tmp_path / "outputs")


def test_training_main_rejects_missing_dataset_for_dataset_experiment(tmp_path: Path) -> None:
    experiment_dir = _write_minimal_experiment(tmp_path / "config" / "dataset_exp", requires_dataset=True)

    with pytest.raises(ValueError, match="requires --dataset-path"):
        main(["--experiment", str(experiment_dir), "--run-dir", str(tmp_path / "outputs")])


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
            "--no-symbiosis-use-candidate-decoder",
            "--no-symbiosis-use-action-conditioning",
            "--no-symbiosis-use-compressed-memory",
            "--no-symbiosis-use-attention-sink",
            "--no-symbiosis-use-lane-mixing",
            "--no-symbiosis-use-semantic-id",
            "--symbiosis-memory-block-size",
            "32",
            "--symbiosis-memory-top-k",
            "4",
            "--symbiosis-recent-tokens",
            "16",
        ],
        package_dir=tmp_path,
    )

    assert args.symbiosis_use_user_item_graph is False
    assert args.symbiosis_use_fourier_time is False
    assert args.symbiosis_use_context_exchange is False
    assert args.symbiosis_use_multi_scale is False
    assert args.symbiosis_use_domain_gate is True
    assert args.symbiosis_use_candidate_decoder is False
    assert args.symbiosis_use_action_conditioning is False
    assert args.symbiosis_use_compressed_memory is False
    assert args.symbiosis_use_attention_sink is False
    assert args.symbiosis_use_lane_mixing is False
    assert args.symbiosis_use_semantic_id is False
    assert args.symbiosis_memory_block_size == 32
    assert args.symbiosis_memory_top_k == 4
    assert args.symbiosis_recent_tokens == 16


def test_parse_pcvr_train_args_accepts_auc_and_optimizer_flags(tmp_path: Path) -> None:
    args = parse_pcvr_train_args(
        [
            "--pairwise-auc-weight",
            "0.2",
            "--pairwise-auc-temperature",
            "0.5",
            "--dense-optimizer-type",
            "orthogonal_adamw",
        ],
        package_dir=tmp_path,
    )

    assert args.pairwise_auc_weight == pytest.approx(0.2)
    assert args.pairwise_auc_temperature == pytest.approx(0.5)
    assert args.dense_optimizer_type == "orthogonal_adamw"


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

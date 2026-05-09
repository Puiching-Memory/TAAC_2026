from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from taac2026.domain.config import (
    PCVRDataCacheConfig,
    PCVRDataConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRLossConfig,
    PCVRLossTermConfig,
    PCVRModelConfig,
    PCVROptimizerConfig,
    PCVRSequenceCropConfig,
    PCVRTrainConfig,
)
from taac2026.application.training.cli import main, parse_train_args
from taac2026.infrastructure.experiments.module_loader import load_module_from_path
from taac2026.infrastructure.io.json import loads
from taac2026.application.training.workflow import PCVRTrainDataBundle, build_pcvr_train_hooks
from taac2026.application.training.args import parse_pcvr_train_args, train_pcvr_model
from taac2026.infrastructure.runtime.execution import RuntimeExecutionConfig


REPO_ROOT = Path(__file__).resolve().parents[4]


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
            "experiments/baseline",
            "--dataset-path",
            "/data/train",
            "--schema-path",
            "/data/schema.json",
            "--batch_size",
            "8",
        ]
    )

    assert args.experiment == "experiments/baseline"
    assert args.dataset_path == "/data/train"
    assert args.schema_path == "/data/schema.json"
    assert extra == ["--batch_size", "8"]


def test_parse_train_args_allows_missing_dataset_path() -> None:
    args, extra = parse_train_args(["--experiment", "experiments/host_device_info"])

    assert args.experiment == "experiments/host_device_info"
    assert args.dataset_path is None
    assert extra == []


def test_parse_train_args_requires_experiment() -> None:
    with pytest.raises(SystemExit):
        parse_train_args(["--dataset-path", "/data/train"])


def test_training_main_allows_experiment_without_dataset_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_dir = _write_minimal_experiment(tmp_path / "experiments" / "maintenance" / "maintenance_exp", requires_dataset=False)

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
    experiment_dir = _write_minimal_experiment(tmp_path / "experiments" / "maintenance" / "dataset_exp", requires_dataset=True)

    with pytest.raises(ValueError, match="requires --dataset-path"):
        main(["--experiment", str(experiment_dir), "--run-dir", str(tmp_path / "outputs")])


def test_training_main_allows_missing_dataset_for_pcvr_kind_experiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_dir = tmp_path / "experiments" / "pcvr" / "pcvr_exp"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "__init__.py").write_text(
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
        "    name=\"pcvr_exp\",\n"
        "    package_dir=Path(__file__).resolve().parent,\n"
        "    train_fn=_train,\n"
        "    metadata={\"requires_dataset\": True, \"kind\": \"pcvr\"},\n"
        ")\n",
        encoding="utf-8",
    )

    exit_code = main([
        "--experiment",
        str(experiment_dir),
        "--run-dir",
        str(tmp_path / "outputs"),
    ])

    captured = capsys.readouterr()
    payload = loads(captured.out)
    assert exit_code == 0
    assert payload["dataset_path"] is None


def test_training_main_rejects_explicit_dataset_for_local_pcvr_kind_experiment(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiments" / "pcvr" / "pcvr_exp"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "__init__.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "from taac2026.domain.experiment import ExperimentSpec\n"
        "\n"
        "\n"
        "def _train(request):\n"
        "    return {}\n"
        "\n"
        "\n"
        "EXPERIMENT = ExperimentSpec(\n"
        "    name=\"pcvr_exp\",\n"
        "    package_dir=Path(__file__).resolve().parent,\n"
        "    train_fn=_train,\n"
        "    metadata={\"requires_dataset\": True, \"kind\": \"pcvr\"},\n"
        ")\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="local PCVR runs no longer accept --dataset-path"):
        main([
            "--experiment",
            str(experiment_dir),
            "--dataset-path",
            "/tmp/custom.parquet",
        ])


def test_training_main_allows_explicit_dataset_for_bundle_pcvr_kind_experiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiments" / "pcvr" / "pcvr_exp"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "__init__.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "from taac2026.domain.experiment import ExperimentSpec\n"
        "\n"
        "\n"
        "def _train(request):\n"
        "    return {\"dataset_path\": None if request.dataset_path is None else str(request.dataset_path)}\n"
        "\n"
        "\n"
        "EXPERIMENT = ExperimentSpec(\n"
        "    name=\"pcvr_exp\",\n"
        "    package_dir=Path(__file__).resolve().parent,\n"
        "    train_fn=_train,\n"
        "    metadata={\"requires_dataset\": True, \"kind\": \"pcvr\"},\n"
        ")\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TAAC_BUNDLE_MODE", "1")

    exit_code = main([
        "--experiment",
        str(experiment_dir),
        "--dataset-path",
        "/tmp/custom.parquet",
    ])

    captured = capsys.readouterr()
    payload = loads(captured.out)
    assert exit_code == 0
    assert payload["dataset_path"] == "/tmp/custom.parquet"


def test_parse_pcvr_train_args_accepts_runtime_flags(tmp_path: Path) -> None:
    args = parse_pcvr_train_args(
        [
            "--amp",
            "--amp-dtype",
            "float16",
            "--compile",
            "--progress-log-interval-steps",
            "25",
            "--gradient-checkpointing",
        ],
        package_dir=tmp_path,
        defaults=PCVRTrainConfig(),
    )

    assert args.amp is True
    assert args.amp_dtype == "float16"
    assert args.compile is True
    assert args.progress_log_interval_steps == 25
    assert args.gradient_checkpointing is True


def test_parse_pcvr_train_args_uses_runtime_progress_log_interval_default(tmp_path: Path) -> None:
    defaults = PCVRTrainConfig(runtime=RuntimeExecutionConfig(progress_log_interval_steps=77))

    args = parse_pcvr_train_args([], package_dir=tmp_path, defaults=defaults)

    assert args.progress_log_interval_steps == 77


def test_parse_pcvr_train_args_accepts_rms_norm_flags(tmp_path: Path) -> None:
    args = parse_pcvr_train_args(
        ["--rms-norm-backend", "triton", "--rms-norm-block-rows", "8"],
        package_dir=tmp_path,
        defaults=PCVRTrainConfig(),
    )

    assert args.rms_norm_backend == "triton"
    assert args.rms_norm_block_rows == 8


def test_parse_pcvr_train_args_accepts_flash_attention_backend_flag(tmp_path: Path) -> None:
    args = parse_pcvr_train_args(
        ["--flash-attention-backend", "tilelang"],
        package_dir=tmp_path,
        defaults=PCVRTrainConfig(),
    )

    assert args.flash_attention_backend == "tilelang"


def test_parse_pcvr_train_args_can_disable_default_rope(tmp_path: Path) -> None:
    defaults = PCVRTrainConfig(model=PCVRModelConfig(use_rope=True))

    args = parse_pcvr_train_args(
        ["--no-use-rope"],
        package_dir=tmp_path,
        defaults=defaults,
    )

    assert args.use_rope is False


def test_parse_pcvr_train_args_accepts_timestamp_split_flags(tmp_path: Path) -> None:
    args = parse_pcvr_train_args(
        [
            "--split-strategy",
            "timestamp_range",
            "--train-timestamp-end",
            "1772582400",
            "--valid-timestamp-start",
            "1772582400",
            "--valid-timestamp-end",
            "1772600400",
        ],
        package_dir=tmp_path,
        defaults=PCVRTrainConfig(),
    )

    assert args.split_strategy == "timestamp_range"
    assert args.train_timestamp_start == 0
    assert args.train_timestamp_end == 1_772_582_400
    assert args.valid_timestamp_start == 1_772_582_400
    assert args.valid_timestamp_end == 1_772_600_400


def test_parse_pcvr_train_args_accepts_sampling_strategy(tmp_path: Path) -> None:
    args = parse_pcvr_train_args(
        ["--sampling-strategy", "row_group_sweep"],
        package_dir=tmp_path,
        defaults=PCVRTrainConfig(),
    )

    assert args.sampling_strategy == "row_group_sweep"


@pytest.mark.parametrize("flag", ["--patience", "--steps-per-epoch"])
def test_parse_pcvr_train_args_rejects_legacy_epoch_flags(tmp_path: Path, flag: str) -> None:
    with pytest.raises(SystemExit):
        parse_pcvr_train_args(
            [flag, "1"],
            package_dir=tmp_path,
            defaults=PCVRTrainConfig(),
        )


def test_parse_pcvr_train_args_uses_timestamp_split_defaults(tmp_path: Path) -> None:
    defaults = PCVRTrainConfig(
        data=PCVRDataConfig(
            split_strategy="timestamp_range",
            train_timestamp_end=10,
            valid_timestamp_start=10,
            valid_timestamp_end=20,
        )
    )

    args = parse_pcvr_train_args([], package_dir=tmp_path, defaults=defaults)

    assert args.split_strategy == "timestamp_range"
    assert args.train_timestamp_end == 10
    assert args.valid_timestamp_start == 10
    assert args.valid_timestamp_end == 20


def test_parse_pcvr_train_args_honors_platform_path_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIN_DATA_PATH", "/env/data")
    monkeypatch.setenv("TAAC_SCHEMA_PATH", "/env/schema.json")
    monkeypatch.setenv("TRAIN_CKPT_PATH", "/env/output")

    args = parse_pcvr_train_args(
        ["--data_dir", "/cli/data", "--schema_path", "/cli/schema.json", "--ckpt_dir", "/cli/output"],
        package_dir=tmp_path,
        defaults=PCVRTrainConfig(),
    )

    assert args.data_dir == "/env/data"
    assert args.schema_path == "/env/schema.json"
    assert args.ckpt_dir == "/env/output"


def test_parse_pcvr_train_args_rejects_symbiosis_ablation_flags(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_pcvr_train_args(
            [
                "--no-symbiosis-use-field-tokens",
                "--no-symbiosis-use-dense-packets",
                "--no-symbiosis-use-sequence-memory",
                "--no-symbiosis-use-compressed-memory",
                "--no-symbiosis-use-candidate-token",
                "--no-symbiosis-use-item-prior",
                "--no-symbiosis-use-domain-type",
                "--symbiosis-memory-block-size",
                "32",
                "--symbiosis-memory-top-k",
                "4",
                "--symbiosis-recent-tokens",
                "16",
            ],
            package_dir=tmp_path,
            defaults=PCVRTrainConfig(),
        )


def test_symbiosis_package_parser_accepts_symbiosis_ablation_flags() -> None:
    symbiosis_module = load_module_from_path(REPO_ROOT / "experiments" / "symbiosis")

    args = symbiosis_module.parse_symbiosis_train_args(
        [
            "--no-symbiosis-use-field-tokens",
            "--no-symbiosis-use-dense-packets",
            "--no-symbiosis-use-sequence-memory",
            "--no-symbiosis-use-compressed-memory",
            "--no-symbiosis-use-candidate-token",
            "--no-symbiosis-use-item-prior",
            "--no-symbiosis-use-domain-type",
            "--symbiosis-memory-block-size",
            "64",
            "--symbiosis-memory-top-k",
            "4",
            "--symbiosis-recent-tokens",
            "16",
        ],
        package_dir=symbiosis_module.EXPERIMENT.package_dir,
        defaults=symbiosis_module.TRAIN_DEFAULTS,
    )

    assert args.symbiosis_use_field_tokens is False
    assert args.symbiosis_use_dense_packets is False
    assert args.symbiosis_use_sequence_memory is False
    assert args.symbiosis_use_compressed_memory is False
    assert args.symbiosis_use_candidate_token is False
    assert args.symbiosis_use_item_prior is False
    assert args.symbiosis_use_domain_type is False
    assert args.symbiosis_memory_block_size == 64
    assert args.symbiosis_memory_top_k == 4
    assert args.symbiosis_recent_tokens == 16


@pytest.mark.parametrize("dense_optimizer_type", ["orthogonal_adamw", "fused_adamw", "muon"])
def test_parse_pcvr_train_args_accepts_loss_terms_and_optimizer_flags(tmp_path: Path, dense_optimizer_type: str) -> None:
    args = parse_pcvr_train_args(
        [
            "--loss-terms",
            "bce:1.0,pairwise_auc:pairwise_auc:0.2",
            "--loss-weight-overrides",
            "pairwise_auc=0.5",
            "--eval-every-n-steps",
            "5000",
            "--dense-optimizer-type",
            dense_optimizer_type,
            "--patience-steps",
            "77",
            "--scheduler-type",
            "cosine",
            "--warmup-steps",
            "256",
            "--min-lr-ratio",
            "0.1",
        ],
        package_dir=tmp_path,
        defaults=PCVRTrainConfig(),
    )

    assert args.loss_terms == [
        {
            "name": "bce",
            "kind": "bce",
            "weight": 1.0,
            "focal_alpha": 0.1,
            "focal_gamma": 2.0,
            "temperature": 1.0,
        },
        {
            "name": "pairwise_auc",
            "kind": "pairwise_auc",
            "weight": 0.5,
            "focal_alpha": 0.1,
            "focal_gamma": 2.0,
            "temperature": 1.0,
        },
    ]
    assert args.eval_every_n_steps == 5000
    assert args.dense_optimizer_type == dense_optimizer_type
    assert args.patience_steps == 77
    assert args.scheduler_type == "cosine"
    assert args.warmup_steps == 256
    assert args.min_lr_ratio == pytest.approx(0.1)


def test_parse_pcvr_train_args_uses_scheduler_defaults(tmp_path: Path) -> None:
    defaults = PCVRTrainConfig(
        optimizer=PCVROptimizerConfig(
            scheduler_type="linear",
            warmup_steps=128,
            min_lr_ratio=0.25,
        )
    )

    args = parse_pcvr_train_args([], package_dir=tmp_path, defaults=defaults)

    assert args.scheduler_type == "linear"
    assert args.warmup_steps == 128
    assert args.min_lr_ratio == pytest.approx(0.25)


def test_parse_pcvr_train_args_rejects_data_pipeline_flags(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_pcvr_train_args(
            ["--data-pipeline-transforms", "sequence_crop,feature_mask"],
            package_dir=tmp_path,
            defaults=PCVRTrainConfig(),
        )


def test_parse_pcvr_train_args_uses_typed_data_pipeline_defaults(
    tmp_path: Path,
) -> None:
    defaults = PCVRTrainConfig(
        data_pipeline=PCVRDataPipelineConfig(
            cache=PCVRDataCacheConfig(mode="lru", max_batches=32),
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
            cache=PCVRDataCacheConfig(mode="lru", max_batches=32),
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
        "cache": {"mode": "lru", "max_batches": 32},
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


def test_pcvr_train_config_serializes_optimizer_schedule_fields() -> None:
    flat_config = PCVRTrainConfig(
        optimizer=PCVROptimizerConfig(
            patience_steps=512,
            scheduler_type="cosine",
            warmup_steps=64,
            min_lr_ratio=0.2,
        )
    ).to_flat_dict()

    assert flat_config["patience_steps"] == 512
    assert flat_config["scheduler_type"] == "cosine"
    assert flat_config["warmup_steps"] == 64
    assert flat_config["min_lr_ratio"] == pytest.approx(0.2)


def test_pcvr_train_config_serializes_data_split_fields() -> None:
    flat_config = PCVRTrainConfig(
        data=PCVRDataConfig(
            train_steps_per_sweep=128,
            split_strategy="timestamp_range",
            train_timestamp_end=10,
            valid_timestamp_start=10,
            valid_timestamp_end=20,
        )
    ).to_flat_dict()

    assert flat_config["eval_every_n_steps"] == 5000
    assert flat_config["train_steps_per_sweep"] == 128
    assert flat_config["split_strategy"] == "timestamp_range"
    assert flat_config["train_timestamp_start"] == 0
    assert flat_config["train_timestamp_end"] == 10
    assert flat_config["valid_timestamp_start"] == 10
    assert flat_config["valid_timestamp_end"] == 20


def test_pcvr_train_config_serializes_gradient_checkpointing_field() -> None:
    flat_config = PCVRTrainConfig(
        model=PCVRModelConfig(gradient_checkpointing=True)
    ).to_flat_dict()

    assert flat_config["gradient_checkpointing"] is True


def test_pcvr_train_config_serializes_rms_norm_fields() -> None:
    flat_config = PCVRTrainConfig(
        model=PCVRModelConfig(rms_norm_backend="triton", rms_norm_block_rows=8)
    ).to_flat_dict()

    assert flat_config["rms_norm_backend"] == "triton"
    assert flat_config["rms_norm_block_rows"] == 8


def test_pcvr_train_config_serializes_flash_attention_backend_field() -> None:
    flat_config = PCVRTrainConfig(
        model=PCVRModelConfig(flash_attention_backend="tilelang")
    ).to_flat_dict()

    assert flat_config["flash_attention_backend"] == "tilelang"


def test_pcvr_train_config_serializes_loss_terms() -> None:
    flat_config = PCVRTrainConfig(
        loss=PCVRLossConfig(
            terms=(
                PCVRLossTermConfig(name="bce", kind="bce", weight=1.0),
                PCVRLossTermConfig(name="aux", kind="model", weight=0.05),
            )
        )
    ).to_flat_dict()

    assert flat_config["loss_terms"] == [
        {
            "name": "bce",
            "kind": "bce",
            "weight": 1.0,
            "focal_alpha": 0.1,
            "focal_gamma": 2.0,
            "temperature": 1.0,
        },
        {
            "name": "aux",
            "kind": "model",
            "weight": 0.05,
            "focal_alpha": 0.1,
            "focal_gamma": 2.0,
            "temperature": 1.0,
        },
    ]


def test_train_pcvr_model_uses_injected_train_hooks(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    ckpt_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    tf_events_dir = tmp_path / "tensorboard"
    schema_path = data_dir / "schema.json"
    data_dir.mkdir()
    schema_path.write_text("{}\n", encoding="utf-8")
    events: list[tuple[str, object]] = []

    def fake_arg_parser(argv, *, package_dir: Path, defaults: PCVRTrainConfig):
        del argv, defaults
        return SimpleNamespace(
            data_dir=str(data_dir),
            schema_path=str(schema_path),
            ckpt_dir=str(ckpt_dir),
            log_dir=str(log_dir),
            tf_events_dir=str(tf_events_dir),
            seed=7,
            amp=False,
            amp_dtype="bfloat16",
            compile=False,
            device="cpu",
            train_ratio=0.8,
            valid_ratio=0.2,
        )

    def build_data(context):
        events.append(("data", context.schema_path))
        return PCVRTrainDataBundle(
            train_loader="train_loader",
            valid_loader="valid_loader",
            dataset="dataset",
            data_module="custom_data_module",
        )

    class _FakeModel:
        def parameters(self):
            return []

    def build_model(context, data_bundle):
        events.append(("model", data_bundle.data_module))
        return _FakeModel()

    class _FakeTrainer:
        trained = False

    def build_trainer(context, data_bundle, model):
        del context
        events.append(("trainer", (data_bundle.train_loader, type(model).__name__)))
        return _FakeTrainer()

    def run_training(context, trainer):
        trainer.trained = True
        events.append(("run", context.config["data_pipeline"]))

    def build_summary(context, trainer):
        return {
            "run_dir": str(context.ckpt_dir),
            "checkpoint_root": str(context.ckpt_dir),
            "schema_path": str(context.schema_path),
            "train_ratio": float(context.args.train_ratio),
            "valid_ratio": float(context.args.valid_ratio),
            "trainer_ran": trainer.trained,
        }

    summary = train_pcvr_model(
        model_module=SimpleNamespace(ModelInput=object),
        model_class_name="InjectedModel",
        package_dir=tmp_path,
        defaults=PCVRTrainConfig(),
        arg_parser=fake_arg_parser,
        train_hooks=build_pcvr_train_hooks(
            build_data=build_data,
            build_model=build_model,
            build_trainer=build_trainer,
            run_training=run_training,
            build_summary=build_summary,
        ),
    )

    assert summary["trainer_ran"] is True
    assert summary["schema_path"] == str(schema_path.resolve())
    assert summary["telemetry"]["label"] == "training"
    assert summary["telemetry"]["model_parameters"] == 0
    assert (ckpt_dir / "training_telemetry.json").exists()
    assert (ckpt_dir / "training_summary.json").exists()
    assert events == [
        ("data", schema_path.resolve()),
        ("model", "custom_data_module"),
        ("trainer", ("train_loader", "_FakeModel")),
        (
            "run",
            {
                "cache": {"mode": "none", "max_batches": 0},
                "seed": None,
                "strict_time_filter": True,
                "transforms": [],
            },
        ),
    ]

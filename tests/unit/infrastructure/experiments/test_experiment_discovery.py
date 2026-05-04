from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from taac2026.infrastructure.experiments.discovery import discover_experiment_paths
from taac2026.infrastructure.io.json_utils import dumps
from tests.unit.infrastructure.pcvr._pcvr_experiment_matrix import build_pcvr_experiment_cases


def _write_minimal_pcvr_experiment(package_dir: Path, *, experiment_name: str, model_class_name: str) -> None:
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        dedent(
            f"""
            from pathlib import Path

            from taac2026.infrastructure.pcvr.config import (
                PCVRDataCacheConfig,
                PCVRDataConfig,
                PCVRDataPipelineConfig,
                PCVRModelConfig,
                PCVRNSConfig,
                PCVROptimizerConfig,
                PCVRSparseOptimizerConfig,
                PCVRTrainConfig,
            )
            from taac2026.infrastructure.pcvr.experiment import PCVRExperiment
            from taac2026.infrastructure.pcvr.prediction_stack import (
                PCVRPredictionHooks,
                default_build_prediction_data,
                default_build_prediction_model,
                default_prepare_prediction_runner,
                default_run_prediction_loop,
            )
            from taac2026.infrastructure.pcvr.runtime_stack import (
                PCVRRuntimeHooks,
                default_build_evaluation_data_diagnostics,
                default_load_runtime_schema,
                default_load_train_config,
                default_resolve_evaluation_checkpoint,
                default_resolve_inference_checkpoint,
                default_write_observed_schema_report,
                default_write_train_split_observed_schema_reports,
            )
            from taac2026.infrastructure.pcvr.train_stack import (
                PCVRTrainHooks,
                default_build_train_data,
                default_build_train_model,
                default_build_train_summary,
                default_build_train_trainer,
                default_run_training,
            )
            from taac2026.infrastructure.pcvr.training import parse_pcvr_train_args
            from taac2026.infrastructure.training.runtime import BinaryClassificationLossConfig, RuntimeExecutionConfig

            TRAIN_HOOKS = PCVRTrainHooks(
                build_data=default_build_train_data,
                build_model=default_build_train_model,
                build_trainer=default_build_train_trainer,
                run_training=default_run_training,
                build_summary=default_build_train_summary,
            )

            PREDICTION_HOOKS = PCVRPredictionHooks(
                build_data=default_build_prediction_data,
                build_model=default_build_prediction_model,
                prepare_predictor=default_prepare_prediction_runner,
                run_loop=default_run_prediction_loop,
            )

            RUNTIME_HOOKS = PCVRRuntimeHooks(
                resolve_evaluation_checkpoint=default_resolve_evaluation_checkpoint,
                resolve_inference_checkpoint=default_resolve_inference_checkpoint,
                load_train_config=default_load_train_config,
                load_runtime_schema=default_load_runtime_schema,
                build_evaluation_data_diagnostics=default_build_evaluation_data_diagnostics,
                write_observed_schema_report=default_write_observed_schema_report,
                write_train_split_observed_schema_reports=default_write_train_split_observed_schema_reports,
            )

            TRAIN_DEFAULTS = PCVRTrainConfig(
                data=PCVRDataConfig(
                    batch_size=256,
                    num_workers=8,
                    buffer_batches=20,
                    train_ratio=1.0,
                    valid_ratio=0.1,
                    eval_every_n_steps=0,
                    seq_max_lens="seq_a:256,seq_b:256,seq_c:512,seq_d:512",
                ),
                data_pipeline=PCVRDataPipelineConfig(
                    cache=PCVRDataCacheConfig(mode="none", max_batches=0),
                    transforms=(),
                    seed=None,
                    strict_time_filter=True,
                ),
                optimizer=PCVROptimizerConfig(
                    lr=1e-4,
                    max_steps=0,
                    patience=5,
                    seed=42,
                    device=None,
                    dense_optimizer_type="adamw",
                    scheduler_type="none",
                    warmup_steps=0,
                    min_lr_ratio=0.0,
                ),
                runtime=RuntimeExecutionConfig(amp=False, amp_dtype="bfloat16", compile=False),
                loss=BinaryClassificationLossConfig(
                    loss_type="bce",
                    focal_alpha=0.1,
                    focal_gamma=2.0,
                    pairwise_auc_weight=0.0,
                    pairwise_auc_temperature=1.0,
                ),
                sparse_optimizer=PCVRSparseOptimizerConfig(
                    sparse_lr=0.05,
                    sparse_weight_decay=0.0,
                    reinit_sparse_every_n_steps=0,
                    reinit_cardinality_threshold=0,
                ),
                model=PCVRModelConfig(
                    d_model=64,
                    emb_dim=64,
                    num_queries=2,
                    num_blocks=2,
                    num_heads=4,
                    seq_encoder_type="transformer",
                    hidden_mult=4,
                    dropout_rate=0.01,
                    seq_top_k=50,
                    seq_causal=False,
                    action_num=1,
                    use_time_buckets=True,
                    rank_mixer_mode="full",
                    use_rope=False,
                    rope_base=10000.0,
                    emb_skip_threshold=1_000_000,
                    seq_id_threshold=10000,
                    gradient_checkpointing=False,
                ),
                ns=PCVRNSConfig(
                    grouping_strategy="explicit",
                    user_groups={{"U1": [0]}},
                    item_groups={{"I1": [0]}},
                    tokenizer_type="rankmixer",
                    user_tokens=5,
                    item_tokens=2,
                ),
            )

            EXPERIMENT = PCVRExperiment(
                name={experiment_name!r},
                package_dir=Path(__file__).resolve().parent,
                model_class_name={model_class_name!r},
                train_defaults=TRAIN_DEFAULTS,
                train_arg_parser=parse_pcvr_train_args,
                train_hooks=TRAIN_HOOKS,
                prediction_hooks=PREDICTION_HOOKS,
                runtime_hooks=RUNTIME_HOOKS,
            )

            __all__ = ["EXPERIMENT", "PREDICTION_HOOKS", "RUNTIME_HOOKS", "TRAIN_DEFAULTS", "TRAIN_HOOKS"]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (package_dir / "model.py").write_text(
        "from taac2026.infrastructure.pcvr.modeling import ModelInput\n"
        "\n"
        f"class {model_class_name}:\n"
        "    pass\n"
        "\n"
        f"__all__ = [\"ModelInput\", \"{model_class_name}\"]\n",
        encoding="utf-8",
    )


def test_discover_experiment_paths_filters_to_valid_packages(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiments" / "pcvr"
    experiment_root.mkdir(parents=True)

    valid = experiment_root / "valid_exp"
    valid.mkdir()
    for name in ("__init__.py", "model.py"):
        (valid / name).write_text("", encoding="utf-8")

    missing_model = experiment_root / "missing_model"
    missing_model.mkdir()
    (missing_model / "__init__.py").write_text("", encoding="utf-8")

    hidden = experiment_root / "__pycache__"
    hidden.mkdir()
    for name in ("__init__.py", "model.py"):
        (hidden / name).write_text("", encoding="utf-8")

    assert discover_experiment_paths(experiment_root) == ["experiments/pcvr/valid_exp"]


def test_build_pcvr_experiment_cases_discovers_minimal_new_package(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiments" / "pcvr"
    _write_minimal_pcvr_experiment(
        experiment_root / "minimal_exp",
        experiment_name="pcvr_minimal_exp",
        model_class_name="PCVRMinimalExp",
    )

    cases = build_pcvr_experiment_cases(experiment_root)

    assert len(cases) == 1
    assert cases[0].path == "experiments/pcvr/minimal_exp"
    assert cases[0].module == "experiments.pcvr.minimal_exp"
    assert cases[0].name == "pcvr_minimal_exp"
    assert cases[0].model_class == "PCVRMinimalExp"
    assert cases[0].package_dir == (experiment_root / "minimal_exp").resolve()
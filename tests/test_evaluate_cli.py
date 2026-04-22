from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from config.baseline.data import DENSE_FEATURE_DIM, load_dataloaders
from taac2026.application.evaluation.cli import _build_single_summary_rows, _format_quantization_summary, parse_args
from taac2026.application.evaluation.inference import normalize_inference_export_mode
from taac2026.application.evaluation.service import _sort_records, evaluate_checkpoint
from taac2026.domain.config import ModelConfig
from taac2026.domain.features import FeatureSchema, FeatureTableSpec
from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter
from tests.support import TestWorkspace, TinyExperimentModel, create_test_workspace


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


def test_evaluate_checkpoint_accepts_compatible_checkpoint(test_workspace: TestWorkspace) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)
    _, _, data_stats = load_dataloaders(
        config=experiment.data,
        vocab_size=experiment.model.vocab_size,
        batch_size=experiment.train.batch_size,
        eval_batch_size=experiment.train.resolved_eval_batch_size,
        num_workers=experiment.train.num_workers,
        seed=experiment.train.seed,
    )
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    checkpoint_path = test_workspace.root / "compatible.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    output_path = test_workspace.root / "evaluation.json"
    payload = evaluate_checkpoint(
        experiment_path=experiment_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )

    assert payload["model_name"] == "temp_experiment"
    assert "coverage" in payload["metrics"]["gauc"]
    assert payload["profiling"]["schema_version"] == 1
    assert payload["runtime_optimization"]["torch_compile"]["active"] is False
    assert "external_profilers" in payload["profiling"]
    assert output_path.exists()
    assert (test_workspace.root / "profiling" / "external_profilers.json").exists()


def test_evaluate_checkpoint_rejects_incompatible_checkpoint(test_workspace: TestWorkspace) -> None:
    experiment_path = test_workspace.write_experiment_package(hidden_dim=16, embedding_dim=16, num_heads=4)
    bad_model_config = ModelConfig(
        name="temp_experiment",
        vocab_size=257,
        embedding_dim=8,
        hidden_dim=8,
        dropout=0.0,
        num_layers=1,
        num_heads=2,
        recent_seq_len=2,
        memory_slots=2,
        ffn_multiplier=2,
        feature_cross_layers=1,
        sequence_layers=1,
        static_layers=1,
        query_decoder_layers=1,
        fusion_layers=1,
        num_queries=2,
        head_hidden_dim=8,
        segment_count=4,
    )
    bad_checkpoint = test_workspace.root / "incompatible.pt"
    bad_model = TinyExperimentModel(test_workspace.data_config, bad_model_config, DENSE_FEATURE_DIM)
    torch.save({"model_state_dict": bad_model.state_dict()}, bad_checkpoint)

    with pytest.raises(RuntimeError, match="incompatible"):
        evaluate_checkpoint(
            experiment_path=experiment_path,
            checkpoint_path=bad_checkpoint,
            output_path=test_workspace.root / "incompatible_evaluation.json",
        )


def test_evaluate_checkpoint_enables_cpu_bfloat16_amp_when_preconfigured(test_workspace: TestWorkspace) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)
    experiment.train.enable_amp = True
    experiment.train.amp_dtype = "bfloat16"
    _, _, data_stats = load_dataloaders(
        config=experiment.data,
        vocab_size=experiment.model.vocab_size,
        batch_size=experiment.train.batch_size,
        eval_batch_size=experiment.train.resolved_eval_batch_size,
        num_workers=experiment.train.num_workers,
        seed=experiment.train.seed,
    )
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    checkpoint_path = test_workspace.root / "compatible_amp.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    output_path = test_workspace.root / "evaluation_amp.json"
    payload = evaluate_checkpoint(
        experiment_path=experiment_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        experiment=experiment,
    )

    assert payload["runtime_optimization"]["amp"]["requested"] is True
    assert payload["runtime_optimization"]["amp"]["active"] is True
    assert payload["runtime_optimization"]["amp"]["resolved_dtype"] == "bfloat16"
    assert "--amp --amp-dtype bfloat16" in payload["profiling"]["external_profilers"]["tools"]["ncu"]["suggested_command_string"]


def test_evaluate_checkpoint_supports_int8_quantized_inference(test_workspace: TestWorkspace) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)
    _, _, data_stats = load_dataloaders(
        config=experiment.data,
        vocab_size=experiment.model.vocab_size,
        batch_size=experiment.train.batch_size,
        eval_batch_size=experiment.train.resolved_eval_batch_size,
        num_workers=experiment.train.num_workers,
        seed=experiment.train.seed,
    )
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    checkpoint_path = test_workspace.root / "compatible_int8.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    payload = evaluate_checkpoint(
        experiment_path=experiment_path,
        checkpoint_path=checkpoint_path,
        output_path=test_workspace.root / "evaluation_int8.json",
        quantization_mode="int8",
    )

    assert payload["device"] == "cpu"
    assert payload["quantization"]["active"] is True
    assert payload["quantization"]["mode"] == "int8"
    assert payload["quantization"]["quantized_linear_layers"] > 0
    assert payload["quantization"]["reason"] == "dynamic int8 inference quantized nn.Linear modules via torchao on cpu"
    assert "requested_mode" not in payload["quantization"]
    assert "quantizable_linear_layers" not in payload["quantization"]
    assert "quantizable_embedding_collections" not in payload["quantization"]
    assert "quantized_embedding_collections" not in payload["quantization"]
    assert "blocked_embedding_collections" not in payload["quantization"]
    assert "blockers" not in payload["quantization"]
    assert payload["quantization"]["runtime_overrides"]["forced_device"] == "cpu"


def test_format_quantization_summary_reports_linear_counts() -> None:
    actual = _format_quantization_summary(
        {
            "mode": "int8",
            "quantized_linear_layers": 2,
        }
    )

    assert actual == "int8 | linear=2"


def test_format_quantization_summary_marks_inactive_int8() -> None:
    actual = _format_quantization_summary(
        {
            "mode": "int8",
            "active": False,
            "quantized_linear_layers": 0,
            "reason": "model has no nn.Linear modules eligible for dynamic int8 quantization",
        }
    )

    assert actual == "int8 | inactive"


def test_build_single_summary_rows_includes_quantization_reason_only() -> None:
    report = {
        "experiment": "baseline",
        "experiment_path": "config/baseline",
        "checkpoint_path": "outputs/best.pt",
        "device": "cpu",
        "quantization": {
            "mode": "int8",
            "quantized_linear_layers": 3,
            "reason": "dynamic int8 inference quantized nn.Linear modules via torchao on cpu",
        },
        "export": {"mode": "none", "artifact_path": None},
        "loss": 0.125,
        "metrics": {"auc": 0.75, "pr_auc": 0.5},
        "mean_latency_ms_per_sample": 0.25,
        "p95_latency_ms_per_sample": 0.4,
    }

    rows = dict(_build_single_summary_rows(report))

    assert rows["quantization"] == "int8 | linear=3"
    assert rows["quantization_reason"] == report["quantization"]["reason"]
    assert "quantization_blockers" not in rows


def test_normalize_inference_export_mode_rejects_legacy_aliases() -> None:
    assert normalize_inference_export_mode(None) == "none"
    with pytest.raises(ValueError, match="Unsupported export mode"):
        normalize_inference_export_mode("export")
    with pytest.raises(ValueError, match="Unsupported export mode"):
        normalize_inference_export_mode("pt2")


def test_evaluate_checkpoint_can_export_model_for_inference(test_workspace: TestWorkspace) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)
    _, _, data_stats = load_dataloaders(
        config=experiment.data,
        vocab_size=experiment.model.vocab_size,
        batch_size=experiment.train.batch_size,
        eval_batch_size=experiment.train.resolved_eval_batch_size,
        num_workers=experiment.train.num_workers,
        seed=experiment.train.seed,
    )
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    checkpoint_path = test_workspace.root / "compatible_export.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    export_path = test_workspace.root / "inference_export.pt2"

    payload = evaluate_checkpoint(
        experiment_path=experiment_path,
        checkpoint_path=checkpoint_path,
        output_path=test_workspace.root / "evaluation_export.json",
        export_mode="torch-export",
        export_path=export_path,
    )

    assert payload["export"]["active"] is True
    assert payload["export"]["mode"] == "torch-export"
    assert payload["export"]["artifact_path"] == str(export_path)
    assert export_path.exists()


def test_evaluate_checkpoint_rejects_quantized_export_combination(test_workspace: TestWorkspace) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)
    _, _, data_stats = load_dataloaders(
        config=experiment.data,
        vocab_size=experiment.model.vocab_size,
        batch_size=experiment.train.batch_size,
        eval_batch_size=experiment.train.resolved_eval_batch_size,
        num_workers=experiment.train.num_workers,
        seed=experiment.train.seed,
    )
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    checkpoint_path = test_workspace.root / "compatible_export_int8.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="requires quantization mode 'none'"):
        evaluate_checkpoint(
            experiment_path=experiment_path,
            checkpoint_path=checkpoint_path,
            output_path=test_workspace.root / "evaluation_export_int8.json",
            quantization_mode="int8",
            export_mode="torch-export",
        )


def test_evaluate_checkpoint_rejects_int8_for_torchrec_embedding_collection_model(
    test_workspace: TestWorkspace,
) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)

    feature_schema = FeatureSchema(
        tables=(
            FeatureTableSpec(name="user_tokens", family="user", num_embeddings=64, embedding_dim=4),
            FeatureTableSpec(name="context_tokens", family="context", num_embeddings=64, embedding_dim=4),
            FeatureTableSpec(name="candidate_tokens", family="candidate", num_embeddings=64, embedding_dim=4),
        ),
        dense_dim=DENSE_FEATURE_DIM,
    )

    class _TorchRecEvaluationModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = TorchRecEmbeddingBagAdapter(
                feature_schema,
                table_names=("user_tokens", "context_tokens", "candidate_tokens"),
            )
            self.output = nn.Linear(self.embedding.output_dim + DENSE_FEATURE_DIM, 1)

        def forward(self, batch) -> torch.Tensor:
            if batch.sparse_features is None:
                raise RuntimeError("Batch is missing sparse_features")
            pooled = self.embedding(batch.sparse_features)
            fused = torch.cat([pooled, batch.dense_features], dim=-1)
            return self.output(fused).squeeze(-1)

    experiment.build_model_component = lambda data_config, model_config, dense_dim: _TorchRecEvaluationModel()
    model = experiment.build_model_component(experiment.data, experiment.model, DENSE_FEATURE_DIM)
    checkpoint_path = test_workspace.root / "torchrec_ebc_int8.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="does not support TorchRec EmbeddingBagCollection modules"):
        evaluate_checkpoint(
            experiment_path=experiment_path,
            checkpoint_path=checkpoint_path,
            output_path=test_workspace.root / "evaluation_torchrec_ebc_int8.json",
            experiment=experiment,
            quantization_mode="int8",
        )


def test_evaluate_checkpoint_skips_export_example_batch_when_export_is_disabled(
    test_workspace: TestWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_path = test_workspace.write_experiment_package()
    experiment = load_experiment_package(experiment_path)
    model = experiment.build_model_component(experiment.data, experiment.model, DENSE_FEATURE_DIM)
    checkpoint_path = test_workspace.root / "compatible_no_export_batch.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    class _SentinelLoader:
        def __iter__(self):
            raise AssertionError("export_mode='none' should not materialize an example batch")

    fake_stats = SimpleNamespace(dense_dim=DENSE_FEATURE_DIM, pos_weight=1.0, val_size=1)

    monkeypatch.setattr(
        "taac2026.application.evaluation.service.resolve_experiment_builders",
        lambda _experiment: SimpleNamespace(
            build_data_pipeline=lambda *args: ([], _SentinelLoader(), fake_stats),
            build_loss_stack=lambda *args: (torch.nn.BCEWithLogitsLoss(), None),
        ),
    )

    class _RuntimeExecution:
        def __init__(self, base_model: torch.nn.Module) -> None:
            self.execution_model = base_model
            self.base_model = base_model
            self.device = torch.device("cpu")

        def summary(self) -> dict[str, object]:
            return {
                "torch_compile": {"active": False},
                "amp": {"active": False},
            }

    monkeypatch.setattr(
        "taac2026.application.evaluation.service.prepare_evaluation_inference",
        lambda model, train_config, device, quantization_mode=None: (
            _RuntimeExecution(model),
            {"mode": "none", "active": False},
            train_config,
        ),
    )
    monkeypatch.setattr(
        "taac2026.application.evaluation.service.collect_loader_outputs",
        lambda *args, **kwargs: (
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([0]),
            0.0,
        ),
    )
    monkeypatch.setattr(
        "taac2026.application.evaluation.service.compute_classification_metrics",
        lambda labels, logits, groups: {"auc": 0.5, "pr_auc": 0.5, "gauc": {"coverage": 1.0}},
    )
    monkeypatch.setattr(
        "taac2026.application.evaluation.service.measure_latency",
        lambda *args, **kwargs: {
            "warmup_batches": 0,
            "warmup_samples": 0,
            "measured_batches": 0,
            "measured_samples": 0,
            "profiled_batches": 0,
            "mean_latency_ms_per_sample": 0.0,
            "p95_latency_ms_per_sample": 0.0,
        },
    )
    monkeypatch.setattr(
        "taac2026.application.evaluation.service.build_evaluation_external_profiler_plan",
        lambda **kwargs: {"tools": {}},
    )
    monkeypatch.setattr(
        "taac2026.application.evaluation.service.write_external_profiler_plan_artifacts",
        lambda *args, **kwargs: None,
    )

    payload = evaluate_checkpoint(
        experiment_path=experiment_path,
        checkpoint_path=checkpoint_path,
        output_path=test_workspace.root / "evaluation_no_export_batch.json",
        experiment=experiment,
        export_mode="none",
    )

    assert payload["export"]["active"] is False
    assert payload["export"]["mode"] == "none"


@pytest.mark.parametrize(
    ("argv", "expected_command", "expected_value"),
    [
        (["single", "--experiment", "config/oo", "--run-dir", "outputs/example"], "single", "outputs/example"),
        (["single", "--experiment", "config/oo", "--compile", "--amp", "--amp-dtype", "bfloat16"], "single", None),
        (
            [
                "batch",
                "--experiment-paths",
                "config/baseline",
                "config/interformer",
            ],
            "batch",
            ["config/baseline", "config/interformer"],
        ),
    ],
)
def test_parse_args_routes_subcommands(argv, expected_command, expected_value) -> None:
    args = parse_args(argv)

    assert args.command == expected_command
    if expected_command == "single":
        assert args.experiment == "config/oo"
        if expected_value is not None:
            assert args.run_dir == expected_value
    else:
        assert args.experiment_paths == expected_value


def test_parse_args_accepts_runtime_optimization_flags() -> None:
    args = parse_args([
        "single",
        "--experiment",
        "config/oo",
        "--quantize",
        "int8",
        "--export-mode",
        "torch-export",
        "--export-path",
        "outputs/exported_model.pt2",
        "--compile",
        "--compile-backend",
        "inductor",
        "--compile-mode",
        "max-autotune",
        "--amp",
        "--amp-dtype",
        "bfloat16",
    ])

    assert args.compile is True
    assert args.compile_backend == "inductor"
    assert args.compile_mode == "max-autotune"
    assert args.amp is True
    assert args.amp_dtype == "bfloat16"
    assert args.quantize == "int8"
    assert args.export_mode == "torch-export"
    assert args.export_path == "outputs/exported_model.pt2"


def test_parse_args_accepts_batch_runtime_optimization_flags() -> None:
    args = parse_args([
        "batch",
        "--experiment-paths",
        "config/baseline",
        "config/interformer",
        "--quantize",
        "int8",
        "--export-mode",
        "torch-export",
        "--compile",
        "--amp",
    ])

    assert args.command == "batch"
    assert args.compile is True
    assert args.amp is True
    assert args.quantize == "int8"
    assert args.export_mode == "torch-export"


def test_parse_args_requires_explicit_batch_experiments() -> None:
    with pytest.raises(SystemExit):
        parse_args(["batch"])


def test_batch_report_sort_prefers_budget_compliant_runs() -> None:
    records = [
        {
            "experiment_id": "E001",
            "experiment_path": "experiments/slow_but_high_auc",
            "auc": 0.91,
            "pr_auc": 0.40,
            "mean_latency_ms_per_sample": 0.30,
            "latency_budget_ms_per_sample": 0.20,
            "latency_budget_met": False,
        },
        {
            "experiment_id": "E002",
            "experiment_path": "experiments/unconstrained",
            "auc": 0.85,
            "pr_auc": 0.32,
            "mean_latency_ms_per_sample": 0.15,
            "latency_budget_ms_per_sample": 0.0,
            "latency_budget_met": True,
        },
        {
            "experiment_id": "E003",
            "experiment_path": "experiments/qualified_a",
            "auc": 0.80,
            "pr_auc": 0.30,
            "mean_latency_ms_per_sample": 0.18,
            "latency_budget_ms_per_sample": 0.20,
            "latency_budget_met": True,
        },
        {
            "experiment_id": "E004",
            "experiment_path": "experiments/qualified_b",
            "auc": 0.88,
            "pr_auc": 0.35,
            "mean_latency_ms_per_sample": 0.19,
            "latency_budget_ms_per_sample": 0.20,
            "latency_budget_met": True,
        },
    ]

    ranked = _sort_records(records)

    assert [record["experiment_id"] for record in ranked] == ["E004", "E003", "E002", "E001"]

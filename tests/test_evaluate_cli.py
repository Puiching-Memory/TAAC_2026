from __future__ import annotations

from pathlib import Path

import pytest
import torch

from config.gen.baseline.data import DENSE_FEATURE_DIM, load_dataloaders
from taac2026.application.evaluation.cli import parse_args
from taac2026.application.evaluation.service import _sort_records, evaluate_checkpoint
from taac2026.domain.config import ModelConfig
from taac2026.infrastructure.experiments.loader import load_experiment_package
from tests.support import TestWorkspace, TinyExperimentModel, create_test_workspace


@pytest.fixture
def test_workspace(tmp_path: Path) -> TestWorkspace:
    return create_test_workspace(tmp_path)


@pytest.mark.integration
@pytest.mark.smoke
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
    assert output_path.exists()


@pytest.mark.integration
@pytest.mark.smoke
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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("argv", "expected_command", "expected_value"),
    [
        (["single", "--experiment", "config/gen/oo", "--run-dir", "outputs/example"], "single", "outputs/example"),
        (
            [
                "batch",
                "--experiment-paths",
                "config/gen/baseline",
                "config/gen/interformer",
            ],
            "batch",
            ["config/gen/baseline", "config/gen/interformer"],
        ),
    ],
)
def test_parse_args_routes_subcommands(argv, expected_command, expected_value) -> None:
    args = parse_args(argv)

    assert args.command == expected_command
    if expected_command == "single":
        assert args.experiment == "config/gen/oo"
        assert args.run_dir == expected_value
    else:
        assert args.experiment_paths == expected_value


@pytest.mark.unit
def test_parse_args_requires_explicit_batch_experiments() -> None:
    with pytest.raises(SystemExit):
        parse_args(["batch"])


@pytest.mark.unit
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

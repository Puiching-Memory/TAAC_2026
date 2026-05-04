from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
import torch

from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.io.json_utils import dumps
from taac2026.infrastructure.pcvr.modeling import safe_key_padding_mask
from tests.unit.infrastructure.pcvr._pcvr_experiment_matrix import discover_pcvr_experiment_cases, get_experiment_case, load_model_module


EXPERIMENT_CASES = discover_pcvr_experiment_cases()


def _sample_model_input(model_module):
    return model_module.ModelInput(
        user_int_feats=torch.tensor([[1, 2, 3], [4, 0, 1]], dtype=torch.long),
        item_int_feats=torch.tensor([[1], [2]], dtype=torch.long),
        user_dense_feats=torch.randn(2, 2),
        item_dense_feats=torch.randn(2, 1),
        seq_data={
            "seq_a": torch.tensor(
                [
                    [[1, 2, 0, 0], [2, 3, 0, 0]],
                    [[4, 1, 2, 3], [1, 2, 3, 4]],
                ],
                dtype=torch.long,
            ),
            "seq_b": torch.tensor([[[1, 0, 0]], [[2, 3, 0]]], dtype=torch.long),
        },
        seq_lens={
            "seq_a": torch.tensor([2, 4], dtype=torch.long),
            "seq_b": torch.tensor([1, 2], dtype=torch.long),
        },
        seq_time_buckets={
            "seq_a": torch.zeros(2, 4, dtype=torch.long),
            "seq_b": torch.zeros(2, 3, dtype=torch.long),
        },
    )


def _make_model(experiment_case, model_module, overrides=None):
    model_class = getattr(model_module, experiment_case.model_class)
    model_kwargs = dict(
        user_int_feature_specs=[(8, 0, 1), (7, 1, 2)],
        item_int_feature_specs=[(5, 0, 1)],
        user_dense_dim=2,
        item_dense_dim=1,
        seq_vocab_sizes={"seq_a": [6, 5], "seq_b": [4]},
        user_ns_groups=[[0], [1]],
        item_ns_groups=[[0]],
        d_model=16,
        emb_dim=8,
        num_blocks=1,
        num_heads=2,
        hidden_mult=2,
        dropout_rate=0.0,
        action_num=1,
        num_time_buckets=0,
        ns_tokenizer_type="rankmixer",
        user_ns_tokens=2,
        item_ns_tokens=1,
    )
    if overrides:
        model_kwargs.update(overrides)
    try:
        return model_class(**model_kwargs)
    except ValueError as error:
        match = re.search(r"=(\d+)\. Valid T values", str(error))
        if "must be divisible by T" not in str(error) or match is None:
            raise
        model_kwargs["d_model"] = max(int(match.group(1)) * 4, model_kwargs["d_model"])
        return model_class(**model_kwargs)


@pytest.mark.parametrize("experiment_case", EXPERIMENT_CASES, ids=lambda case: case.path)
def test_discovered_experiment_packages_load(experiment_case) -> None:
    experiment = load_experiment_package(experiment_case.path)
    train_defaults = experiment.train_defaults.to_flat_dict()

    assert experiment.name == experiment_case.name
    assert experiment.package_dir == experiment_case.package_dir
    assert experiment.train_defaults is not None
    assert experiment.metadata["kind"] == "pcvr"
    assert experiment.metadata["model_class"] == experiment_case.model_class
    if experiment_case.path == "experiments/pcvr/symbiosis":
        assert experiment.metadata["train_arg_parser"] == "parse_symbiosis_train_args"
        assert experiment.metadata["train_build_model"] == "build_symbiosis_train_model"
        assert experiment.metadata["prediction_build_model"] == "build_symbiosis_prediction_model"
        assert experiment.metadata["runtime_load_train_config"] == "load_symbiosis_train_config"
    else:
        assert experiment.metadata["train_arg_parser"] == "parse_pcvr_train_args"
        assert experiment.metadata["train_build_model"] == "default_build_train_model"
        assert experiment.metadata["prediction_build_model"] == "default_build_prediction_model"
        assert experiment.metadata["runtime_load_train_config"] == "default_load_train_config"
    assert experiment.metadata["train_build_data"] == "default_build_train_data"
    assert experiment.metadata["train_build_trainer"] == "default_build_train_trainer"
    assert experiment.metadata["train_run_training"] == "default_run_training"
    assert experiment.metadata["prediction_build_data"] == "default_build_prediction_data"
    assert experiment.metadata["prediction_prepare_predictor"] == "default_prepare_prediction_runner"
    assert experiment.metadata["prediction_run_loop"] == "default_run_prediction_loop"
    assert experiment.metadata["runtime_resolve_evaluation_checkpoint"] == "default_resolve_evaluation_checkpoint"
    assert experiment.metadata["runtime_resolve_inference_checkpoint"] == "default_resolve_inference_checkpoint"
    assert experiment.metadata["runtime_load_runtime_schema"] == "default_load_runtime_schema"
    assert experiment.metadata["runtime_build_evaluation_data_diagnostics"] == "default_build_evaluation_data_diagnostics"
    assert experiment.metadata["runtime_write_observed_schema_report"] == "default_write_observed_schema_report"
    assert experiment.metadata["runtime_write_train_split_observed_schema_reports"] == "default_write_train_split_observed_schema_reports"
    assert train_defaults["ns_grouping_strategy"] == "explicit"
    assert isinstance(train_defaults["user_ns_groups"], dict)
    assert isinstance(train_defaults["item_ns_groups"], dict)
    assert train_defaults["user_ns_groups"]
    assert train_defaults["item_ns_groups"]
    assert experiment.train_defaults.ns.metadata["_purpose"]
    assert experiment.train_defaults.ns.metadata["_usage"]
    assert "num_hyformer_blocks" not in train_defaults
    assert "symbiosis_use_candidate_decoder" not in train_defaults
    assert "symbiosis_recent_tokens" not in train_defaults


@pytest.mark.parametrize("experiment_case", EXPERIMENT_CASES, ids=lambda case: case.path)
def test_discovered_experiment_models_forward_and_predict(experiment_case) -> None:
    model_module = load_model_module(experiment_case)
    if experiment_case.path not in {"experiments/pcvr/baseline", "experiments/pcvr/hyformer"}:
        assert not hasattr(model_module, "PCVRHyFormer")
    assert hasattr(model_module, experiment_case.model_class)
    model = _make_model(experiment_case, model_module)
    model_input = _sample_model_input(model_module)

    logits = model(model_input)
    loss = logits.sum()
    loss.backward()

    model.eval()
    with torch.no_grad():
        predicted_logits, embeddings = model.predict(model_input)

    assert logits.shape == (2, 1)
    assert predicted_logits.shape == (2, 1)
    assert embeddings.shape[0] == 2
    assert model.num_ns > 0
    assert torch.isfinite(logits).all()
    assert torch.isfinite(predicted_logits).all()


def test_symbiosis_enables_amp_and_compile_by_default() -> None:
    experiment = load_experiment_package("experiments/pcvr/symbiosis")
    train_defaults = experiment.train_defaults.to_flat_dict()
    symbiosis_module = importlib.import_module("experiments.pcvr.symbiosis")
    symbiosis_args = symbiosis_module.parse_symbiosis_train_args(
        [],
        package_dir=symbiosis_module.EXPERIMENT.package_dir,
        defaults=symbiosis_module.TRAIN_DEFAULTS,
    )

    assert train_defaults["amp"] is True
    assert train_defaults["amp_dtype"] == "bfloat16"
    assert train_defaults["compile"] is True
    assert train_defaults["pairwise_auc_weight"] == pytest.approx(0.05)
    assert train_defaults["dense_optimizer_type"] == "orthogonal_adamw"
    assert train_defaults["scheduler_type"] == "cosine"
    assert train_defaults["warmup_steps"] == 2000
    assert train_defaults["min_lr_ratio"] == pytest.approx(0.1)
    assert "symbiosis_use_candidate_decoder" not in train_defaults
    assert symbiosis_args.symbiosis_use_candidate_decoder is True
    assert symbiosis_args.symbiosis_use_action_conditioning is True
    assert symbiosis_args.symbiosis_use_compressed_memory is True
    assert symbiosis_args.symbiosis_use_attention_sink is True
    assert symbiosis_args.symbiosis_use_lane_mixing is True
    assert symbiosis_args.symbiosis_use_semantic_id is True


def test_symbiosis_runtime_config_requires_package_specific_keys(tmp_path: Path) -> None:
    symbiosis_module = importlib.import_module("experiments.pcvr.symbiosis")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "train_config.json").write_text(
        dumps(symbiosis_module.TRAIN_DEFAULTS.to_flat_dict()),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="symbiosis_use_user_item_graph"):
        symbiosis_module.load_symbiosis_train_config(symbiosis_module.EXPERIMENT, checkpoint_dir)

@pytest.mark.parametrize("experiment_case", EXPERIMENT_CASES, ids=lambda case: case.path)
def test_discovered_experiment_models_backward_with_gradient_checkpointing(experiment_case) -> None:
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module, overrides={"gradient_checkpointing": True})
    model_input = _sample_model_input(model_module)

    logits = model(model_input)
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_symbiosis_keeps_sequence_width_stable_for_compile() -> None:
    experiment_case = get_experiment_case("experiments/pcvr/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module)
    model_input = model_module.ModelInput(
        user_int_feats=torch.tensor([[1, 2, 3], [4, 0, 1]], dtype=torch.long),
        item_int_feats=torch.tensor([[1], [2]], dtype=torch.long),
        user_dense_feats=torch.randn(2, 2),
        item_dense_feats=torch.randn(2, 1),
        seq_data={
            "seq_a": torch.tensor(
                [
                    [[1, 2, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0]],
                    [[4, 1, 2, 0, 0, 0], [1, 2, 3, 0, 0, 0]],
                ],
                dtype=torch.long,
            ),
            "seq_b": torch.tensor([[[1, 0, 0, 0, 0]], [[2, 3, 0, 0, 0]]], dtype=torch.long),
        },
        seq_lens={
            "seq_a": torch.tensor([2, 3], dtype=torch.long),
            "seq_b": torch.tensor([1, 2], dtype=torch.long),
        },
        seq_time_buckets={
            "seq_a": torch.zeros(2, 6, dtype=torch.long),
            "seq_b": torch.zeros(2, 5, dtype=torch.long),
        },
    )

    sequences, masks, lengths = model._encode_sequences(model_input)

    assert [tensor.shape[1] for tensor in sequences] == [6, 5]
    assert [mask.shape[1] for mask in masks] == [6, 5]
    assert [length.tolist() for length in lengths] == [[2, 3], [1, 2]]


def test_safe_key_padding_mask_unmasks_first_position_for_fully_padded_rows() -> None:
    mask = torch.tensor(
        [
            [True, True, True],
            [False, True, True],
        ],
        dtype=torch.bool,
    )

    safe_mask = safe_key_padding_mask(mask)

    assert torch.equal(
        safe_mask,
        torch.tensor(
            [
                [False, True, True],
                [False, True, True],
            ],
            dtype=torch.bool,
        ),
    )


def test_symbiosis_compile_handles_multiple_sequence_lengths() -> None:
    experiment_case = get_experiment_case("experiments/pcvr/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module)
    input_a = _sample_model_input(model_module)
    input_b = input_a._replace(
        seq_lens={
            "seq_a": torch.tensor([1, 2], dtype=torch.long),
            "seq_b": torch.tensor([3, 1], dtype=torch.long),
        },
        seq_time_buckets={
            "seq_a": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            "seq_b": torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long),
        },
    )

    with torch.no_grad():
        eager_logits_a = model(input_a)
        eager_logits_b = model(input_b)

    compiled = torch.compile(model, backend="eager")
    with torch.no_grad():
        logits_a = compiled(input_a)
        logits_b = compiled(input_b)

    assert logits_a.shape == (2, 1)
    assert logits_b.shape == (2, 1)
    assert torch.allclose(logits_a, eager_logits_a)
    assert torch.allclose(logits_b, eager_logits_b)


def test_symbiosis_unified_stage_uses_compact_context_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment_case = get_experiment_case("experiments/pcvr/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module)
    model_input = _sample_model_input(model_module)
    observed_unified_token_counts: list[int] = []
    original_forward = model.unified_blocks[0].forward

    def recording_forward(tokens, padding_mask, sequences, masks, modulation):
        observed_unified_token_counts.append(int(tokens.shape[1]))
        return original_forward(tokens, padding_mask, sequences, masks, modulation)

    monkeypatch.setattr(model.unified_blocks[0], "forward", recording_forward)

    logits = model(model_input)

    sequence_token_count = sum(int(sequence.shape[2]) for sequence in model_input.seq_data.values())
    context_token_count = model.num_prompt_tokens + model.num_ns

    assert logits.shape == (2, 1)
    assert observed_unified_token_counts == [context_token_count]
    assert context_token_count < context_token_count + sequence_token_count


def test_symbiosis_ablation_flags_disable_optional_modules() -> None:
    experiment_case = get_experiment_case("experiments/pcvr/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(
        experiment_case,
        model_module,
        overrides={
            "symbiosis_use_user_item_graph": False,
            "symbiosis_use_fourier_time": False,
            "symbiosis_use_context_exchange": False,
            "symbiosis_use_multi_scale": False,
            "symbiosis_use_domain_gate": True,
            "symbiosis_use_candidate_decoder": False,
            "symbiosis_use_action_conditioning": False,
            "symbiosis_use_compressed_memory": False,
            "symbiosis_use_attention_sink": False,
            "symbiosis_use_lane_mixing": False,
            "symbiosis_use_semantic_id": False,
        },
    )
    model_input = _sample_model_input(model_module)

    assert len(model.graph_blocks) == 0
    assert len(model.context_blocks) == 0
    assert model.symbiosis_use_fourier_time is False
    assert model.symbiosis_use_multi_scale is False
    assert model.candidate_decoder is None
    assert model.lane_mixer is None
    assert model.semantic_projection is None
    assert all(block.use_domain_gate for block in model.unified_blocks)

    logits = model(model_input)
    predicted_logits, embeddings = model.predict(model_input)

    assert logits.shape == (2, 1)
    assert predicted_logits.shape == (2, 1)
    assert embeddings.shape == (2, model.d_model)
    assert torch.isfinite(logits).all()
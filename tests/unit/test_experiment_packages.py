from __future__ import annotations

import re

import pytest
import torch
import torch._dynamo as dynamo
from torch._dynamo.utils import counters

from taac2026.infrastructure.experiments.loader import load_experiment_package
from taac2026.infrastructure.pcvr.modeling import safe_key_padding_mask
from tests.unit._pcvr_experiment_matrix import discover_pcvr_experiment_cases, get_experiment_case, load_model_module


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
    assert (experiment.package_dir / "ns_groups.json").exists()
    assert train_defaults["ns_groups_json"] == "ns_groups.json"
    assert "num_hyformer_blocks" not in train_defaults


@pytest.mark.parametrize("experiment_case", EXPERIMENT_CASES, ids=lambda case: case.path)
def test_discovered_experiment_models_forward_and_predict(experiment_case) -> None:
    model_module = load_model_module(experiment_case)
    if experiment_case.path not in {"config/baseline", "config/hyformer"}:
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
    experiment = load_experiment_package("config/symbiosis")
    train_defaults = experiment.train_defaults.to_flat_dict()

    assert train_defaults["amp"] is True
    assert train_defaults["amp_dtype"] == "bfloat16"
    assert train_defaults["compile"] is True


def test_symbiosis_keeps_sequence_width_stable_for_compile() -> None:
    experiment_case = get_experiment_case("config/symbiosis")
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


def test_symbiosis_compile_reuses_a_single_graph_across_sequence_lengths() -> None:
    experiment_case = get_experiment_case("config/symbiosis")
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

    dynamo.reset()
    counters.clear()
    compiled = torch.compile(model, backend="eager")
    try:
        with torch.no_grad():
            logits_a = compiled(input_a)
            logits_b = compiled(input_b)
    finally:
        graph_breaks = dict(counters["graph_break"])
        unique_graphs = counters["stats"].get("unique_graphs", 0)
        dynamo.reset()
        counters.clear()

    assert logits_a.shape == (2, 1)
    assert logits_b.shape == (2, 1)
    assert graph_breaks == {}
    assert unique_graphs == 1


def test_symbiosis_unified_attention_uses_context_bottleneck() -> None:
    experiment_case = get_experiment_case("config/symbiosis")
    model_module = load_model_module(experiment_case)
    model = _make_model(experiment_case, model_module)
    model_input = _sample_model_input(model_module)
    observed_self_attention_lengths: list[int] = []

    handle = model.unified_blocks[0].attention.register_forward_pre_hook(
        lambda _module, inputs: observed_self_attention_lengths.append(int(inputs[0].shape[1]))
    )
    try:
        logits = model(model_input)
    finally:
        handle.remove()

    sequence_token_count = sum(int(sequence.shape[2]) for sequence in model_input.seq_data.values())
    context_token_count = model.num_prompt_tokens + model.num_ns

    assert logits.shape == (2, 1)
    assert observed_self_attention_lengths == [context_token_count]
    assert context_token_count < context_token_count + sequence_token_count


def test_symbiosis_ablation_flags_disable_optional_modules() -> None:
    experiment_case = get_experiment_case("config/symbiosis")
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
        },
    )
    model_input = _sample_model_input(model_module)

    assert len(model.graph_blocks) == 0
    assert len(model.context_blocks) == 0
    assert model.symbiosis_use_fourier_time is False
    assert model.symbiosis_use_multi_scale is False
    assert all(block.use_domain_gate for block in model.unified_blocks)

    logits = model(model_input)
    predicted_logits, embeddings = model.predict(model_input)

    assert logits.shape == (2, 1)
    assert predicted_logits.shape == (2, 1)
    assert embeddings.shape == (2, model.d_model)
    assert torch.isfinite(logits).all()
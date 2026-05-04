from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from taac2026.infrastructure.pcvr.protocol import (
    batch_to_model_input,
    build_feature_specs,
    load_ns_groups,
    parse_seq_max_lens,
    resolve_schema_path,
)


class _ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict[str, torch.Tensor]
    seq_lens: dict[str, torch.Tensor]
    seq_time_buckets: dict[str, torch.Tensor]


def test_parse_seq_max_lens_and_feature_specs() -> None:
    schema = SimpleNamespace(entries=[(10, 0, 1), (20, 1, 2)])

    assert parse_seq_max_lens("seq_a:4, seq_b:8") == {"seq_a": 4, "seq_b": 8}
    assert build_feature_specs(schema, [11, 21, 22]) == [(11, 0, 1), (22, 1, 2)]


def test_resolve_schema_path_prefers_explicit_path(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "ckpt" / "global_step1"
    explicit_schema = tmp_path / "explicit_schema.json"
    dataset_dir.mkdir()
    checkpoint_dir.mkdir(parents=True)
    explicit_schema.write_text("{}", encoding="utf-8")
    (dataset_dir / "schema.json").write_text("{}", encoding="utf-8")

    assert resolve_schema_path(dataset_dir, explicit_schema, checkpoint_dir) == explicit_schema.resolve()


def test_load_ns_groups_maps_feature_ids_to_schema_positions(tmp_path) -> None:
    dataset = type(
        "Dataset",
        (),
        {
            "user_int_schema": type("Schema", (), {"entries": [(10, 0, 1), (20, 1, 1)]})(),
            "item_int_schema": type("Schema", (), {"entries": [(7, 0, 1)]})(),
        },
    )()

    assert load_ns_groups(
        dataset,
        {
            "ns_grouping_strategy": "explicit",
            "user_ns_groups": {"u": [20, 10]},
            "item_ns_groups": {"i": [7]},
        },
        tmp_path,
        tmp_path,
    ) == ([[1, 0]], [[0]])


def test_load_ns_groups_supports_explicit_singleton_strategy(tmp_path) -> None:
    dataset = type(
        "Dataset",
        (),
        {
            "user_int_schema": type("Schema", (), {"entries": [(10, 0, 1), (20, 1, 1)]})(),
            "item_int_schema": type("Schema", (), {"entries": [(7, 0, 1)]})(),
        },
    )()

    assert load_ns_groups(
        dataset,
        {
            "ns_grouping_strategy": "singleton",
            "user_ns_groups": {},
            "item_ns_groups": {},
        },
        tmp_path,
        tmp_path,
    ) == ([[0], [1]], [[0]])


def test_load_ns_groups_requires_explicit_group_maps(tmp_path) -> None:
    dataset = type(
        "Dataset",
        (),
        {
            "user_int_schema": type("Schema", (), {"entries": [(10, 0, 1)]})(),
            "item_int_schema": type("Schema", (), {"entries": [(7, 0, 1)]})(),
        },
    )()

    with pytest.raises(KeyError, match="user_ns_groups"):
        load_ns_groups(
            dataset,
            {
                "ns_grouping_strategy": "explicit",
                "item_ns_groups": {"i": [7]},
            },
            tmp_path,
            tmp_path,
        )


def test_load_ns_groups_requires_explicit_config_key(tmp_path) -> None:
    dataset = type(
        "Dataset",
        (),
        {
            "user_int_schema": type("Schema", (), {"entries": [(10, 0, 1)]})(),
            "item_int_schema": type("Schema", (), {"entries": [(7, 0, 1)]})(),
        },
    )()

    with pytest.raises(KeyError, match="ns_grouping_strategy"):
        load_ns_groups(dataset, {}, tmp_path, tmp_path)


def test_batch_to_model_input_uses_zero_time_buckets_when_missing() -> None:
    batch = {
        "user_int_feats": torch.ones(2, 1, dtype=torch.long),
        "item_int_feats": torch.ones(2, 1, dtype=torch.long),
        "user_dense_feats": torch.ones(2, 1),
        "item_dense_feats": torch.zeros(2, 0),
        "_seq_domains": ["seq_a"],
        "seq_a": torch.ones(2, 1, 3, dtype=torch.long),
        "seq_a_len": torch.tensor([3, 2]),
    }

    model_input = batch_to_model_input(batch, _ModelInput, torch.device("cpu"))

    assert set(model_input.seq_data) == {"seq_a"}
    assert model_input.seq_time_buckets["seq_a"].shape == (2, 3)
    assert torch.equal(model_input.seq_time_buckets["seq_a"], torch.zeros(2, 3, dtype=torch.long))
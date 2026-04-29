from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from taac2026.infrastructure.io.json_utils import dumps
from taac2026.infrastructure.pcvr.config import (
    PCVRDataCacheConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRSequenceCropConfig,
)
from taac2026.infrastructure.pcvr.data import PCVRParquetDataset
from taac2026.infrastructure.pcvr.data_pipeline import (
    PCVRDataPipeline,
    PCVRDomainDropoutTransform,
    PCVRFeatureMaskTransform,
    PCVRMemoryBatchCache,
    PCVRSequenceCropTransform,
    build_pcvr_batch_transforms,
)


def _make_batch() -> dict[str, object]:
    return {
        "user_int_feats": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "user_dense_feats": torch.tensor([[0.1], [0.2]], dtype=torch.float32),
        "item_int_feats": torch.tensor([[5], [6]], dtype=torch.long),
        "item_dense_feats": torch.zeros(2, 0, dtype=torch.float32),
        "label": torch.tensor([1, 0], dtype=torch.long),
        "timestamp": torch.tensor([100, 200], dtype=torch.long),
        "user_id": ["u0", "u1"],
        "_seq_domains": ["seq_a"],
        "seq_a": torch.tensor(
            [
                [[1, 2, 3, 4]],
                [[7, 8, 9, 0]],
            ],
            dtype=torch.long,
        ),
        "seq_a_len": torch.tensor([4, 3], dtype=torch.long),
        "seq_a_time_bucket": torch.tensor(
            [
                [4, 3, 2, 1],
                [3, 2, 1, 0],
            ],
            dtype=torch.long,
        ),
    }


def _assert_tensor_dict_equal(
    left: dict[str, object], right: dict[str, object]
) -> None:
    assert left.keys() == right.keys()
    for key, left_value in left.items():
        right_value = right[key]
        if isinstance(left_value, torch.Tensor):
            assert isinstance(right_value, torch.Tensor)
            assert torch.equal(left_value, right_value), key
        else:
            assert left_value == right_value


def test_empty_pipeline_config_builds_no_transforms() -> None:
    config = PCVRDataPipelineConfig()

    assert config.transform_names == ()
    assert build_pcvr_batch_transforms(config) == ()


def test_disabled_transform_preserves_batch_content() -> None:
    batch = _make_batch()
    transform = PCVRSequenceCropTransform(PCVRSequenceCropConfig(enabled=False))

    augmented = transform(batch, generator=torch.Generator().manual_seed(1))

    _assert_tensor_dict_equal(augmented, batch)
    assert augmented is not batch


def test_sequence_crop_expands_rows_and_keeps_metadata_aligned() -> None:
    batch = _make_batch()
    transform = PCVRSequenceCropTransform(
        PCVRSequenceCropConfig(
            views_per_row=2,
            seq_window_mode="random_tail",
            seq_window_min_len=2,
        )
    )

    augmented = transform(batch, generator=torch.Generator().manual_seed(7))

    assert augmented["label"].tolist() == [1, 1, 0, 0]
    assert augmented["timestamp"].tolist() == [100, 100, 200, 200]
    assert augmented["user_id"] == ["u0", "u0", "u1", "u1"]
    assert augmented["user_int_feats"].shape == (4, 2)
    assert augmented["seq_a"].shape == (4, 1, 4)
    assert augmented["seq_a_len"].min().item() >= 2
    assert augmented["seq_a_len"].max().item() <= 4
    for row_index, length in enumerate(augmented["seq_a_len"].tolist()):
        assert torch.equal(
            augmented["seq_a"][row_index, :, length:],
            torch.zeros(1, 4 - length, dtype=torch.long),
        )
        assert torch.equal(
            augmented["seq_a_time_bucket"][row_index, length:],
            torch.zeros(4 - length, dtype=torch.long),
        )


def test_domain_dropout_clears_sequence_tokens_lengths_and_time_buckets() -> None:
    batch = _make_batch()
    transform = PCVRDomainDropoutTransform(PCVRDomainDropoutConfig(probability=1.0))

    augmented = transform(batch, generator=torch.Generator().manual_seed(3))

    assert torch.equal(augmented["seq_a"], torch.zeros_like(augmented["seq_a"]))
    assert torch.equal(augmented["seq_a_len"], torch.zeros_like(augmented["seq_a_len"]))
    assert torch.equal(
        augmented["seq_a_time_bucket"], torch.zeros_like(augmented["seq_a_time_bucket"])
    )


def test_feature_masking_compacts_sequence_lengths() -> None:
    batch = _make_batch()
    transform = PCVRFeatureMaskTransform(PCVRFeatureMaskConfig(probability=1.0))

    augmented = transform(batch, generator=torch.Generator().manual_seed(5))

    assert torch.equal(
        augmented["user_int_feats"], torch.zeros_like(augmented["user_int_feats"])
    )
    assert torch.equal(
        augmented["item_int_feats"], torch.zeros_like(augmented["item_int_feats"])
    )
    assert torch.equal(augmented["seq_a"], torch.zeros_like(augmented["seq_a"]))
    assert torch.equal(augmented["seq_a_len"], torch.zeros_like(augmented["seq_a_len"]))
    assert torch.equal(
        augmented["seq_a_time_bucket"], torch.zeros_like(augmented["seq_a_time_bucket"])
    )


def test_augmentation_is_reproducible_with_fixed_generator_seed() -> None:
    batch = _make_batch()
    pipeline_config = PCVRDataPipelineConfig(
        transforms=(
            PCVRSequenceCropConfig(
                views_per_row=2,
                seq_window_mode="rolling",
                seq_window_min_len=1,
            ),
            PCVRFeatureMaskConfig(probability=0.3),
            PCVRDomainDropoutConfig(probability=0.2),
        ),
    )
    pipeline = PCVRDataPipeline(transforms=build_pcvr_batch_transforms(pipeline_config))

    first = pipeline.apply_transforms(
        batch, generator=torch.Generator().manual_seed(99)
    )
    second = pipeline.apply_transforms(
        batch, generator=torch.Generator().manual_seed(99)
    )

    _assert_tensor_dict_equal(first, second)


def test_memory_batch_cache_returns_isolated_clones() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="memory", max_batches=1)
    )
    cache.put(("file", 0, 0), _make_batch())

    cached = cache.get(("file", 0, 0))
    assert cached is not None
    cached["user_int_feats"][0, 0] = 999

    cached_again = cache.get(("file", 0, 0))
    assert cached_again is not None
    assert cached_again["user_int_feats"][0, 0].item() == 1


def test_strict_time_filter_removes_future_sequence_events(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    schema = {
        "user_int": [[1, 10, 1]],
        "item_int": [[2, 10, 1]],
        "user_dense": [[3, 2]],
        "seq": {
            "seq_a": {
                "prefix": "domain_a_seq",
                "ts_fid": 10,
                "features": [[10, 1000], [11, 100]],
            }
        },
    }
    schema_path.write_text(dumps(schema), encoding="utf-8")
    table = pa.table(
        {
            "timestamp": [100],
            "label_type": [2],
            "user_id": ["u0"],
            "user_int_feats_1": [1],
            "item_int_feats_2": [2],
            "user_dense_feats_3": [[0.1, 0.2]],
            "domain_a_seq_10": [[10, 150, 90]],
            "domain_a_seq_11": [[1, 2, 3]],
        }
    )
    pq.write_table(table, parquet_path, row_group_size=1)
    dataset = PCVRParquetDataset(
        parquet_path=str(parquet_path),
        schema_path=str(schema_path),
        batch_size=1,
        seq_max_lens={"seq_a": 3},
        shuffle=False,
        buffer_batches=0,
        data_pipeline_config=PCVRDataPipelineConfig(
            transforms=(PCVRSequenceCropConfig(),),
            strict_time_filter=True,
        ),
    )

    batch = next(iter(dataset))

    assert batch["seq_a_len"].tolist() == [2]
    assert batch["seq_a"].tolist() == [[[1, 3, 0]]]
    assert batch["seq_a_time_bucket"][0, :2].gt(0).all()
    assert batch["seq_a_time_bucket"][0, 2].item() == 0

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from taac2026.infrastructure.io.json import dumps
from taac2026.domain.config import (
    PCVRDataCacheConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRNonSequentialSparseDropoutConfig,
    PCVRSequenceCropConfig,
)
from taac2026.infrastructure.data import cache as cache_module
from taac2026.infrastructure.data.dataset import PCVRParquetDataset
from taac2026.infrastructure.data.pipeline import (
    PCVRDataPipeline,
    PCVRDomainDropoutTransform,
    PCVRFeatureMaskTransform,
    PCVRMemoryBatchCache,
    PCVRNonSequentialSparseDropoutTransform,
    PCVRSharedBatchCache,
    PCVRSharedTensorSpec,
    PCVRSequenceCropTransform,
    build_pcvr_batch_transforms,
    concat_pcvr_batches,
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
    batch["user_int_missing_mask"] = torch.zeros_like(batch["user_int_feats"], dtype=torch.bool)
    batch["item_int_missing_mask"] = torch.zeros_like(batch["item_int_feats"], dtype=torch.bool)
    batch["seq_a_stats"] = torch.ones(2, 6, dtype=torch.float32)
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
    assert torch.equal(
        augmented["user_int_missing_mask"], torch.ones_like(augmented["user_int_missing_mask"])
    )
    assert torch.equal(
        augmented["item_int_missing_mask"], torch.ones_like(augmented["item_int_missing_mask"])
    )
    assert torch.equal(augmented["seq_a_stats"], torch.zeros_like(augmented["seq_a_stats"]))


def test_nonseq_sparse_dropout_masks_full_rows_without_touching_sequences() -> None:
    batch = _make_batch()
    batch["user_int_missing_mask"] = torch.zeros_like(batch["user_int_feats"], dtype=torch.bool)
    batch["item_int_missing_mask"] = torch.zeros_like(batch["item_int_feats"], dtype=torch.bool)
    original_sequence = batch["seq_a"].clone()
    original_lengths = batch["seq_a_len"].clone()
    original_dense = batch["user_dense_feats"].clone()
    transform = PCVRNonSequentialSparseDropoutTransform(PCVRNonSequentialSparseDropoutConfig(probability=1.0))

    augmented = transform(batch, generator=torch.Generator().manual_seed(11))

    assert torch.equal(augmented["user_int_feats"], torch.zeros_like(augmented["user_int_feats"]))
    assert torch.equal(augmented["item_int_feats"], torch.zeros_like(augmented["item_int_feats"]))
    assert torch.equal(augmented["user_int_missing_mask"], torch.ones_like(augmented["user_int_missing_mask"]))
    assert torch.equal(augmented["item_int_missing_mask"], torch.ones_like(augmented["item_int_missing_mask"]))
    assert torch.equal(augmented["seq_a"], original_sequence)
    assert torch.equal(augmented["seq_a_len"], original_lengths)
    assert torch.equal(augmented["user_dense_feats"], original_dense)


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


def test_lru_batch_cache_returns_isolated_clones() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="lru", max_batches=1)
    )
    cache.put(("file", 0, 0), _make_batch())

    cached = cache.get(("file", 0, 0))
    assert cached is not None
    cached["user_int_feats"][0, 0] = 999

    cached_again = cache.get(("file", 0, 0))
    assert cached_again is not None
    assert cached_again["user_int_feats"][0, 0].item() == 1


def test_fifo_batch_cache_uses_configured_eviction_policy() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="fifo", max_batches=2)
    )

    cache.put(("file", 0, 0), _make_batch())
    cache.put(("file", 0, 1), _make_batch())
    assert cache.get(("file", 0, 0)) is not None
    cache.put(("file", 0, 2), _make_batch())

    assert cache.get(("file", 0, 0)) is None
    assert cache.get(("file", 0, 1)) is not None
    assert cache.get(("file", 0, 2)) is not None


def test_lfu_batch_cache_uses_configured_eviction_policy() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="lfu", max_batches=2)
    )
    cache.configure_key_universe(
        [("file", 0, 0), ("file", 0, 1), ("file", 0, 2)]
    )

    cache.put(("file", 0, 0), _make_batch())
    cache.put(("file", 0, 1), _make_batch())
    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 0)) is not None
    cache.put(("file", 0, 2), _make_batch())

    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is None
    assert cache.get(("file", 0, 2)) is not None
    stats = cache.stats()
    assert stats["native_cache_active"] is True
    assert stats["effective_policy"] == "lfu"


def test_rr_batch_cache_uses_native_index() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="rr", max_batches=2)
    )
    cache.configure_key_universe(
        [("file", 0, 0), ("file", 0, 1), ("file", 0, 2)]
    )

    cache.put(("file", 0, 0), _make_batch())
    cache.put(("file", 0, 1), _make_batch())
    cache.put(("file", 0, 2), _make_batch())

    assert len(cache) == 2
    stats = cache.stats()
    assert stats["native_cache_active"] is True
    assert stats["effective_policy"] == "rr"


def test_data_pipeline_keeps_explicit_empty_cache_instance() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )

    pipeline = PCVRDataPipeline(cache=cache)

    assert pipeline.cache is cache
    assert pipeline.cache._opt_enabled is True


def test_data_pipeline_materialize_composes_cache_preprocess_and_stages() -> None:
    events: list[str] = []

    class MarkStage:
        name = "mark"

        def __call__(self, batch, *, generator):
            del generator
            events.append("stage")
            batch = dict(batch)
            batch["marked"] = True
            return batch

    def factory():
        events.append("factory")
        return _make_batch()

    def preprocess(batch):
        events.append("preprocess")
        return batch

    pipeline = PCVRDataPipeline(cache=PCVRMemoryBatchCache(enabled=True, max_batches=1), stages=(MarkStage(),))

    first = pipeline.materialize(("file", 0, 0), factory, preprocess=preprocess)
    second = pipeline.materialize(("file", 0, 0), factory, preprocess=preprocess)

    assert first is not None and first["marked"] is True
    assert second is not None and second["marked"] is True
    assert events == ["factory", "preprocess", "stage", "preprocess", "stage"]


def test_concat_batch_drops_optional_metadata_missing_from_cached_batches() -> None:
    batch_a = _make_batch()
    batch_b = _make_batch()
    del batch_b["user_id"]

    merged = concat_pcvr_batches([batch_a, batch_b])

    assert "user_id" not in merged
    assert merged["label"].tolist() == [1, 0, 1, 0]


def test_opt_batch_cache_evicts_farthest_future_key() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )
    cache.configure_access_trace(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 2),
            ("file", 0, 3),
        ]
    )

    for key in (("file", 0, 0), ("file", 0, 1), ("file", 0, 2)):
        assert cache.get(key) is None
        cache.put(key, _make_batch())

    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is not None
    assert cache.get(("file", 0, 2)) is None
    stats = cache.stats()
    assert stats["opt_active"] is True
    assert stats["native_cache_active"] is True
    assert stats["native_opt_active"] is True
    assert stats["trace_length"] == 4


def test_opt_batch_cache_supports_repeated_step_trace() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )
    trace = [
        ("file", 0, 0),
        ("file", 0, 1),
        ("file", 0, 0),
        ("file", 0, 2),
        ("file", 0, 0),
    ]
    cache.configure_access_trace(trace, cyclic=False)

    assert cache.get(("file", 0, 0)) is None
    cache.put(("file", 0, 0), _make_batch())
    assert cache.get(("file", 0, 1)) is None
    cache.put(("file", 0, 1), _make_batch())
    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 2)) is None
    cache.put(("file", 0, 2), _make_batch())

    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is not None
    assert cache.get(("file", 0, 2)) is None
    stats = cache.stats()
    assert stats["opt_active"] is True
    assert stats["native_cache_active"] is True
    assert stats["native_opt_active"] is True
    assert stats["trace_length"] == 5


def test_opt_batch_cache_skips_clone_for_rejected_candidate(monkeypatch) -> None:
    clone_calls = 0
    original_clone = cache_module.clone_pcvr_batch

    def counted_clone(batch):
        nonlocal clone_calls
        clone_calls += 1
        return original_clone(batch)

    monkeypatch.setattr(cache_module, "clone_pcvr_batch", counted_clone)
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )
    cache.configure_access_trace(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 2),
            ("file", 0, 3),
        ]
    )

    for key in (("file", 0, 0), ("file", 0, 1)):
        assert cache.get(key) is None
        cache.put(key, _make_batch())

    assert clone_calls == 2
    assert cache.get(("file", 0, 2)) is None
    cache.put(("file", 0, 2), _make_batch())

    assert clone_calls == 2
    assert len(cache) == 2


def test_opt_batch_cache_rejects_candidate_without_future_use() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )
    cache.configure_access_trace(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 2),
            ("file", 0, 3),
        ]
    )

    for key in (("file", 0, 0), ("file", 0, 1)):
        assert cache.get(key) is None
        cache.put(key, _make_batch())

    assert cache.get(("file", 0, 2)) is None
    cache.put(("file", 0, 2), _make_batch())

    assert len(cache) == 2
    assert cache.get(("file", 0, 2)) is None


def test_opt_batch_cache_reconfigures_same_trace() -> None:
    trace = [
        ("file", 0, 0),
        ("file", 0, 1),
        ("file", 0, 2),
    ]
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )
    cache.configure_access_trace(trace)

    assert cache.get(("file", 0, 0)) is None
    cache.put(("file", 0, 0), _make_batch())

    cache.configure_access_trace(trace)

    assert cache.get(("file", 0, 0)) is not None
    assert cache._access_count == 1


def test_shared_opt_batch_cache_evicts_farthest_future_key() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="opt",
        tensor_specs={
            "label": PCVRSharedTensorSpec(shape=(2,), dtype=torch.long),
        },
        static_values={"_seq_domains": []},
    )
    cache.configure_access_trace(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 2),
            ("file", 0, 3),
        ]
    )

    for key, value in zip(
        (("file", 0, 0), ("file", 0, 1), ("file", 0, 2)),
        (1, 2, 3),
        strict=True,
    ):
        assert cache.get(key) is None
        cache.put(key, {"label": torch.tensor([value, value + 10], dtype=torch.long)})

    cached_0 = cache.get(("file", 0, 0))
    cached_1 = cache.get(("file", 0, 1))
    cached_2 = cache.get(("file", 0, 2))

    assert cached_0 is not None
    assert cached_0["label"].tolist() == [1, 11]
    assert cached_1 is not None
    assert cached_1["label"].tolist() == [2, 12]
    assert cached_2 is None


def test_shared_opt_batch_cache_supports_repeated_step_trace() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="opt",
        tensor_specs={
            "label": PCVRSharedTensorSpec(shape=(2,), dtype=torch.long),
        },
        static_values={"_seq_domains": []},
    )
    cache.configure_access_trace(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 0),
            ("file", 0, 2),
            ("file", 0, 0),
        ],
        cyclic=False,
        key_universe=[("file", 0, 0), ("file", 0, 1), ("file", 0, 2)],
    )

    assert cache.get(("file", 0, 0)) is None
    cache.put(("file", 0, 0), {"label": torch.tensor([1, 11], dtype=torch.long)})
    assert cache.get(("file", 0, 1)) is None
    cache.put(("file", 0, 1), {"label": torch.tensor([2, 12], dtype=torch.long)})
    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 2)) is None
    cache.put(("file", 0, 2), {"label": torch.tensor([3, 13], dtype=torch.long)})

    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is not None
    assert cache.get(("file", 0, 2)) is None
    stats = cache.stats()
    assert stats["native_opt_active"] is True
    assert stats["trace_length"] == 5


def test_shared_opt_batch_cache_requires_trace_keys() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="opt",
        tensor_specs={
            "label": PCVRSharedTensorSpec(shape=(2,), dtype=torch.long),
        },
        static_values={"_seq_domains": []},
    )
    cache.configure_access_trace([("file", 0, 0)])

    assert cache.get(("file", 0, 0)) is None
    try:
        cache.get(("file", 0, 1))
    except KeyError as exc:
        assert "missing from configured access trace" in str(exc)
    else:
        raise AssertionError("expected KeyError for an untraced OPT cache key")


def test_shared_lru_batch_cache_reuses_slot_for_existing_key() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=1,
        policy="lru",
        tensor_specs={
            "label": PCVRSharedTensorSpec(shape=(2,), dtype=torch.long),
        },
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe([("file", 0, 0)])

    cache.put(("file", 0, 0), {"label": torch.tensor([1, 11], dtype=torch.long)})
    cache.put(("file", 0, 0), {"label": torch.tensor([2, 12], dtype=torch.long)})

    cached = cache.get(("file", 0, 0))

    assert len(cache) == 1
    assert cached is not None
    assert cached["label"].tolist() == [2, 12]


def test_shared_batch_cache_treats_busy_slot_as_miss() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=1,
        policy="lru",
        tensor_specs={
            "label": PCVRSharedTensorSpec(shape=(2,), dtype=torch.long),
        },
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe([("file", 0, 0)])
    cache.put(("file", 0, 0), {"label": torch.tensor([1, 11], dtype=torch.long)})

    cache._slot_versions[0] += 1
    cached = cache.get(("file", 0, 0))

    assert cached is None
    stats = cache.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1


def test_shared_batch_cache_discards_payload_when_version_changes(monkeypatch) -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=1,
        policy="lru",
        tensor_specs={
            "label": PCVRSharedTensorSpec(shape=(2,), dtype=torch.long),
        },
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe([("file", 0, 0)])
    cache.put(("file", 0, 0), {"label": torch.tensor([1, 11], dtype=torch.long)})
    original_materialize = cache._materialize_slot

    def materialize_and_invalidate(slot_index: int, row_count: int):
        batch = original_materialize(slot_index, row_count)
        cache._slot_versions[slot_index] += 2
        return batch

    monkeypatch.setattr(cache, "_materialize_slot", materialize_and_invalidate)

    cached = cache.get(("file", 0, 0))

    assert cached is None
    stats = cache.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1


def test_shared_fifo_batch_cache_uses_native_index() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="fifo",
        tensor_specs={
            "label": PCVRSharedTensorSpec(shape=(2,), dtype=torch.long),
        },
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe(
        [("file", 0, 0), ("file", 0, 1), ("file", 0, 2)]
    )

    cache.put(("file", 0, 0), {"label": torch.tensor([1, 11], dtype=torch.long)})
    cache.put(("file", 0, 1), {"label": torch.tensor([2, 12], dtype=torch.long)})
    assert cache.get(("file", 0, 0)) is not None
    cache.put(("file", 0, 2), {"label": torch.tensor([3, 13], dtype=torch.long)})

    assert cache.get(("file", 0, 0)) is None
    assert cache.get(("file", 0, 1)) is not None
    assert cache.get(("file", 0, 2)) is not None
    assert cache.stats()["native_cache_active"] is True


def test_shared_lfu_batch_cache_uses_native_index() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="lfu",
        tensor_specs={
            "label": PCVRSharedTensorSpec(shape=(2,), dtype=torch.long),
        },
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe(
        [("file", 0, 0), ("file", 0, 1), ("file", 0, 2)]
    )

    cache.put(("file", 0, 0), {"label": torch.tensor([1, 11], dtype=torch.long)})
    cache.put(("file", 0, 1), {"label": torch.tensor([2, 12], dtype=torch.long)})
    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 0)) is not None
    cache.put(("file", 0, 2), {"label": torch.tensor([3, 13], dtype=torch.long)})

    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is None
    assert cache.get(("file", 0, 2)) is not None
    assert cache.stats()["native_cache_active"] is True


def test_shared_rr_batch_cache_uses_native_index() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="rr",
        tensor_specs={
            "label": PCVRSharedTensorSpec(shape=(2,), dtype=torch.long),
        },
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe(
        [("file", 0, 0), ("file", 0, 1), ("file", 0, 2)]
    )

    cache.put(("file", 0, 0), {"label": torch.tensor([1, 11], dtype=torch.long)})
    cache.put(("file", 0, 1), {"label": torch.tensor([2, 12], dtype=torch.long)})
    cache.put(("file", 0, 2), {"label": torch.tensor([3, 13], dtype=torch.long)})

    assert len(cache) == 2
    assert cache.stats()["native_cache_active"] is True


def test_shared_lru_batch_cache_uses_key_universe_and_tracks_hits() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="lru",
        tensor_specs={
            "label": PCVRSharedTensorSpec(shape=(2,), dtype=torch.long),
        },
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 2),
        ]
    )

    assert cache.get(("file", 0, 0)) is None
    cache.put(("file", 0, 0), {"label": torch.tensor([1, 11], dtype=torch.long)})
    cache.put(("file", 0, 1), {"label": torch.tensor([2, 12], dtype=torch.long)})

    cached = cache.get(("file", 0, 0))
    assert cached is not None
    assert cached["label"].tolist() == [1, 11]

    cache.put(("file", 0, 2), {"label": torch.tensor([3, 13], dtype=torch.long)})

    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is None
    assert cache.get(("file", 0, 2)) is not None

    stats = cache.stats()
    assert stats["shared_lru_active"] is True
    assert stats["hits"] == 3
    assert stats["misses"] == 2
    assert stats["hit_rate"] == 0.6


def test_shared_batch_cache_overwrites_partial_slot_without_stale_rows() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=1,
        policy="lru",
        tensor_specs={
            "label": PCVRSharedTensorSpec(shape=(2,), dtype=torch.long),
        },
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe([("file", 0, 0)])

    cache.put(("file", 0, 0), {"label": torch.tensor([1, 11], dtype=torch.long)})
    cache.put(("file", 0, 0), {"label": torch.tensor([2], dtype=torch.long)})

    cached = cache.get(("file", 0, 0))

    assert cached is not None
    assert cached["label"].tolist() == [2]


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


def test_dataset_logs_schema_payload_with_dataset_role(tmp_path: Path, log_capture) -> None:
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
            "domain_a_seq_10": [[10]],
            "domain_a_seq_11": [[1]],
        }
    )
    pq.write_table(table, parquet_path, row_group_size=1)

    with log_capture.at_level(logging.INFO):
        PCVRParquetDataset(
            parquet_path=str(parquet_path),
            schema_path=str(schema_path),
            batch_size=1,
            seq_max_lens={"seq_a": 1},
            shuffle=False,
            buffer_batches=0,
            dataset_role="train",
        )

    assert "Loaded PCVR schema for train dataset" in log_capture.text
    assert str(schema_path.resolve()) in log_capture.text
    assert "PCVR train schema payload" in log_capture.text
    payload_message = next(
        record.getMessage()
        for record in log_capture.records
        if record.getMessage().startswith("PCVR train schema payload: ")
    )
    assert "\n" not in payload_message
    assert '"user_int":[[1,10,1]]' in payload_message
    assert '"prefix":"domain_a_seq"' in payload_message

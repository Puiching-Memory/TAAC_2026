from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, IterableDataset

import taac2026.infrastructure.data.dataset as pcvr_data
import taac2026.infrastructure.data.observation as pcvr_observation
from taac2026.infrastructure.io.json import dumps
from taac2026.domain.config import (
    PCVRDataCacheConfig,
    PCVRDataPipelineConfig,
    PCVRSequenceCropConfig,
)
from taac2026.infrastructure.data.step_dataset import (
    PCVRStepDataset,
    PCVRStepIndexSampler,
)
from taac2026.infrastructure.data.batch_converter import pad_list_offsets_values


class _FakeDataset(IterableDataset):
    def __init__(
        self,
        *_args,
        row_group_range: tuple[int, int] | None = None,
        timestamp_range: tuple[int | None, int | None] | None = None,
        data_pipeline_config: PCVRDataPipelineConfig | None = None,
        dataset_role: str = "dataset",
        **_kwargs,
    ) -> None:
        self.row_group_range = row_group_range
        self.timestamp_range = timestamp_range
        self.data_pipeline_config = data_pipeline_config
        self.dataset_role = dataset_role
        self.num_rows = 0

    def __iter__(self):
        return iter(())


class _FakeRowGroup:
    def __init__(self, num_rows: int) -> None:
        self.num_rows = num_rows


class _FakeMetadata:
    def __init__(self, row_group_rows: list[int]) -> None:
        self._row_group_rows = row_group_rows
        self.num_row_groups = len(row_group_rows)

    def row_group(self, index: int) -> _FakeRowGroup:
        return _FakeRowGroup(self._row_group_rows[index])


class _FakeParquetFile:
    def __init__(self, _path: str, row_group_rows: list[int]) -> None:
        self.metadata = _FakeMetadata(row_group_rows)


def _patch_parquet_runtime(monkeypatch, row_group_rows: list[int]) -> None:
    monkeypatch.setattr(pcvr_data, "PCVRParquetDataset", _FakeDataset)
    monkeypatch.setattr(
        pcvr_observation.pq,
        "ParquetFile",
        lambda path: _FakeParquetFile(path, row_group_rows),
    )


def _write_observed_schema_fixture(schema_path: Path, parquet_path: Path) -> None:
    schema = {
        "user_int": [[1, 10, 1], [2, 20, 4]],
        "item_int": [[3, 10, 1]],
        "user_dense": [[4, 4]],
        "seq": {
            "seq_a": {
                "prefix": "domain_a_seq",
                "ts_fid": 10,
                "features": [[10, 0], [11, 20]],
            }
        },
    }
    schema_path.write_text(dumps(schema), encoding="utf-8")
    pq.write_table(
        pa.table(
            {
                "user_int_feats_1": [1, 2],
                "user_int_feats_2": [[1, 2], [2, 3, 4]],
                "item_int_feats_3": [3, 4],
                "user_dense_feats_4": [[0.1, 0.2, 0.3], [0.9]],
                "domain_a_seq_10": [[100, 101], [103]],
                "domain_a_seq_11": [[5, 6], [6, 7, 7]],
            }
        ),
        parquet_path,
        row_group_size=1,
    )


def _write_single_row_group_multi_batch_fixture(schema_path: Path, parquet_path: Path) -> None:
    schema = {
        "user_int": [[1, 10, 1], [2, 20, 2]],
        "item_int": [[3, 10, 1]],
        "user_dense": [[4, 2]],
        "seq": {},
    }
    schema_path.write_text(dumps(schema), encoding="utf-8")
    pq.write_table(
        pa.table(
            {
                "timestamp": [100, 101, 102, 103],
                "label_type": [2, 0, 2, 0],
                "user_id": ["u0", "u1", "u2", "u3"],
                "user_int_feats_1": [1, 2, 3, 4],
                "user_int_feats_2": [[1, 2], [2, 3], [3, 4], [4, 5]],
                "item_int_feats_3": [10, 11, 12, 13],
                "user_dense_feats_4": [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
            }
        ),
        parquet_path,
        row_group_size=4,
    )


def _write_multi_row_group_partial_batch_fixture(schema_path: Path, parquet_path: Path) -> None:
    schema = {
        "user_int": [[1, 10, 1], [2, 20, 2]],
        "item_int": [[3, 10, 1]],
        "user_dense": [[4, 2]],
        "seq": {},
    }
    schema_path.write_text(dumps(schema), encoding="utf-8")
    pq.write_table(
        pa.table(
            {
                "timestamp": [100, 101, 102, 103, 104],
                "label_type": [2, 0, 2, 0, 2],
                "user_id": ["u0", "u1", "u2", "u3", "u4"],
                "user_int_feats_1": [1, 2, 3, 4, 5],
                "user_int_feats_2": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
                "item_int_feats_3": [10, 11, 12, 13, 14],
                "user_dense_feats_4": [
                    [0.1, 0.2],
                    [0.2, 0.3],
                    [0.3, 0.4],
                    [0.4, 0.5],
                    [0.5, 0.6],
                ],
            }
        ),
        parquet_path,
        row_group_size=2,
    )


def test_pad_list_offsets_values_handles_empty_and_truncated_rows() -> None:
    values = pa.array([[1, 2, 3], [], [-1, 4], [5]], type=pa.list_(pa.int64()))

    padded, lengths = pad_list_offsets_values(
        values.offsets.to_numpy(),
        values.values.to_numpy(),
        row_count=4,
        width=2,
        dtype=np.int64,
    )

    assert lengths.tolist() == [2, 0, 2, 1]
    assert padded.tolist() == [[1, 2], [0, 0], [-1, 4], [5, 0]]


def test_get_pcvr_data_reuses_single_row_group_for_validation(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_parquet_runtime(monkeypatch, [1000])
    parquet_path = tmp_path / "demo.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")

    train_loader, valid_loader, train_dataset = pcvr_data.get_pcvr_data(
        data_dir=str(parquet_path),
        schema_path=str(tmp_path / "schema.json"),
        batch_size=8,
        num_workers=0,
        buffer_batches=1,
    )

    assert train_dataset.row_group_range == (0, 1)
    assert train_dataset.dataset_role == "train"
    assert train_loader.dataset.row_group_range == (0, 1)
    assert train_loader.dataset.dataset_role == "train"
    assert valid_loader.dataset.row_group_range == (0, 1)
    assert valid_loader.dataset.dataset_role == "valid"

    split_plan = pcvr_data.plan_pcvr_row_group_split([("demo.parquet", 0, 1000)])
    assert split_plan.reuse_train_for_valid is True
    assert split_plan.is_l1_ready is False


def test_get_pcvr_data_keeps_disjoint_ranges_when_multiple_row_groups(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_parquet_runtime(monkeypatch, [400, 300, 300])
    parquet_path = tmp_path / "demo.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")

    train_loader, valid_loader, train_dataset = pcvr_data.get_pcvr_data(
        data_dir=str(parquet_path),
        schema_path=str(tmp_path / "schema.json"),
        batch_size=8,
        valid_ratio=0.34,
        num_workers=0,
        buffer_batches=1,
    )

    assert train_dataset.row_group_range == (0, 2)
    assert train_dataset.dataset_role == "train"
    assert train_loader.dataset.row_group_range == (0, 2)
    assert train_loader.dataset.dataset_role == "train"
    assert valid_loader.dataset.row_group_range == (2, 3)
    assert valid_loader.dataset.dataset_role == "valid"

    split_plan = pcvr_data.plan_pcvr_row_group_split(
        [
            ("demo.parquet", 0, 400),
            ("demo.parquet", 1, 300),
            ("demo.parquet", 2, 300),
        ],
        valid_ratio=0.34,
    )
    assert split_plan.is_l1_ready is True
    assert split_plan.train_rows == 700
    assert split_plan.valid_rows == 300


def test_get_pcvr_data_supports_timestamp_range_split(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_parquet_runtime(monkeypatch, [400, 300, 300])
    monkeypatch.setattr(
        pcvr_data,
        "count_pcvr_rows_in_timestamp_range",
        lambda _row_groups, timestamp_range: {
            (None, 200): 700,
            (200, 400): 300,
        }[timestamp_range],
    )
    parquet_path = tmp_path / "demo.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")

    train_loader, valid_loader, train_dataset = pcvr_data.get_pcvr_data(
        data_dir=str(parquet_path),
        schema_path=str(tmp_path / "schema.json"),
        batch_size=8,
        split_strategy="timestamp_range",
        train_timestamp_end=200,
        valid_timestamp_start=200,
        valid_timestamp_end=400,
        num_workers=0,
        buffer_batches=1,
    )

    assert train_dataset.row_group_range == (0, 3)
    assert train_dataset.timestamp_range == (None, 200)
    assert train_loader.dataset.timestamp_range == (None, 200)
    assert valid_loader.dataset.row_group_range == (0, 3)
    assert valid_loader.dataset.timestamp_range == (200, 400)


def test_get_pcvr_data_applies_augmentation_only_to_train_dataset(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_parquet_runtime(monkeypatch, [400, 300, 300])
    parquet_path = tmp_path / "demo.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")
    data_pipeline_config = PCVRDataPipelineConfig(
        cache=PCVRDataCacheConfig(mode="lru", max_batches=4),
        transforms=(PCVRSequenceCropConfig(views_per_row=2),),
        seed=11,
    )

    train_loader, valid_loader, train_dataset = pcvr_data.get_pcvr_data(
        data_dir=str(parquet_path),
        schema_path=str(tmp_path / "schema.json"),
        batch_size=8,
        valid_ratio=0.34,
        num_workers=0,
        buffer_batches=1,
        data_pipeline_config=data_pipeline_config,
    )

    assert train_dataset.data_pipeline_config == data_pipeline_config
    assert train_loader.dataset.data_pipeline_config == data_pipeline_config
    assert valid_loader.dataset.data_pipeline_config is not None
    assert valid_loader.dataset.data_pipeline_config.transforms == ()
    assert train_loader.dataset.data_pipeline_config.cache == data_pipeline_config.cache
    assert valid_loader.dataset.data_pipeline_config.cache == data_pipeline_config.cache


def test_get_pcvr_data_uses_shared_opt_cache_for_step_random_training(
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    _write_observed_schema_fixture(schema_path, parquet_path)
    data_pipeline_config = PCVRDataPipelineConfig(
        cache=PCVRDataCacheConfig(mode="opt", max_batches=4),
    )

    train_loader, _valid_loader, train_dataset = pcvr_data.get_pcvr_data(
        data_dir=str(parquet_path),
        schema_path=str(schema_path),
        batch_size=1,
        valid_ratio=0.5,
        num_workers=2,
        buffer_batches=20,
        sampling_strategy="step_random",
        data_pipeline_config=data_pipeline_config,
        max_steps=12,
    )

    assert train_loader.dataset is train_dataset
    assert isinstance(train_dataset, PCVRStepDataset)
    assert train_dataset.uses_step_random_sampling is True
    assert train_dataset.buffer_batches == 1
    assert train_dataset.logical_sweep_steps() == 12
    assert _valid_loader.dataset.uses_step_random_sampling is False
    assert train_dataset.pipeline.cache.__class__.__name__ == "PCVRSharedBatchCache"
    assert getattr(train_dataset.pipeline.cache, "uses_global_access_trace", False) is True
    stats = train_dataset.pipeline.cache.stats()
    assert stats["opt_active"] is True
    assert stats["native_opt_active"] is True
    assert stats["trace_length"] == 12


def test_get_pcvr_data_uses_shared_native_cache_for_all_policies(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    _write_observed_schema_fixture(schema_path, parquet_path)

    for mode in ("lru", "fifo", "lfu", "rr"):
        data_pipeline_config = PCVRDataPipelineConfig(
            cache=PCVRDataCacheConfig(mode=mode, max_batches=4),
        )

        train_loader, _valid_loader, train_dataset = pcvr_data.get_pcvr_data(
            data_dir=str(parquet_path),
            schema_path=str(schema_path),
            batch_size=1,
            valid_ratio=0.5,
            num_workers=2,
            buffer_batches=1,
            data_pipeline_config=data_pipeline_config,
            max_steps=12,
        )

        assert train_loader.dataset is train_dataset
        assert train_dataset.pipeline.cache.__class__.__name__ == "PCVRSharedBatchCache"
        assert getattr(train_dataset.pipeline.cache, "uses_global_access_trace", True) is False
        stats = train_dataset.pipeline.cache.stats()
        assert stats["policy"] == mode
        assert stats["effective_policy"] == mode
        assert stats["native_cache_active"] is True
        assert stats["opt_active"] is False
        assert stats["trace_length"] == 0


def test_step_dataset_draws_batches_with_replacement(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    _write_single_row_group_multi_batch_fixture(schema_path, parquet_path)
    source_dataset = pcvr_data.PCVRParquetDataset(
        parquet_path=str(parquet_path),
        schema_path=str(schema_path),
        batch_size=1,
        shuffle=True,
        buffer_batches=0,
        data_pipeline_config=PCVRDataPipelineConfig(),
        is_training=True,
        dataset_role="train",
    )
    dataset = PCVRStepDataset(source_dataset, planned_steps=8, seed=42)

    draws = [dataset.plan_step(step_index).batch_key for step_index in range(8)]

    assert len(dataset) == 8
    assert len(draws) == 8
    assert len(set(draws)) < len(draws)


def test_step_dataset_keeps_one_optimizer_batch_after_multi_view_transform(
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    _write_single_row_group_multi_batch_fixture(schema_path, parquet_path)
    source_dataset = pcvr_data.PCVRParquetDataset(
        parquet_path=str(parquet_path),
        schema_path=str(schema_path),
        batch_size=2,
        shuffle=True,
        buffer_batches=0,
        data_pipeline_config=PCVRDataPipelineConfig(
            transforms=(PCVRSequenceCropConfig(views_per_row=2),),
            seed=7,
        ),
        is_training=True,
        dataset_role="train",
    )
    dataset = PCVRStepDataset(source_dataset, planned_steps=4, seed=42)

    batch = dataset[0]

    assert batch["label"].shape[0] == 2
    assert batch["user_int_feats"].shape[0] == 2


def test_get_pcvr_data_step_loader_materializes_optimizer_batches(
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    _write_single_row_group_multi_batch_fixture(schema_path, parquet_path)

    train_loader, _valid_loader, train_dataset = pcvr_data.get_pcvr_data(
        data_dir=str(parquet_path),
        schema_path=str(schema_path),
        batch_size=2,
        num_workers=0,
        buffer_batches=1,
        sampling_strategy="step_random",
        data_pipeline_config=PCVRDataPipelineConfig(
            transforms=(PCVRSequenceCropConfig(views_per_row=2),),
            seed=7,
        ),
        max_steps=3,
    )

    batch = next(iter(train_loader))

    assert isinstance(train_dataset, PCVRStepDataset)
    assert len(train_loader) == 3
    assert batch["label"].shape[0] == 2
    assert batch["user_int_feats"].shape[0] == 2


def test_step_index_sampler_offsets_indices_by_start_step() -> None:
    sampler = PCVRStepIndexSampler(step_count=3)

    assert list(sampler) == [0, 1, 2]

    sampler.set_start_step(6)

    assert list(sampler) == [6, 7, 8]


def test_step_dataset_train_steps_per_sweep_overrides_planned_steps_for_loader_length(
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    _write_single_row_group_multi_batch_fixture(schema_path, parquet_path)
    source_dataset = pcvr_data.PCVRParquetDataset(
        parquet_path=str(parquet_path),
        schema_path=str(schema_path),
        batch_size=1,
        shuffle=True,
        buffer_batches=1,
        data_pipeline_config=PCVRDataPipelineConfig(),
        is_training=True,
        dataset_role="train",
    )
    dataset = PCVRStepDataset(source_dataset, train_steps_per_sweep=2, planned_steps=5, seed=42)

    assert len(dataset) == 2
    assert dataset.logical_sweep_steps() == 2
    assert list(dataset.iter_step_batch_keys(steps=5))


def test_scan_dataset_does_not_use_step_random_sampling(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    _write_single_row_group_multi_batch_fixture(schema_path, parquet_path)
    dataset = pcvr_data.PCVRParquetDataset(
        parquet_path=str(parquet_path),
        schema_path=str(schema_path),
        batch_size=1,
        shuffle=True,
        buffer_batches=1,
        data_pipeline_config=PCVRDataPipelineConfig(),
        is_training=True,
        dataset_role="train",
    )

    assert dataset.uses_step_random_sampling is False
    assert dataset.logical_sweep_steps() == len(dataset)


def test_scan_dataset_len_counts_row_group_partial_batches(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    _write_multi_row_group_partial_batch_fixture(schema_path, parquet_path)
    dataset = pcvr_data.PCVRParquetDataset(
        parquet_path=str(parquet_path),
        schema_path=str(schema_path),
        batch_size=3,
        shuffle=False,
        buffer_batches=0,
        data_pipeline_config=PCVRDataPipelineConfig(),
        is_training=True,
        dataset_role="valid",
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    with warnings.catch_warnings(record=True) as caught:
        batches = list(loader)

    assert len(dataset) == 3
    assert len(loader) == 3
    assert len(batches) == 3
    assert not any("Length of IterableDataset" in str(warning.message) for warning in caught)


def test_scan_dataset_shared_opt_cache_uses_scan_trace(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    _write_single_row_group_multi_batch_fixture(schema_path, parquet_path)
    dataset = pcvr_data.PCVRParquetDataset(
        parquet_path=str(parquet_path),
        schema_path=str(schema_path),
        batch_size=1,
        shuffle=False,
        buffer_batches=0,
        data_pipeline_config=PCVRDataPipelineConfig(
            cache=PCVRDataCacheConfig(mode="opt", max_batches=4),
        ),
        is_training=True,
        dataset_role="train",
    )
    cache = dataset.build_shared_batch_cache(num_workers=2)

    stats = cache.stats()
    assert stats["policy"] == "opt"
    assert stats["opt_active"] is True
    assert stats["trace_length"] == len(dataset)


def test_build_pcvr_observed_schema_report_respects_row_group_range(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    _write_observed_schema_fixture(schema_path, parquet_path)

    train_report = pcvr_data.build_pcvr_observed_schema_report(
        parquet_path,
        schema_path,
        row_group_range=(0, 1),
        dataset_role="train_split",
    )
    valid_report = pcvr_data.build_pcvr_observed_schema_report(
        parquet_path,
        schema_path,
        row_group_range=(1, 2),
        dataset_role="valid_split",
    )

    assert train_report["dataset_role"] == "train_split"
    assert train_report["row_group_range"] == [0, 1]
    assert train_report["row_count"] == 1
    assert train_report["schema"]["user_int"] == [[1, 1, 1], [2, 2, 2]]
    assert train_report["schema"]["item_int"] == [[3, 1, 1]]
    assert train_report["schema"]["user_dense"] == [[4, 3]]
    assert train_report["schema"]["seq"]["seq_a"]["features"] == [[10, 101], [11, 2]]

    assert valid_report["dataset_role"] == "valid_split"
    assert valid_report["row_group_range"] == [1, 2]
    assert valid_report["row_count"] == 1
    assert valid_report["schema"]["user_int"] == [[1, 1, 1], [2, 3, 3]]
    assert valid_report["schema"]["item_int"] == [[3, 1, 1]]
    assert valid_report["schema"]["user_dense"] == [[4, 1]]
    assert valid_report["schema"]["seq"]["seq_a"]["features"] == [[10, 103], [11, 2]]

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset

import taac2026.infrastructure.data.dataset as pcvr_data
from taac2026.infrastructure.io.json import dumps
from taac2026.domain.config import (
    PCVRDataCacheConfig,
    PCVRDataPipelineConfig,
    PCVRSequenceCropConfig,
)


class _FakeDataset(IterableDataset):
    def __init__(
        self,
        *_args,
        row_group_range: tuple[int, int] | None = None,
        data_pipeline_config: PCVRDataPipelineConfig | None = None,
        dataset_role: str = "dataset",
        **_kwargs,
    ) -> None:
        self.row_group_range = row_group_range
        self.data_pipeline_config = data_pipeline_config
        self.dataset_role = dataset_role

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
        pcvr_data.pq,
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
                "user_int_feats_1": [1, 2, 3, 4],
                "user_int_feats_2": [[1, 2], [2, 3], [3, 4], [4, 5]],
                "item_int_feats_3": [10, 11, 12, 13],
                "user_dense_feats_4": [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
            }
        ),
        parquet_path,
        row_group_size=4,
    )


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


def test_get_pcvr_data_applies_augmentation_only_to_train_dataset(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_parquet_runtime(monkeypatch, [400, 300, 300])
    parquet_path = tmp_path / "demo.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")
    data_pipeline_config = PCVRDataPipelineConfig(
        cache=PCVRDataCacheConfig(mode="memory", max_batches=4),
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


def test_get_pcvr_data_uses_shared_opt_cache_for_multi_worker_training(
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
        buffer_batches=1,
        data_pipeline_config=data_pipeline_config,
    )

    assert train_loader.dataset is train_dataset
    assert train_dataset.pipeline.cache.__class__.__name__ == "PCVRSharedBatchCache"
    assert getattr(train_dataset.pipeline.cache, "uses_global_access_trace", False) is True


def test_global_batch_schedule_splits_single_row_group_by_batch(tmp_path: Path) -> None:
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
    dataset.configure_global_batch_schedule(num_workers=2, cyclic=False)

    worker0_keys = list(
        dataset._iter_worker_scheduled_batch_keys(
            worker_id=0,
            num_workers=2,
            cyclic=False,
        )
    )
    worker1_keys = list(
        dataset._iter_worker_scheduled_batch_keys(
            worker_id=1,
            num_workers=2,
            cyclic=False,
        )
    )

    assert [batch_position for batch_position, _batch_key in worker0_keys] == [0, 2]
    assert [batch_position for batch_position, _batch_key in worker1_keys] == [1, 3]
    assert [batch_key for _batch_position, batch_key in worker0_keys] == [
        (str(parquet_path), 0, 0),
        (str(parquet_path), 0, 2),
    ]
    assert [batch_key for _batch_position, batch_key in worker1_keys] == [
        (str(parquet_path), 0, 1),
        (str(parquet_path), 0, 3),
    ]


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

from __future__ import annotations

from pathlib import Path

from torch.utils.data import IterableDataset

import taac2026.infrastructure.pcvr.data as pcvr_data
from taac2026.infrastructure.pcvr.config import (
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
        **_kwargs,
    ) -> None:
        self.row_group_range = row_group_range
        self.data_pipeline_config = data_pipeline_config

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
    assert train_loader.dataset.row_group_range == (0, 1)
    assert valid_loader.dataset.row_group_range == (0, 1)

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
    assert train_loader.dataset.row_group_range == (0, 2)
    assert valid_loader.dataset.row_group_range == (2, 3)

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

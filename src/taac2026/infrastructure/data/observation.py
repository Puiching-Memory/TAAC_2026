"""Observed-schema and row-group split helpers for PCVR data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from taac2026.infrastructure.io.json import read_path
from taac2026.infrastructure.logging import logger


DEFAULT_PCVR_OBSERVED_SCHEMA_BATCH_SIZE = 1024
PCVRTimestampRange = tuple[int | None, int | None]


@dataclass(frozen=True, slots=True)
class PCVRRowGroupSplitPlan:
    total_row_groups: int
    train_row_groups: int
    valid_row_groups: int
    train_row_group_range: tuple[int, int]
    valid_row_group_range: tuple[int, int]
    train_rows: int
    valid_rows: int
    reuse_train_for_valid: bool

    @property
    def is_disjoint(self) -> bool:
        return (
            not self.reuse_train_for_valid
            and self.train_row_group_range[1] <= self.valid_row_group_range[0]
        )

    @property
    def is_l1_ready(self) -> bool:
        return (
            self.is_disjoint
            and self.train_rows > 0
            and self.valid_rows > 0
            and self.train_row_groups > 0
            and self.valid_row_groups > 0
        )


class _ExactPositiveCardinalityCounter:
    def __init__(self, max_value_hint: int) -> None:
        self.max_value_hint = max(0, int(max_value_hint))
        self._bitset = bytearray((self.max_value_hint // 8) + 1) if self.max_value_hint > 0 else bytearray()
        self._overflow_values: set[int] = set()
        self.count = 0

    def add_values(self, values: NDArray[np.int64]) -> None:
        if values.size == 0:
            return
        positive_values = values[values > 0]
        if positive_values.size == 0:
            return
        for raw_value in np.unique(positive_values).tolist():
            value = int(raw_value)
            if 0 < value <= self.max_value_hint:
                byte_index = value >> 3
                bit_mask = 1 << (value & 7)
                if self._bitset[byte_index] & bit_mask:
                    continue
                self._bitset[byte_index] |= bit_mask
                self.count += 1
                continue
            if value not in self._overflow_values:
                self._overflow_values.add(value)
                self.count += 1


def _is_list_like_arrow_type(data_type: pa.DataType) -> bool:
    return pa.types.is_list(data_type) or pa.types.is_large_list(data_type)


def _list_lengths(array: pa.Array) -> NDArray[np.int64]:
    offsets = array.offsets.to_numpy().astype(np.int64, copy=False)
    return np.diff(offsets)


def _scalar_positive_values(array: pa.Array) -> NDArray[np.int64]:
    return array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _list_positive_values(array: pa.Array) -> NDArray[np.int64]:
    return array.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def normalize_pcvr_timestamp_range(
    start: int | None,
    end: int | None,
    *,
    label: str = "timestamp_range",
) -> PCVRTimestampRange | None:
    start_value = int(start or 0)
    end_value = int(end or 0)
    if start_value < 0 or end_value < 0:
        raise ValueError(f"{label} bounds must be non-negative")
    if start_value == 0 and end_value == 0:
        return None
    if end_value and start_value >= end_value:
        raise ValueError(f"{label} start must be < end")
    return (start_value or None, end_value or None)


def pcvr_timestamp_range_to_dict(
    timestamp_range: PCVRTimestampRange | None,
) -> dict[str, int | None] | None:
    if timestamp_range is None:
        return None
    start, end = timestamp_range
    return {"start": start, "end": end}


def _timestamp_mask(
    timestamps: NDArray[np.int64],
    timestamp_range: PCVRTimestampRange | None,
) -> NDArray[np.bool_]:
    mask = np.ones(timestamps.shape[0], dtype=np.bool_)
    if timestamp_range is None:
        return mask
    start, end = timestamp_range
    if start is not None:
        mask &= timestamps >= start
    if end is not None:
        mask &= timestamps < end
    return mask


def filter_pcvr_record_batch_by_timestamp_range(
    batch: pa.RecordBatch,
    timestamp_range: PCVRTimestampRange | None,
    *,
    timestamp_column: str = "timestamp",
) -> pa.RecordBatch | None:
    if timestamp_range is None:
        return batch
    timestamp_index = batch.schema.get_field_index(timestamp_column)
    if timestamp_index < 0:
        raise KeyError(f"timestamp column {timestamp_column!r} not found in parquet batch")
    timestamps = batch.column(timestamp_index).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    mask = _timestamp_mask(timestamps, timestamp_range)
    if bool(mask.all()):
        return batch
    if not bool(mask.any()):
        return None
    return batch.filter(pa.array(mask))


def count_pcvr_rows_in_timestamp_range(
    row_groups: list[tuple[str, int, int]],
    timestamp_range: PCVRTimestampRange | None,
    *,
    batch_size: int = DEFAULT_PCVR_OBSERVED_SCHEMA_BATCH_SIZE,
) -> int:
    if timestamp_range is None:
        return int(sum(num_rows for _file_path, _row_group_index, num_rows in row_groups))

    row_count = 0
    current_file_path: str | None = None
    current_parquet_file: pq.ParquetFile | None = None
    for file_path, row_group_index, _num_rows in row_groups:
        if file_path != current_file_path:
            current_file_path = file_path
            current_parquet_file = pq.ParquetFile(file_path)
        if current_parquet_file is None:
            continue
        for batch in current_parquet_file.iter_batches(
            batch_size=batch_size,
            row_groups=[row_group_index],
            columns=["timestamp"],
        ):
            timestamps = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            row_count += int(_timestamp_mask(timestamps, timestamp_range).sum())
    return row_count


def build_pcvr_observed_schema_report(
    data_dir: str | Path,
    schema_path: str | Path,
    *,
    row_group_range: tuple[int, int] | None = None,
    timestamp_range: PCVRTimestampRange | None = None,
    batch_size: int = DEFAULT_PCVR_OBSERVED_SCHEMA_BATCH_SIZE,
    dataset_role: str = "dataset",
) -> dict[str, Any]:
    resolved_dataset_path = Path(data_dir).expanduser().resolve()
    resolved_schema_path = Path(schema_path).expanduser().resolve()
    raw_schema = read_path(resolved_schema_path)
    rg_info = collect_pcvr_row_groups(resolved_dataset_path)
    total_row_groups = len(rg_info)

    if row_group_range is None:
        start_index, end_index = 0, total_row_groups
    else:
        start_index, end_index = row_group_range
        if start_index < 0 or end_index > total_row_groups or start_index >= end_index:
            raise ValueError(
                f"invalid row_group_range={row_group_range}; available row groups: 0..{total_row_groups}"
            )

    selected_row_groups = rg_info[start_index:end_index]
    if not selected_row_groups:
        raise ValueError("at least one Row Group is required to build observed schema report")

    user_int_specs = [
        {
            "fid": int(fid),
            "column": f"user_int_feats_{fid}",
            "declared_dim": int(dim),
            "observed_dim": 1 if int(dim) == 1 else 0,
            "counter": _ExactPositiveCardinalityCounter(int(cardinality)),
        }
        for fid, cardinality, dim in raw_schema["user_int"]
    ]
    item_int_specs = [
        {
            "fid": int(fid),
            "column": f"item_int_feats_{fid}",
            "declared_dim": int(dim),
            "observed_dim": 1 if int(dim) == 1 else 0,
            "counter": _ExactPositiveCardinalityCounter(int(cardinality)),
        }
        for fid, cardinality, dim in raw_schema["item_int"]
    ]
    user_dense_specs = [
        {
            "fid": int(fid),
            "column": f"user_dense_feats_{fid}",
            "observed_dim": 0,
        }
        for fid, _dim in raw_schema["user_dense"]
    ]
    seq_specs: dict[str, dict[str, Any]] = {}
    for domain, config in sorted(raw_schema["seq"].items()):
        prefix = str(config["prefix"])
        ts_fid = config.get("ts_fid")
        feature_specs: list[dict[str, Any]] = []
        for fid, cardinality in config["features"]:
            feature_id = int(fid)
            if ts_fid is not None and feature_id == int(ts_fid):
                feature_specs.append(
                    {
                        "fid": feature_id,
                        "column": f"{prefix}_{feature_id}",
                        "is_timestamp": True,
                        "max_value": 0,
                    }
                )
            else:
                feature_specs.append(
                    {
                        "fid": feature_id,
                        "column": f"{prefix}_{feature_id}",
                        "is_timestamp": False,
                        "counter": _ExactPositiveCardinalityCounter(int(cardinality)),
                    }
                )
        seq_specs[domain] = {
            "prefix": prefix,
            "ts_fid": int(ts_fid) if ts_fid is not None else None,
            "features": feature_specs,
        }

    requested_columns = [spec["column"] for spec in user_int_specs]
    requested_columns.extend(spec["column"] for spec in item_int_specs)
    requested_columns.extend(spec["column"] for spec in user_dense_specs)
    for config in seq_specs.values():
        requested_columns.extend(feature["column"] for feature in config["features"])
    if timestamp_range is not None and "timestamp" not in requested_columns:
        requested_columns.append("timestamp")

    observed_row_count = 0
    current_file_path: str | None = None
    current_parquet_file: pq.ParquetFile | None = None
    for file_path, row_group_index, _num_rows in selected_row_groups:
        if file_path != current_file_path:
            current_file_path = file_path
            current_parquet_file = pq.ParquetFile(file_path)
        if current_parquet_file is None:
            continue
        for batch in current_parquet_file.iter_batches(
            batch_size=batch_size,
            row_groups=[row_group_index],
            columns=requested_columns,
        ):
            batch = filter_pcvr_record_batch_by_timestamp_range(batch, timestamp_range)
            if batch is None:
                continue
            observed_row_count += batch.num_rows
            column_index = {name: index for index, name in enumerate(batch.schema.names)}

            for spec in user_int_specs:
                column = batch.column(column_index[spec["column"]])
                if _is_list_like_arrow_type(column.type):
                    lengths = _list_lengths(column)
                    if lengths.size > 0:
                        spec["observed_dim"] = max(spec["observed_dim"], int(lengths.max()))
                    spec["counter"].add_values(_list_positive_values(column))
                else:
                    spec["counter"].add_values(_scalar_positive_values(column))

            for spec in item_int_specs:
                column = batch.column(column_index[spec["column"]])
                if _is_list_like_arrow_type(column.type):
                    lengths = _list_lengths(column)
                    if lengths.size > 0:
                        spec["observed_dim"] = max(spec["observed_dim"], int(lengths.max()))
                    spec["counter"].add_values(_list_positive_values(column))
                else:
                    spec["counter"].add_values(_scalar_positive_values(column))

            for spec in user_dense_specs:
                column = batch.column(column_index[spec["column"]])
                if _is_list_like_arrow_type(column.type):
                    lengths = _list_lengths(column)
                    if lengths.size > 0:
                        spec["observed_dim"] = max(spec["observed_dim"], int(lengths.max()))
                elif batch.num_rows > 0:
                    spec["observed_dim"] = max(spec["observed_dim"], 1)

            for config in seq_specs.values():
                for feature in config["features"]:
                    column = batch.column(column_index[feature["column"]])
                    values = _list_positive_values(column) if _is_list_like_arrow_type(column.type) else _scalar_positive_values(column)
                    if feature["is_timestamp"]:
                        positive_values = values[values > 0]
                        if positive_values.size > 0:
                            feature["max_value"] = max(feature["max_value"], int(positive_values.max()))
                    else:
                        feature["counter"].add_values(values)

    observed_schema = {
        "user_int": [
            [spec["fid"], int(spec["counter"].count), int(spec["observed_dim"])]
            for spec in user_int_specs
        ],
        "item_int": [
            [spec["fid"], int(spec["counter"].count), int(spec["observed_dim"])]
            for spec in item_int_specs
        ],
        "user_dense": [
            [spec["fid"], int(spec["observed_dim"])]
            for spec in user_dense_specs
        ],
        "seq": {
            domain: {
                "prefix": config["prefix"],
                "ts_fid": config["ts_fid"],
                "features": [
                    [feature["fid"], int(feature["max_value"]) if feature["is_timestamp"] else int(feature["counter"].count)]
                    for feature in config["features"]
                ],
            }
            for domain, config in seq_specs.items()
        },
    }
    report = {
        "dataset_role": str(dataset_role).strip() or "dataset",
        "dataset_path": str(resolved_dataset_path),
        "schema_path": str(resolved_schema_path),
        "row_group_range": [start_index, end_index],
        "timestamp_range": pcvr_timestamp_range_to_dict(timestamp_range),
        "row_group_count": len(selected_row_groups),
        "row_count": int(
            observed_row_count
            if timestamp_range is not None
            else sum(num_rows for _file_path, _rg_index, num_rows in selected_row_groups)
        ),
        "schema": observed_schema,
    }
    logger.info(
        "Built PCVR observed schema report for {}: row_groups={}, rows={}",
        report["dataset_role"],
        report["row_group_range"],
        report["row_count"],
    )
    return report


def collect_pcvr_row_groups(data_dir: str | Path) -> list[tuple[str, int, int]]:
    data_path = Path(data_dir).expanduser()
    if data_path.is_dir():
        pq_files = sorted(str(path) for path in data_path.glob("*.parquet"))
    else:
        pq_files = [str(data_path)]
    if not pq_files:
        raise FileNotFoundError(f"No .parquet files found at {data_dir}")

    rg_info: list[tuple[str, int, int]] = []
    for file_path in pq_files:
        parquet_file = pq.ParquetFile(file_path)
        for index in range(parquet_file.metadata.num_row_groups):
            rg_info.append((file_path, index, parquet_file.metadata.row_group(index).num_rows))
    return rg_info


def plan_pcvr_row_group_split(
    rg_info: list[tuple[str, int, int]],
    *,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
) -> PCVRRowGroupSplitPlan:
    total_rgs = len(rg_info)
    if total_rgs == 0:
        raise ValueError("at least one Row Group is required")

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs
    reuse_train_for_valid = False

    if total_rgs == 1:
        n_train_rgs = 1
        n_valid_rgs = 1
        reuse_train_for_valid = True
    elif n_train_rgs == 0:
        n_train_rgs = total_rgs - 1
        n_valid_rgs = 1

    if train_ratio < 1.0 and not reuse_train_for_valid:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))

    train_row_group_range = (0, n_train_rgs)
    valid_row_group_range = (
        (0, total_rgs) if reuse_train_for_valid else (n_train_rgs, total_rgs)
    )
    train_rows = sum(
        row_group[2] for row_group in rg_info[train_row_group_range[0] : train_row_group_range[1]]
    )
    valid_rows = sum(
        row_group[2] for row_group in rg_info[valid_row_group_range[0] : valid_row_group_range[1]]
    )
    return PCVRRowGroupSplitPlan(
        total_row_groups=total_rgs,
        train_row_groups=n_train_rgs,
        valid_row_groups=n_valid_rgs,
        train_row_group_range=train_row_group_range,
        valid_row_group_range=valid_row_group_range,
        train_rows=train_rows,
        valid_rows=valid_rows,
        reuse_train_for_valid=reuse_train_for_valid,
    )


__all__ = [
    "DEFAULT_PCVR_OBSERVED_SCHEMA_BATCH_SIZE",
    "PCVRRowGroupSplitPlan",
    "PCVRTimestampRange",
    "build_pcvr_observed_schema_report",
    "collect_pcvr_row_groups",
    "count_pcvr_rows_in_timestamp_range",
    "filter_pcvr_record_batch_by_timestamp_range",
    "normalize_pcvr_timestamp_range",
    "pcvr_timestamp_range_to_dict",
    "plan_pcvr_row_group_split",
]
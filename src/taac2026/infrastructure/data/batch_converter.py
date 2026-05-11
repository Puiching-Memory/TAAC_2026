from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from numpy.typing import NDArray

from taac2026.domain.schema import BUCKET_BOUNDARIES
from taac2026.infrastructure.data.schema_layout import PCVRSchemaLayout
from taac2026.infrastructure.logging import logger


SEQUENCE_STATS_DIM = 6


@dataclass(frozen=True, slots=True)
class IntColumnPlan:
    column_index: int
    dim: int
    output_offset: int
    vocab_size: int


@dataclass(frozen=True, slots=True)
class DenseColumnPlan:
    column_index: int
    dim: int
    output_offset: int


@dataclass(frozen=True, slots=True)
class SequenceSideColumnPlan:
    column_index: int
    slot: int
    vocab_size: int


@dataclass(frozen=True, slots=True)
class SequenceColumnPlan:
    domain: str
    max_len: int
    side_columns: tuple[SequenceSideColumnPlan, ...]
    timestamp_column_index: int | None


@dataclass(frozen=True, slots=True)
class PCVRColumnPlan:
    required_column_names: tuple[str, ...]
    column_indices: dict[str, int]
    user_int: tuple[IntColumnPlan, ...]
    item_int: tuple[IntColumnPlan, ...]
    user_dense: tuple[DenseColumnPlan, ...]
    sequences: dict[str, SequenceColumnPlan]

    def record_batch_columns(self) -> list[str] | None:
        return list(self.required_column_names) or None


def pad_list_offsets_values(
    offsets: NDArray[Any],
    values: NDArray[Any],
    *,
    row_count: int,
    width: int,
    dtype: np.dtype[Any] | type[np.generic],
) -> tuple[NDArray[Any], NDArray[np.int64]]:
    padded = np.zeros((row_count, width), dtype=dtype)
    if row_count <= 0 or width <= 0:
        return padded, np.zeros(row_count, dtype=np.int64)

    starts = np.asarray(offsets[:row_count], dtype=np.int64)
    ends = np.asarray(offsets[1 : row_count + 1], dtype=np.int64)
    raw_lengths = np.maximum(ends - starts, 0)
    lengths = np.minimum(raw_lengths, int(width)).astype(np.int64, copy=False)
    if int(lengths.sum()) <= 0:
        return padded, lengths

    fixed_raw_length = int(raw_lengths[0])
    if fixed_raw_length > 0 and bool(np.all(raw_lengths == fixed_raw_length)):
        source_start = int(starts[0])
        source_end = source_start + row_count * fixed_raw_length
        flat_values = values[source_start:source_end]
        if flat_values.shape[0] == row_count * fixed_raw_length:
            use_len = min(fixed_raw_length, int(width))
            padded[:, :use_len] = flat_values.reshape(row_count, fixed_raw_length)[
                :, :use_len
            ]
            return padded, lengths

    for row_index, use_len in enumerate(lengths):
        if use_len <= 0:
            continue
        start = int(starts[row_index])
        padded[row_index, :use_len] = values[start : start + int(use_len)]
    return padded, lengths


def build_pcvr_column_plan(
    layout: PCVRSchemaLayout,
    parquet_schema_names: list[str],
) -> PCVRColumnPlan:
    required_column_names = layout.required_column_names(parquet_schema_names)
    column_indices = {name: index for index, name in enumerate(required_column_names)}

    user_int = _build_int_column_plan(
        layout.user_int_cols,
        column_indices=column_indices,
        prefix="user_int_feats",
    )
    item_int = _build_int_column_plan(
        layout.item_int_cols,
        column_indices=column_indices,
        prefix="item_int_feats",
    )

    user_dense: list[DenseColumnPlan] = []
    output_offset = 0
    for fid, dim in layout.user_dense_cols:
        user_dense.append(
            DenseColumnPlan(
                column_index=column_indices[f"user_dense_feats_{fid}"],
                dim=dim,
                output_offset=output_offset,
            )
        )
        output_offset += dim

    sequences: dict[str, SequenceColumnPlan] = {}
    for domain in layout.seq_domains:
        sequence_layout = layout.sequences[domain]
        side_columns = tuple(
            SequenceSideColumnPlan(
                column_index=column_indices[f"{sequence_layout.prefix}_{fid}"],
                slot=slot,
                vocab_size=sequence_layout.vocab_sizes[fid],
            )
            for slot, fid in enumerate(sequence_layout.sideinfo_fids)
        )
        timestamp_column_index = None
        if sequence_layout.timestamp_fid is not None:
            timestamp_column_index = column_indices[
                f"{sequence_layout.prefix}_{sequence_layout.timestamp_fid}"
            ]
        sequences[domain] = SequenceColumnPlan(
            domain=domain,
            max_len=sequence_layout.max_len,
            side_columns=side_columns,
            timestamp_column_index=timestamp_column_index,
        )

    return PCVRColumnPlan(
        required_column_names=required_column_names,
        column_indices=column_indices,
        user_int=user_int,
        item_int=item_int,
        user_dense=tuple(user_dense),
        sequences=sequences,
    )


def _build_int_column_plan(
    columns: tuple[tuple[int, int, int], ...],
    *,
    column_indices: dict[str, int],
    prefix: str,
) -> tuple[IntColumnPlan, ...]:
    plan: list[IntColumnPlan] = []
    output_offset = 0
    for fid, vocab_size, dim in columns:
        plan.append(
            IntColumnPlan(
                column_index=column_indices[f"{prefix}_{fid}"],
                dim=dim,
                output_offset=output_offset,
                vocab_size=vocab_size,
            )
        )
        output_offset += dim
    return tuple(plan)


class PCVRRecordBatchConverter:
    def __init__(
        self,
        *,
        layout: PCVRSchemaLayout,
        column_plan: PCVRColumnPlan,
        batch_size: int,
        clip_vocab: bool,
        is_training: bool,
        strict_time_filter: bool,
    ) -> None:
        self.layout = layout
        self.column_plan = column_plan
        self.batch_size = batch_size
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.strict_time_filter = strict_time_filter
        self.oob_stats: dict[tuple[str, int], dict[str, int]] = {}

        self.user_int_buffer = np.zeros(
            (batch_size, layout.user_int_schema.total_dim), dtype=np.int64
        )
        self.item_int_buffer = np.zeros(
            (batch_size, layout.item_int_schema.total_dim), dtype=np.int64
        )
        self.user_dense_buffer = np.zeros(
            (batch_size, layout.user_dense_schema.total_dim), dtype=np.float32
        )
        self.user_int_missing_buffer = np.ones(
            (batch_size, layout.user_int_schema.total_dim), dtype=np.bool_
        )
        self.item_int_missing_buffer = np.ones(
            (batch_size, layout.item_int_schema.total_dim), dtype=np.bool_
        )
        self.user_dense_missing_buffer = np.ones(
            (batch_size, layout.user_dense_schema.total_dim), dtype=np.bool_
        )
        self.sequence_buffers = {
            domain: np.zeros(
                (
                    batch_size,
                    len(layout.sideinfo_fids[domain]),
                    layout.seq_maxlen[domain],
                ),
                dtype=np.int64,
            )
            for domain in layout.seq_domains
        }
        self.sequence_lengths = {
            domain: np.zeros(batch_size, dtype=np.int64) for domain in layout.seq_domains
        }
        self.sequence_time_buckets = {
            domain: np.zeros((batch_size, layout.seq_maxlen[domain]), dtype=np.int64)
            for domain in layout.seq_domains
        }
        self.sequence_stats = {
            domain: np.zeros((batch_size, SEQUENCE_STATS_DIM), dtype=np.float32)
            for domain in layout.seq_domains
        }

    def pad_int_column(
        self,
        arrow_col: pa.ListArray,
        width: int,
        row_count: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        padded, lengths = pad_list_offsets_values(
            arrow_col.offsets.to_numpy(),
            arrow_col.values.to_numpy(),
            row_count=row_count,
            width=width,
            dtype=np.int64,
        )
        padded[padded <= 0] = 0
        return padded, lengths

    def pad_float_column(
        self,
        arrow_col: pa.ListArray,
        width: int,
        row_count: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        padded, lengths = pad_list_offsets_values(
            arrow_col.offsets.to_numpy(),
            arrow_col.values.to_numpy(),
            row_count=row_count,
            width=width,
            dtype=np.float32,
        )
        positions = np.arange(width).reshape(1, -1) if width > 0 else np.zeros((1, 0), dtype=np.int64)
        present = positions < lengths.reshape(-1, 1)
        finite = np.isfinite(padded)
        missing = ~(present & finite)
        padded[~finite] = 0.0
        return padded, missing

    def convert(self, batch: pa.RecordBatch) -> dict[str, Any]:
        row_count = batch.num_rows
        timestamps = self._timestamps(batch)
        result = self._base_result(batch, row_count, timestamps)
        self._fill_int_features(
            batch,
            row_count,
            plan=self.column_plan.user_int,
            buffer=self.user_int_buffer[:row_count],
            missing_buffer=self.user_int_missing_buffer[:row_count],
            group="user_int",
        )
        self._fill_int_features(
            batch,
            row_count,
            plan=self.column_plan.item_int,
            buffer=self.item_int_buffer[:row_count],
            missing_buffer=self.item_int_missing_buffer[:row_count],
            group="item_int",
        )
        self._fill_dense_features(batch, row_count)

        result["user_int_feats"] = torch.from_numpy(self.user_int_buffer[:row_count].copy())
        result["item_int_feats"] = torch.from_numpy(self.item_int_buffer[:row_count].copy())
        result["user_dense_feats"] = torch.from_numpy(
            self.user_dense_buffer[:row_count].copy()
        )
        result["user_int_missing_mask"] = torch.from_numpy(self.user_int_missing_buffer[:row_count].copy())
        result["item_int_missing_mask"] = torch.from_numpy(self.item_int_missing_buffer[:row_count].copy())
        result["user_dense_missing_mask"] = torch.from_numpy(self.user_dense_missing_buffer[:row_count].copy())
        result["item_dense_missing_mask"] = torch.zeros(row_count, 0, dtype=torch.bool)
        self._add_sequence_features(batch, row_count, timestamps, result)
        return result

    def dump_oob_stats(self, path: str | None = None) -> None:
        if not self.oob_stats:
            logger.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, column_index), stats in sorted(self.oob_stats.items()):
            direction = "TOO_HIGH" if stats["min_oob"] >= stats["vocab"] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={column_index}: vocab={stats['vocab']}, "
                f"oob_count={stats['count']}, range=[{stats['min_oob']}, {stats['max']}], "
                f"{direction}"
            )
        message = "\n".join(lines)
        if path:
            with Path(path).open("w") as file:
                file.write(message + "\n")
            logger.info("OOB stats written to {}", path)
        else:
            logger.info(message)

    def _timestamps(self, batch: pa.RecordBatch) -> NDArray[np.int64]:
        return batch.column(self.column_plan.column_indices["timestamp"]).to_numpy().astype(np.int64)

    def _base_result(
        self,
        batch: pa.RecordBatch,
        row_count: int,
        timestamps: NDArray[np.int64],
    ) -> dict[str, Any]:
        labels = self._labels(batch, row_count)
        user_ids = batch.column(self.column_plan.column_indices["user_id"]).to_pylist()
        return {
            "item_dense_feats": torch.zeros(row_count, 0, dtype=torch.float32),
            "label": torch.from_numpy(labels),
            "timestamp": torch.from_numpy(timestamps),
            "user_id": user_ids,
            "_seq_domains": self.layout.seq_domains,
        }

    def _labels(self, batch: pa.RecordBatch, row_count: int) -> NDArray[np.int64]:
        if not self.is_training:
            return np.zeros(row_count, dtype=np.int64)
        return (
            batch.column(self.column_plan.column_indices["label_type"])
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
            == 2
        ).astype(np.int64)

    def _fill_int_features(
        self,
        batch: pa.RecordBatch,
        row_count: int,
        *,
        plan: tuple[IntColumnPlan, ...],
        buffer: NDArray[np.int64],
        missing_buffer: NDArray[np.bool_],
        group: str,
    ) -> None:
        buffer[:] = 0
        missing_buffer[:] = True
        for feature in plan:
            column = batch.column(feature.column_index)
            if feature.dim == 1:
                values, missing = self._scalar_int_values_and_missing(column)
                if feature.vocab_size > 0:
                    self.record_oob(group, feature.column_index, values, feature.vocab_size)
                else:
                    values[:] = 0
                buffer[:, feature.output_offset] = values
                missing_buffer[:, feature.output_offset] = missing | (values <= 0)
                continue

            padded, _lengths = self.pad_int_column(column, feature.dim, row_count)
            if feature.vocab_size > 0:
                self.record_oob(group, feature.column_index, padded, feature.vocab_size)
            else:
                padded[:] = 0
            buffer[:, feature.output_offset : feature.output_offset + feature.dim] = padded
            missing_buffer[:, feature.output_offset : feature.output_offset + feature.dim] = padded <= 0

    def _scalar_int_values_and_missing(self, column: pa.Array) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        values = column.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        missing = column.is_null().to_numpy(zero_copy_only=False).astype(np.bool_) | (values <= 0)
        values[values <= 0] = 0
        return values, missing

    def _fill_dense_features(self, batch: pa.RecordBatch, row_count: int) -> None:
        buffer = self.user_dense_buffer[:row_count]
        missing_buffer = self.user_dense_missing_buffer[:row_count]
        buffer[:] = 0
        missing_buffer[:] = True
        for feature in self.column_plan.user_dense:
            padded, missing = self.pad_float_column(batch.column(feature.column_index), feature.dim, row_count)
            buffer[:, feature.output_offset : feature.output_offset + feature.dim] = padded
            missing_buffer[:, feature.output_offset : feature.output_offset + feature.dim] = missing

    def _add_sequence_features(
        self,
        batch: pa.RecordBatch,
        row_count: int,
        timestamps: NDArray[np.int64],
        result: dict[str, Any],
    ) -> None:
        for domain in self.layout.seq_domains:
            sequence_plan = self.column_plan.sequences[domain]
            tokens = self.sequence_buffers[domain][:row_count]
            lengths = self.sequence_lengths[domain][:row_count]
            time_buckets = self.sequence_time_buckets[domain][:row_count]
            stats = self.sequence_stats[domain][:row_count]
            tokens[:] = 0
            lengths[:] = 0
            time_buckets[:] = 0
            stats[:] = 0.0

            side_columns = self._sequence_side_arrays(batch, sequence_plan)
            timestamps_padded = np.zeros((row_count, sequence_plan.max_len), dtype=np.int64)
            if self.strict_time_filter and sequence_plan.timestamp_column_index is not None:
                self._fill_strict_sequence(
                    batch=batch,
                    row_count=row_count,
                    timestamps=timestamps,
                    sequence_plan=sequence_plan,
                    side_columns=side_columns,
                    tokens=tokens,
                    lengths=lengths,
                    timestamps_padded=timestamps_padded,
                )
            else:
                self._fill_sequence(
                    batch=batch,
                    row_count=row_count,
                    sequence_plan=sequence_plan,
                    side_columns=side_columns,
                    tokens=tokens,
                    lengths=lengths,
                    timestamps_padded=timestamps_padded,
                )

            tokens[tokens <= 0] = 0
            self._clip_sequence_vocab(domain, sequence_plan, tokens)
            self._fill_raw_sequence_stats(tokens, lengths, stats)
            self._deduplicate_sequence(tokens, lengths, timestamps_padded)
            self._fill_time_buckets(timestamps, timestamps_padded, time_buckets)
            self._fill_sequence_time_stats(lengths, time_buckets, stats)

            result[domain] = torch.from_numpy(tokens.copy())
            result[f"{domain}_len"] = torch.from_numpy(lengths.copy())
            result[f"{domain}_time_bucket"] = torch.from_numpy(time_buckets.copy())
            result[f"{domain}_stats"] = torch.from_numpy(stats.copy())

    def _sequence_side_arrays(
        self,
        batch: pa.RecordBatch,
        sequence_plan: SequenceColumnPlan,
    ) -> tuple[tuple[NDArray[Any], NDArray[Any], SequenceSideColumnPlan], ...]:
        arrays = []
        for side_column in sequence_plan.side_columns:
            column = batch.column(side_column.column_index)
            arrays.append((column.offsets.to_numpy(), column.values.to_numpy(), side_column))
        return tuple(arrays)

    def _fill_sequence(
        self,
        *,
        batch: pa.RecordBatch,
        row_count: int,
        sequence_plan: SequenceColumnPlan,
        side_columns: tuple[tuple[NDArray[Any], NDArray[Any], SequenceSideColumnPlan], ...],
        tokens: NDArray[np.int64],
        lengths: NDArray[np.int64],
        timestamps_padded: NDArray[np.int64],
    ) -> None:
        for offsets, values, side_column in side_columns:
            padded, feature_lengths = pad_list_offsets_values(
                offsets,
                values,
                row_count=row_count,
                width=sequence_plan.max_len,
                dtype=np.int64,
            )
            tokens[:, side_column.slot, :] = padded
            np.maximum(lengths, feature_lengths, out=lengths)

        if sequence_plan.timestamp_column_index is None:
            return
        timestamp_column = batch.column(sequence_plan.timestamp_column_index)
        timestamps_padded[:, :] = pad_list_offsets_values(
            timestamp_column.offsets.to_numpy(),
            timestamp_column.values.to_numpy(),
            row_count=row_count,
            width=sequence_plan.max_len,
            dtype=np.int64,
        )[0]

    def _fill_strict_sequence(
        self,
        *,
        batch: pa.RecordBatch,
        row_count: int,
        timestamps: NDArray[np.int64],
        sequence_plan: SequenceColumnPlan,
        side_columns: tuple[tuple[NDArray[Any], NDArray[Any], SequenceSideColumnPlan], ...],
        tokens: NDArray[np.int64],
        lengths: NDArray[np.int64],
        timestamps_padded: NDArray[np.int64],
    ) -> None:
        timestamp_column = batch.column(sequence_plan.timestamp_column_index)
        timestamp_offsets = timestamp_column.offsets.to_numpy()
        timestamp_values = timestamp_column.values.to_numpy()
        for row_index in range(row_count):
            start = int(timestamp_offsets[row_index])
            end = int(timestamp_offsets[row_index + 1])
            if end <= start:
                continue
            row_timestamps = timestamp_values[start:end]
            valid_positions = np.flatnonzero(
                (row_timestamps > 0) & (row_timestamps < timestamps[row_index])
            )
            if valid_positions.size == 0:
                continue
            if valid_positions.size > sequence_plan.max_len:
                valid_positions = valid_positions[-sequence_plan.max_len :]

            valid_length = int(valid_positions.size)
            lengths[row_index] = valid_length
            timestamps_padded[row_index, :valid_length] = row_timestamps[valid_positions]
            self._copy_strict_side_columns(row_index, valid_positions, side_columns, tokens)

    def _copy_strict_side_columns(
        self,
        row_index: int,
        valid_positions: NDArray[np.int64],
        side_columns: tuple[tuple[NDArray[Any], NDArray[Any], SequenceSideColumnPlan], ...],
        tokens: NDArray[np.int64],
    ) -> None:
        for offsets, values, side_column in side_columns:
            start = int(offsets[row_index])
            end = int(offsets[row_index + 1])
            side_len = end - start
            if side_len <= 0:
                continue
            side_positions = valid_positions[valid_positions < side_len]
            if side_positions.size == 0:
                continue
            tokens[row_index, side_column.slot, : side_positions.size] = values[
                start + side_positions
            ]

    def _clip_sequence_vocab(
        self,
        domain: str,
        sequence_plan: SequenceColumnPlan,
        tokens: NDArray[np.int64],
    ) -> None:
        for side_column in sequence_plan.side_columns:
            slice_tokens = tokens[:, side_column.slot, :]
            if side_column.vocab_size > 0:
                self.record_oob(
                    f"seq_{domain}",
                    side_column.column_index,
                    slice_tokens,
                    side_column.vocab_size,
                )
            else:
                slice_tokens[:] = 0

    def _fill_time_buckets(
        self,
        timestamps: NDArray[np.int64],
        timestamps_padded: NDArray[np.int64],
        time_buckets: NDArray[np.int64],
    ) -> None:
        if timestamps_padded.shape[1] == 0:
            return
        time_diff = np.maximum(timestamps.reshape(-1, 1) - timestamps_padded, 0)
        raw_buckets = np.clip(
            np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
            0,
            len(BUCKET_BOUNDARIES) - 1,
        )
        buckets = raw_buckets.reshape(timestamps_padded.shape) + 1
        buckets[timestamps_padded == 0] = 0
        time_buckets[:] = buckets

    def _fill_raw_sequence_stats(
        self,
        tokens: NDArray[np.int64],
        lengths: NDArray[np.int64],
        stats: NDArray[np.float32],
    ) -> None:
        batch_size, feature_count, max_len = tokens.shape
        for row_index in range(batch_size):
            raw_length = min(max(int(lengths[row_index]), 0), max_len)
            if raw_length <= 0 or feature_count <= 0:
                continue
            row_tokens = tokens[row_index, :, :raw_length]
            active_events = np.any(row_tokens > 0, axis=0)
            active_count = int(active_events.sum())
            if active_count <= 0:
                continue
            event_signatures = {
                tuple(int(value) for value in row_tokens[:, event_index])
                for event_index in np.flatnonzero(active_events)
            }
            unique_count = len(event_signatures)
            nonzero_fraction = float((row_tokens > 0).sum()) / float(max(1, raw_length * feature_count))
            stats[row_index, 0] = float(raw_length)
            stats[row_index, 1] = float(active_count)
            stats[row_index, 2] = float(unique_count)
            stats[row_index, 3] = 1.0 - float(unique_count) / float(max(1, active_count))
            stats[row_index, 4] = nonzero_fraction

    def _deduplicate_sequence(
        self,
        tokens: NDArray[np.int64],
        lengths: NDArray[np.int64],
        timestamps_padded: NDArray[np.int64],
    ) -> None:
        batch_size, _feature_count, max_len = tokens.shape
        for row_index in range(batch_size):
            raw_length = min(max(int(lengths[row_index]), 0), max_len)
            if raw_length <= 1:
                continue
            seen: set[tuple[int, ...]] = set()
            keep_positions: list[int] = []
            for position in range(raw_length - 1, -1, -1):
                event = tuple(int(value) for value in tokens[row_index, :, position])
                if not any(value > 0 for value in event) or event in seen:
                    continue
                seen.add(event)
                keep_positions.append(position)
            keep_positions.reverse()
            new_length = len(keep_positions)
            if new_length == raw_length:
                continue
            if new_length <= 0:
                tokens[row_index].fill(0)
                timestamps_padded[row_index].fill(0)
                lengths[row_index] = 0
                continue
            selected_tokens = tokens[row_index][:, keep_positions].copy()
            selected_timestamps = timestamps_padded[row_index, keep_positions].copy()
            tokens[row_index].fill(0)
            timestamps_padded[row_index].fill(0)
            tokens[row_index, :, :new_length] = selected_tokens
            timestamps_padded[row_index, :new_length] = selected_timestamps
            lengths[row_index] = new_length

    def _fill_sequence_time_stats(
        self,
        lengths: NDArray[np.int64],
        time_buckets: NDArray[np.int64],
        stats: NDArray[np.float32],
    ) -> None:
        max_len = time_buckets.shape[1]
        for row_index, length_value in enumerate(lengths):
            length = min(max(int(length_value), 0), max_len)
            if length <= 0:
                continue
            stats[row_index, 1] = float(length)
            stats[row_index, 5] = float(time_buckets[row_index, length - 1])

    def record_oob(
        self,
        group: str,
        column_index: int,
        values: NDArray[np.int64],
        vocab_size: int,
    ) -> None:
        oob_mask = values >= vocab_size
        if not oob_mask.any():
            return
        oob_values = values[oob_mask]
        count = int(oob_mask.sum())
        max_value = int(oob_values.max())
        min_oob = int(oob_values.min())
        key = (group, column_index)
        if key in self.oob_stats:
            stats = self.oob_stats[key]
            stats["count"] += count
            stats["max"] = max(stats["max"], max_value)
            stats["min_oob"] = min(stats["min_oob"], min_oob)
        else:
            self.oob_stats[key] = {
                "count": count,
                "max": max_value,
                "min_oob": min_oob,
                "vocab": vocab_size,
            }
        if self.clip_vocab:
            values[oob_mask] = 0
            return
        raise ValueError(
            f"{group} col_idx={column_index}: {count} values out of range "
            f"[0, {vocab_size}), actual=[{min_oob}, {max_value}]. "
            "Use clip_vocab=True to clip or fix schema.json"
        )

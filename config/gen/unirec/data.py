from __future__ import annotations

"""Experiment-private parquet-to-batch pipeline for the UniRec package.

This file stays package-local on purpose: each experiment package owns its
raw-data parsing and row-to-batch encoding logic so model behavior remains
reproducible and easy to trace back to the package itself.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from taac2026.domain.config import DataConfig
from taac2026.domain.types import BatchTensors, DataStats
from taac2026.infrastructure.io.files import stable_hash64


DENSE_FEATURE_DIM = 16
PADDING_TOKEN_ID = 0
TIMESTAMP_FEATURE_ID = 99
TIME_GAP_BUCKET_COUNT = 64
AUTHOR_TOKEN_COUNT = 2
SEQUENCE_TIMESTAMP_FEATURE_IDS = {
	"action_seq": 28,
	"content_seq": 41,
	"item_seq": 29,
}
SEQUENCE_POST_FEATURE_IDS = {
	"action_seq": 25,
	"content_seq": 44,
	"item_seq": 49,
}
SEQUENCE_AUTHOR_FEATURE_IDS = {
	"action_seq": (21, 20),
	"content_seq": (45, 46),
	"item_seq": (31, 36),
}
SEQUENCE_ACTION_FEATURE_IDS = {
	"action_seq": (19, 26, 27),
	"content_seq": (42, 43, 48),
	"item_seq": (30, 34, 35),
}


@dataclass(slots=True)
class _EncodedSample:
	user_tokens: np.ndarray
	user_mask: np.ndarray
	candidate_tokens: np.ndarray
	candidate_mask: np.ndarray
	candidate_post_tokens: np.ndarray
	candidate_post_mask: np.ndarray
	candidate_author_tokens: np.ndarray
	candidate_author_mask: np.ndarray
	context_tokens: np.ndarray
	context_mask: np.ndarray
	history_tokens: np.ndarray
	history_mask: np.ndarray
	history_post_tokens: np.ndarray
	history_author_tokens: np.ndarray
	history_action_tokens: np.ndarray
	history_time_gap: np.ndarray
	history_group_ids: np.ndarray
	sequence_tokens: np.ndarray
	sequence_mask: np.ndarray
	dense_features: np.ndarray
	label: float
	user_index: int
	item_index: int
	item_logq: float


@dataclass(slots=True)
class _SequenceEvent:
	sequence_index: int
	timestamp: int | None
	post_signature: str
	author_signature: str
	action_signature: str
	composite_signature: str
	gap_bucket: int


class _EncodedDataset(Dataset[_EncodedSample]):
	def __init__(self, samples: list[_EncodedSample]) -> None:
		self._samples = samples

	def __len__(self) -> int:
		return len(self._samples)

	def __getitem__(self, index: int) -> _EncodedSample:
		return self._samples[index]


def _resolve_dataset_path(dataset_path: str) -> Path:
	path = Path(dataset_path).expanduser()
	if path.is_file():
		return path
	if path.is_dir():
		candidates = sorted(path.rglob("*.parquet"))
		if candidates:
			return candidates[0]
	raise FileNotFoundError(f"Cannot resolve parquet dataset from {dataset_path}")


def _iter_rows(dataset_path: Path, batch_rows: int) -> Iterable[dict[str, Any]]:
	parquet = pq.ParquetFile(dataset_path)
	for batch in parquet.iter_batches(batch_size=max(1, batch_rows)):
		for row in batch.to_pylist():
			yield row


def _sort_rows_by_timestamp(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
	materialized = list(rows)
	materialized.sort(key=lambda row: int(row.get("timestamp", 0) or 0))
	return materialized


def _time_split(rows: list[dict[str, Any]], val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	if not rows:
		return [], []
	if len(rows) == 1:
		return rows, rows
	train_size = int(len(rows) * (1.0 - val_ratio))
	train_size = max(1, min(len(rows) - 1, train_size))
	train_rows = rows[:train_size]
	val_rows = rows[train_size:] or rows[-1:]
	return train_rows, val_rows


def _token_id(signature: str, vocab_size: int) -> int:
	if vocab_size <= 1:
		raise ValueError("vocab_size must be greater than 1")
	return (stable_hash64(signature) % (vocab_size - 1)) + 1


def _iter_features(value: Any) -> Iterable[dict[str, Any]]:
	if value is None:
		return []
	if isinstance(value, list):
		return [item for item in value if isinstance(item, dict)]
	if isinstance(value, dict):
		return [value]
	return []


def _array_values(feature: dict[str, Any], key: str) -> list[float]:
	values = feature.get(key) or []
	if not isinstance(values, list):
		return []
	return [float(item) for item in values if item is not None]


def _numeric_feature_value(feature: dict[str, Any]) -> float:
	if feature.get("int_value") is not None:
		return float(feature["int_value"])
	if feature.get("float_value") is not None:
		return float(feature["float_value"])
	if feature.get("int_array") is not None:
		values = _array_values(feature, "int_array")
		return float(np.mean(values)) if values else 0.0
	if feature.get("float_array") is not None:
		values = _array_values(feature, "float_array")
		return float(np.mean(values)) if values else 0.0
	return 0.0


def _feature_signature(prefix: str, feature: dict[str, Any]) -> str:
	feature_id = feature.get("feature_id", "unknown")
	if feature.get("int_value") is not None:
		return f"{prefix}|{feature_id}|iv|{int(feature['int_value'])}"
	if feature.get("float_value") is not None:
		return f"{prefix}|{feature_id}|fv|{float(feature['float_value']):.4f}"
	if feature.get("int_array") is not None:
		values = _array_values(feature, "int_array")[:4]
		joined = ",".join(str(int(item)) for item in values)
		return f"{prefix}|{feature_id}|ia|{joined}|n={len(values)}"
	if feature.get("float_array") is not None:
		values = _array_values(feature, "float_array")[:4]
		joined = ",".join(f"{item:.3f}" for item in values)
		return f"{prefix}|{feature_id}|fa|{joined}|n={len(values)}"
	return f"{prefix}|{feature_id}|empty"


def _bucket_gap(reference_timestamp: int, event_timestamp: int | None) -> int:
	if event_timestamp is None:
		return 0
	gap = max(0, reference_timestamp - event_timestamp)
	if gap <= 0:
		return 0
	return min(TIME_GAP_BUCKET_COUNT - 1, int(math.log2(gap + 1)))


def _sequence_feature_arrays(sequence_payload: Any) -> dict[int, list[int]]:
	arrays: dict[int, list[int]] = {}
	for feature in _iter_features(sequence_payload):
		if feature.get("int_array") is None:
			continue
		feature_id = int(feature.get("feature_id", -1))
		values = [int(item) for item in feature.get("int_array", []) if item is not None]
		if values:
			arrays[feature_id] = values
	return arrays


def _is_timestamp_array(values: list[int]) -> bool:
	if not values:
		return False
	timestamp_like = sum(1 for value in values if 1_500_000_000 <= value <= 2_100_000_000)
	return timestamp_like / max(1, len(values)) >= 0.8


def _is_small_categorical_array(values: list[int]) -> bool:
	if not values:
		return False
	return max(values) <= 128 and min(values) >= 0


def _rank_sequence_feature_ids(arrays: dict[int, list[int]], excluded_ids: set[int]) -> list[int]:
	ranked: list[tuple[int, int, int, int]] = []
	for feature_id, values in arrays.items():
		if feature_id in excluded_ids or not values:
			continue
		preview = values[: min(256, len(values))]
		ranked.append((len(set(preview)), max(abs(value) for value in preview), len(values), feature_id))
	ranked.sort(reverse=True)
	return [feature_id for _, _, _, feature_id in ranked]


def _resolve_timestamp_feature_id(sequence_name: str, arrays: dict[int, list[int]]) -> int | None:
	preferred = SEQUENCE_TIMESTAMP_FEATURE_IDS.get(sequence_name)
	if preferred in arrays:
		return preferred
	if TIMESTAMP_FEATURE_ID in arrays:
		return TIMESTAMP_FEATURE_ID
	for feature_id, values in arrays.items():
		if _is_timestamp_array(values):
			return feature_id
	return None


def _resolve_post_feature_id(
	sequence_name: str,
	arrays: dict[int, list[int]],
	timestamp_feature_id: int | None,
) -> int | None:
	preferred = SEQUENCE_POST_FEATURE_IDS.get(sequence_name)
	if preferred in arrays and preferred != timestamp_feature_id:
		return preferred
	for feature_id in _rank_sequence_feature_ids(arrays, {timestamp_feature_id} if timestamp_feature_id is not None else set()):
		if not _is_small_categorical_array(arrays[feature_id]):
			return feature_id
	ranked = _rank_sequence_feature_ids(arrays, {timestamp_feature_id} if timestamp_feature_id is not None else set())
	return ranked[0] if ranked else None


def _resolve_author_feature_ids(
	sequence_name: str,
	arrays: dict[int, list[int]],
	timestamp_feature_id: int | None,
	post_feature_id: int | None,
) -> list[int]:
	excluded_ids = {feature_id for feature_id in (timestamp_feature_id, post_feature_id) if feature_id is not None}
	selected: list[int] = []
	for feature_id in SEQUENCE_AUTHOR_FEATURE_IDS.get(sequence_name, ()):
		if feature_id in arrays and feature_id not in excluded_ids:
			selected.append(feature_id)
	for feature_id in _rank_sequence_feature_ids(arrays, excluded_ids):
		if feature_id in selected:
			continue
		if _is_small_categorical_array(arrays[feature_id]):
			continue
		selected.append(feature_id)
		if len(selected) >= AUTHOR_TOKEN_COUNT:
			break
	if not selected:
		fallback = _rank_sequence_feature_ids(arrays, excluded_ids)
		if fallback:
			selected.append(fallback[0])
	return selected[:AUTHOR_TOKEN_COUNT]


def _resolve_action_feature_ids(
	sequence_name: str,
	arrays: dict[int, list[int]],
	timestamp_feature_id: int | None,
	post_feature_id: int | None,
	author_feature_ids: list[int],
) -> list[int]:
	excluded_ids = {
		feature_id
		for feature_id in (timestamp_feature_id, post_feature_id, *author_feature_ids)
		if feature_id is not None
	}
	selected: list[int] = []
	for feature_id in SEQUENCE_ACTION_FEATURE_IDS.get(sequence_name, ()):
		if feature_id in arrays and feature_id not in excluded_ids:
			selected.append(feature_id)
	for feature_id, values in arrays.items():
		if feature_id in excluded_ids or feature_id in selected:
			continue
		if _is_small_categorical_array(values):
			selected.append(feature_id)
	if not selected and post_feature_id is not None:
		selected.append(post_feature_id)
	return selected[:4]


def _value_at(values: list[int] | None, index: int, default: int | None) -> int | None:
	if values is None or index >= len(values):
		return default
	return values[index]


def _role_token_array(
	signatures: list[str],
	array_size: int,
	prefix: str,
	vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
	tokens = np.zeros(array_size, dtype=np.int64)
	mask = np.zeros(array_size, dtype=np.bool_)
	for index, signature in enumerate(signatures[:array_size]):
		tokens[index] = _token_id(f"{prefix}|{signature}", vocab_size)
		mask[index] = True
	return tokens, mask


def _user_tokens_from_row(row: dict[str, Any], config: DataConfig, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
	signatures = [f"user_id|{row.get('user_id', '0')}"]
	for feature in _iter_features(row.get("user_feature")):
		signatures.append(_feature_signature("user", feature))
	return _role_token_array(signatures, config.max_feature_tokens, "user", vocab_size)


def _dense_features_from_row(row: dict[str, Any], config: DataConfig) -> np.ndarray:
	dense = np.zeros(config.dense_feature_dim, dtype=np.float32)
	groups = (
		("user", row.get("user_feature")),
		("context", row.get("context_feature")),
		("item", row.get("item_feature")),
		("cross", row.get("cross_feature")),
	)
	for group_name, payload in groups:
		for feature in _iter_features(payload):
			feature_id = feature.get("feature_id", 0)
			bucket = stable_hash64(f"dense|{group_name}|{feature_id}") % config.dense_feature_dim
			dense[bucket] += np.float32(math.tanh(_numeric_feature_value(feature) / 100.0))
			if feature.get("float_array") is not None:
				for offset, value in enumerate(_array_values(feature, "float_array")[:4]):
					dense[(bucket + offset) % config.dense_feature_dim] += np.float32(value)
	dense[-3] = np.float32(math.tanh((row.get("timestamp", 0) or 0) / 1.0e10))
	dense[-2] = np.float32(len(list(_iter_features(row.get("user_feature")))) / max(1, config.max_feature_tokens))
	dense[-1] = np.float32(len(list(_iter_features(row.get("context_feature")))) / max(1, config.max_feature_tokens))
	return dense


def _context_tokens_from_row(row: dict[str, Any], config: DataConfig, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
	timestamp = int(row.get("timestamp", 0) or 0)
	hour_bucket = (timestamp // 3600) % 24
	signatures = [f"request_hour|{hour_bucket}"]
	for feature in _iter_features(row.get("context_feature")):
		signatures.append(_feature_signature("context", feature))
	return _role_token_array(signatures, config.max_feature_tokens, "context", vocab_size)


def _candidate_tokens_from_row(
	row: dict[str, Any],
	config: DataConfig,
	vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	item_features = list(_iter_features(row.get("item_feature")))
	post_signatures = [f"item_id|{row.get('item_id', 0)}"]
	for feature in item_features[: max(0, config.max_event_features - 1)]:
		post_signatures.append(_feature_signature("item_post", feature))
	candidate_post_tokens, candidate_post_mask = _role_token_array(
		post_signatures,
		max(1, config.max_event_features),
		"candidate_post",
		vocab_size,
	)

	ranked_features = sorted(
		item_features,
		key=lambda feature: (abs(_numeric_feature_value(feature)), int(feature.get("feature_id", 0))),
		reverse=True,
	)
	author_signatures = [_feature_signature("item_author", feature) for feature in ranked_features[:AUTHOR_TOKEN_COUNT]]
	if not author_signatures:
		author_signatures = [f"item_author|missing|{row.get('item_id', 0)}"]
	candidate_author_tokens, candidate_author_mask = _role_token_array(
		author_signatures,
		AUTHOR_TOKEN_COUNT,
		"candidate_author",
		vocab_size,
	)

	composite_signature = "|".join([f"item_id|{row.get('item_id', 0)}", *post_signatures[1:], *author_signatures])
	candidate_tokens, candidate_mask = _role_token_array([composite_signature], 1, "candidate", vocab_size)
	return (
		candidate_tokens,
		candidate_mask,
		candidate_post_tokens,
		candidate_post_mask,
		candidate_author_tokens,
		candidate_author_mask,
	)


def _sequence_events_from_payload(
	sequence_name: str,
	sequence_index: int,
	sequence_payload: Any,
	config: DataConfig,
	vocab_size: int,
	reference_timestamp: int,
) -> tuple[list[_SequenceEvent], np.ndarray, np.ndarray]:
	sequence_tokens = np.zeros(config.max_seq_len, dtype=np.int64)
	sequence_mask = np.zeros(config.max_seq_len, dtype=np.bool_)
	arrays = _sequence_feature_arrays(sequence_payload)
	if not arrays:
		return [], sequence_tokens, sequence_mask

	timestamp_feature_id = _resolve_timestamp_feature_id(sequence_name, arrays)
	post_feature_id = _resolve_post_feature_id(sequence_name, arrays, timestamp_feature_id)
	author_feature_ids = _resolve_author_feature_ids(sequence_name, arrays, timestamp_feature_id, post_feature_id)
	action_feature_ids = _resolve_action_feature_ids(
		sequence_name,
		arrays,
		timestamp_feature_id,
		post_feature_id,
		author_feature_ids,
	)

	primary_values = None
	if post_feature_id is not None:
		primary_values = arrays.get(post_feature_id)
	if not primary_values and timestamp_feature_id is not None:
		primary_values = arrays.get(timestamp_feature_id)
	if not primary_values:
		first_feature_id = next(iter(arrays.keys()), None)
		primary_values = arrays.get(first_feature_id) if first_feature_id is not None else None
	if not primary_values:
		return [], sequence_tokens, sequence_mask

	event_count = len(primary_values)
	start_index = max(0, event_count - config.max_seq_len)
	events: list[_SequenceEvent] = []
	for output_index, source_index in enumerate(range(start_index, event_count)):
		post_value = _value_at(arrays.get(post_feature_id) if post_feature_id is not None else None, source_index, 0) or 0
		event_timestamp = _value_at(
			arrays.get(timestamp_feature_id) if timestamp_feature_id is not None else None,
			source_index,
			None,
		)
		author_parts = [
			f"{feature_id}:{_value_at(arrays.get(feature_id), source_index, 0) or 0}"
			for feature_id in author_feature_ids
		]
		if not author_parts:
			author_parts = [f"fallback:{post_value}"]
		action_parts = [
			f"{feature_id}:{_value_at(arrays.get(feature_id), source_index, 0) or 0}"
			for feature_id in action_feature_ids
		]
		if not action_parts:
			action_parts = [f"sequence:{sequence_name}"]
		gap_bucket = _bucket_gap(reference_timestamp, event_timestamp)
		post_signature = f"{sequence_name}|post={post_value}"
		author_signature = f"{sequence_name}|author={'|'.join(author_parts)}"
		action_signature = f"{sequence_name}|action={'|'.join(action_parts)}"
		composite_signature = f"{post_signature}|{author_signature}|{action_signature}|gap={gap_bucket}"
		sequence_tokens[output_index] = _token_id(f"sequence|{post_signature}", vocab_size)
		sequence_mask[output_index] = True
		events.append(
			_SequenceEvent(
				sequence_index=sequence_index,
				timestamp=event_timestamp,
				post_signature=post_signature,
				author_signature=author_signature,
				action_signature=action_signature,
				composite_signature=composite_signature,
				gap_bucket=gap_bucket,
			)
		)
	return events, sequence_tokens, sequence_mask


def _history_and_sequence_tokens_from_row(
	row: dict[str, Any],
	config: DataConfig,
	vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	sequence_count = len(config.sequence_names)
	history_capacity = sequence_count * config.max_seq_len
	sequence_tokens = np.zeros((sequence_count, config.max_seq_len), dtype=np.int64)
	sequence_mask = np.zeros((sequence_count, config.max_seq_len), dtype=np.bool_)
	history_tokens = np.zeros(history_capacity, dtype=np.int64)
	history_mask = np.zeros(history_capacity, dtype=np.bool_)
	history_post_tokens = np.zeros(history_capacity, dtype=np.int64)
	history_author_tokens = np.zeros(history_capacity, dtype=np.int64)
	history_action_tokens = np.zeros(history_capacity, dtype=np.int64)
	history_time_gap = np.zeros(history_capacity, dtype=np.int64)
	history_group_ids = np.zeros(history_capacity, dtype=np.int64)
	reference_timestamp = int(row.get("timestamp", 0) or 0)
	seq_feature = row.get("seq_feature") or {}
	merged_events: list[_SequenceEvent] = []

	for sequence_index, sequence_name in enumerate(config.sequence_names):
		events, sequence_row_tokens, sequence_row_mask = _sequence_events_from_payload(
			sequence_name=sequence_name,
			sequence_index=sequence_index,
			sequence_payload=(seq_feature or {}).get(sequence_name),
			config=config,
			vocab_size=vocab_size,
			reference_timestamp=reference_timestamp,
		)
		sequence_tokens[sequence_index] = sequence_row_tokens
		sequence_mask[sequence_index] = sequence_row_mask
		merged_events.extend(events)

	merged_events.sort(key=lambda event: (event.timestamp if event.timestamp is not None else -1, event.sequence_index))
	trimmed_events = merged_events[-history_capacity:]
	for history_slot, event in enumerate(trimmed_events):
		history_tokens[history_slot] = _token_id(f"history|{event.composite_signature}", vocab_size)
		history_post_tokens[history_slot] = _token_id(f"history_post|{event.post_signature}", vocab_size)
		history_author_tokens[history_slot] = _token_id(f"history_author|{event.author_signature}", vocab_size)
		history_action_tokens[history_slot] = _token_id(f"history_action|{event.action_signature}", vocab_size)
		history_time_gap[history_slot] = event.gap_bucket + 1
		history_group_ids[history_slot] = event.sequence_index + 1
		history_mask[history_slot] = True

	return (
		history_tokens,
		history_mask,
		history_post_tokens,
		history_author_tokens,
		history_action_tokens,
		history_time_gap,
		history_group_ids,
		sequence_tokens,
		sequence_mask,
	)


def _encode_row(
	row: dict[str, Any],
	config: DataConfig,
	vocab_size: int,
	item_logq_lookup: dict[int, float],
	default_item_logq: float,
) -> _EncodedSample:
	user_tokens, user_mask = _user_tokens_from_row(row, config, vocab_size)
	(
		candidate_tokens,
		candidate_mask,
		candidate_post_tokens,
		candidate_post_mask,
		candidate_author_tokens,
		candidate_author_mask,
	) = _candidate_tokens_from_row(row, config, vocab_size)
	context_tokens, context_mask = _context_tokens_from_row(row, config, vocab_size)
	(
		history_tokens,
		history_mask,
		history_post_tokens,
		history_author_tokens,
		history_action_tokens,
		history_time_gap,
		history_group_ids,
		sequence_tokens,
		sequence_mask,
	) = _history_and_sequence_tokens_from_row(
		row,
		config,
		vocab_size,
	)
	dense_features = _dense_features_from_row(row, config)
	label = 1.0 if any(int(item.get("action_type", -1)) == config.label_action_type for item in _iter_features(row.get("label"))) else 0.0
	user_index = stable_hash64(f"user|{row.get('user_id', '0')}")
	item_index = stable_hash64(f"item|{row.get('item_id', 0)}")
	item_logq = item_logq_lookup.get(item_index, default_item_logq)
	return _EncodedSample(
		user_tokens=user_tokens,
		user_mask=user_mask,
		candidate_tokens=candidate_tokens,
		candidate_mask=candidate_mask,
		candidate_post_tokens=candidate_post_tokens,
		candidate_post_mask=candidate_post_mask,
		candidate_author_tokens=candidate_author_tokens,
		candidate_author_mask=candidate_author_mask,
		context_tokens=context_tokens,
		context_mask=context_mask,
		history_tokens=history_tokens,
		history_mask=history_mask,
		history_post_tokens=history_post_tokens,
		history_author_tokens=history_author_tokens,
		history_action_tokens=history_action_tokens,
		history_time_gap=history_time_gap,
		history_group_ids=history_group_ids,
		sequence_tokens=sequence_tokens,
		sequence_mask=sequence_mask,
		dense_features=dense_features,
		label=label,
		user_index=user_index,
		item_index=item_index,
		item_logq=item_logq,
	)


def _build_item_logq_lookup(train_rows: list[dict[str, Any]]) -> tuple[dict[int, float], float]:
	counts: dict[int, int] = {}
	for row in train_rows:
		item_index = stable_hash64(f"item|{row.get('item_id', 0)}")
		counts[item_index] = counts.get(item_index, 0) + 1
	total = sum(counts.values())
	if total <= 0:
		return {}, 0.0
	lookup = {item_index: math.log(count / total) for item_index, count in counts.items()}
	return lookup, math.log(1.0 / total)


def _collate_batch(samples: list[_EncodedSample]) -> BatchTensors:
	return BatchTensors(
		candidate_tokens=torch.as_tensor(np.stack([sample.candidate_tokens for sample in samples]), dtype=torch.long),
		candidate_mask=torch.as_tensor(np.stack([sample.candidate_mask for sample in samples]), dtype=torch.bool),
		context_tokens=torch.as_tensor(np.stack([sample.context_tokens for sample in samples]), dtype=torch.long),
		context_mask=torch.as_tensor(np.stack([sample.context_mask for sample in samples]), dtype=torch.bool),
		history_tokens=torch.as_tensor(np.stack([sample.history_tokens for sample in samples]), dtype=torch.long),
		history_mask=torch.as_tensor(np.stack([sample.history_mask for sample in samples]), dtype=torch.bool),
		sequence_tokens=torch.as_tensor(np.stack([sample.sequence_tokens for sample in samples]), dtype=torch.long),
		sequence_mask=torch.as_tensor(np.stack([sample.sequence_mask for sample in samples]), dtype=torch.bool),
		dense_features=torch.as_tensor(np.stack([sample.dense_features for sample in samples]), dtype=torch.float32),
		labels=torch.as_tensor([sample.label for sample in samples], dtype=torch.float32),
		user_indices=torch.as_tensor([sample.user_index for sample in samples], dtype=torch.long),
		item_indices=torch.as_tensor([sample.item_index for sample in samples], dtype=torch.long),
		item_logq=torch.as_tensor([sample.item_logq for sample in samples], dtype=torch.float32),
		user_tokens=torch.as_tensor(np.stack([sample.user_tokens for sample in samples]), dtype=torch.long),
		user_mask=torch.as_tensor(np.stack([sample.user_mask for sample in samples]), dtype=torch.bool),
		candidate_post_tokens=torch.as_tensor(np.stack([sample.candidate_post_tokens for sample in samples]), dtype=torch.long),
		candidate_post_mask=torch.as_tensor(np.stack([sample.candidate_post_mask for sample in samples]), dtype=torch.bool),
		candidate_author_tokens=torch.as_tensor(np.stack([sample.candidate_author_tokens for sample in samples]), dtype=torch.long),
		candidate_author_mask=torch.as_tensor(np.stack([sample.candidate_author_mask for sample in samples]), dtype=torch.bool),
		history_post_tokens=torch.as_tensor(np.stack([sample.history_post_tokens for sample in samples]), dtype=torch.long),
		history_author_tokens=torch.as_tensor(np.stack([sample.history_author_tokens for sample in samples]), dtype=torch.long),
		history_action_tokens=torch.as_tensor(np.stack([sample.history_action_tokens for sample in samples]), dtype=torch.long),
		history_time_gap=torch.as_tensor(np.stack([sample.history_time_gap for sample in samples]), dtype=torch.long),
		history_group_ids=torch.as_tensor(np.stack([sample.history_group_ids for sample in samples]), dtype=torch.long),
	)


def load_dataloaders(
	config: DataConfig,
	vocab_size: int,
	batch_size: int,
	eval_batch_size: int,
	num_workers: int,
	seed: int,
) -> tuple[DataLoader[BatchTensors], DataLoader[BatchTensors], DataStats]:
	del seed
	dataset_path = _resolve_dataset_path(config.dataset_path)
	sorted_rows = _sort_rows_by_timestamp(_iter_rows(dataset_path, config.stream_batch_rows))
	train_rows, val_rows = _time_split(sorted_rows, config.val_ratio)
	item_logq_lookup, default_item_logq = _build_item_logq_lookup(train_rows)

	train_samples = [
		_encode_row(row, config, vocab_size, item_logq_lookup, default_item_logq)
		for row in train_rows
	]
	val_samples = [
		_encode_row(row, config, vocab_size, item_logq_lookup, default_item_logq)
		for row in val_rows
	]

	positive_count = sum(sample.label for sample in train_samples)
	negative_count = max(0.0, len(train_samples) - positive_count)
	pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

	stats = DataStats(
		dense_dim=config.dense_feature_dim,
		pos_weight=float(pos_weight),
		train_size=len(train_samples),
		val_size=len(val_samples),
	)

	train_loader = DataLoader(
		_EncodedDataset(train_samples),
		batch_size=max(1, batch_size),
		shuffle=False,
		num_workers=num_workers,
		collate_fn=_collate_batch,
	)
	val_loader = DataLoader(
		_EncodedDataset(val_samples),
		batch_size=max(1, eval_batch_size),
		shuffle=False,
		num_workers=num_workers,
		collate_fn=_collate_batch,
	)
	return train_loader, val_loader, stats


def build_data_pipeline(data_config, model_config, train_config):
	return load_dataloaders(
		config=data_config,
		vocab_size=model_config.vocab_size,
		batch_size=train_config.batch_size,
		eval_batch_size=train_config.resolved_eval_batch_size,
		num_workers=train_config.num_workers,
		seed=train_config.seed,
	)

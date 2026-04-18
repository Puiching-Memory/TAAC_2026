from __future__ import annotations

from functools import partial
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ...domain.config import DataConfig
from ...domain.types import BatchTensors, DataStats
from .datasets import iter_dataset_rows
from .files import stable_hash64
from .sparse_collate import build_batch_torchrec_features


DENSE_FEATURE_DIM = 16
PADDING_TOKEN_ID = 0
TIMESTAMP_FEATURE_ID = 99
TIME_GAP_BUCKET_COUNT = 64
AUTHOR_TOKEN_COUNT = 2
DOMAIN_COLUMN_PREFIXES: dict[str, str] = {
	"domain_a": "domain_a_seq_",
	"domain_b": "domain_b_seq_",
	"domain_c": "domain_c_seq_",
	"domain_d": "domain_d_seq_",
}
SEQUENCE_TIMESTAMP_FEATURE_IDS: dict[str, int] = {}
SEQUENCE_POST_FEATURE_IDS: dict[str, int] = {}
SEQUENCE_AUTHOR_FEATURE_IDS: dict[str, tuple[int, ...]] = {}
SEQUENCE_ACTION_FEATURE_IDS: dict[str, tuple[int, ...]] = {}


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


def _iter_int_features(row: dict[str, Any], prefix: str) -> Iterable[tuple[int, Any]]:
	for col_name, value in row.items():
		if not col_name.startswith(prefix):
			continue
		try:
			fid = int(col_name[len(prefix):])
		except (ValueError, TypeError):
			continue
		if value is not None:
			yield fid, value


def _iter_dense_features(row: dict[str, Any], prefix: str) -> Iterable[tuple[int, list[float]]]:
	for col_name, value in row.items():
		if not col_name.startswith(prefix):
			continue
		try:
			fid = int(col_name[len(prefix):])
		except (ValueError, TypeError):
			continue
		if value is not None and isinstance(value, list):
			yield fid, [float(v) for v in value if v is not None]


def _flat_numeric_value(value: Any) -> float:
	if value is None:
		return 0.0
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, list) and value:
		return float(np.mean([float(v) for v in value if v is not None])) if value else 0.0
	return 0.0


def _flat_feature_signature(prefix: str, fid: int, value: Any) -> str:
	if isinstance(value, list):
		if value and isinstance(value[0], float):
			vals = [f"{v:.3f}" for v in value[:4]]
			return f"{prefix}|{fid}|fa|{','.join(vals)}|n={len(value)}"
		vals = [str(int(v)) for v in value[:4]]
		return f"{prefix}|{fid}|ia|{','.join(vals)}|n={len(value)}"
	if isinstance(value, (int, np.integer)):
		return f"{prefix}|{fid}|iv|{int(value)}"
	if isinstance(value, float):
		return f"{prefix}|{fid}|fv|{value:.4f}"
	return f"{prefix}|{fid}|empty"


def _bucket_gap(reference_timestamp: int, event_timestamp: int | None) -> int:
	if event_timestamp is None:
		return 0
	gap = max(0, reference_timestamp - event_timestamp)
	if gap <= 0:
		return 0
	return min(TIME_GAP_BUCKET_COUNT - 1, int(math.log2(gap + 1)))


def _sequence_feature_arrays_from_row(row: dict[str, Any], domain_name: str) -> dict[int, list[int]]:
	prefix = DOMAIN_COLUMN_PREFIXES.get(domain_name, f"{domain_name}_seq_")
	arrays: dict[int, list[int]] = {}
	for col_name, value in row.items():
		if not col_name.startswith(prefix):
			continue
		try:
			fid = int(col_name[len(prefix):])
		except (ValueError, TypeError):
			continue
		if value is None:
			continue
		if isinstance(value, list):
			values = [int(v) for v in value if v is not None]
		else:
			values = [int(value)]
		if values:
			arrays[fid] = values
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
	for fid, value in _iter_int_features(row, "user_int_feats_"):
		signatures.append(_flat_feature_signature("user", fid, value))
	return _role_token_array(signatures, config.max_feature_tokens, "user", vocab_size)


def _dense_features_from_row(row: dict[str, Any], config: DataConfig) -> np.ndarray:
	dense = np.zeros(config.dense_feature_dim, dtype=np.float32)
	for fid, value in _iter_int_features(row, "user_int_feats_"):
		bucket = stable_hash64(f"dense|user|{fid}") % config.dense_feature_dim
		dense[bucket] += np.float32(math.tanh(_flat_numeric_value(value) / 100.0))
	for fid, float_arr in _iter_dense_features(row, "user_dense_feats_"):
		bucket = stable_hash64(f"dense|user_dense|{fid}") % config.dense_feature_dim
		for offset, val in enumerate(float_arr[:4]):
			dense[(bucket + offset) % config.dense_feature_dim] += np.float32(val)
	for fid, value in _iter_int_features(row, "item_int_feats_"):
		bucket = stable_hash64(f"dense|item|{fid}") % config.dense_feature_dim
		dense[bucket] += np.float32(math.tanh(_flat_numeric_value(value) / 100.0))
	dense[-3] = np.float32(math.tanh((row.get("timestamp", 0) or 0) / 1.0e10))
	dense[-2] = np.float32(sum(1 for _ in _iter_int_features(row, "user_int_feats_")) / max(1, config.max_feature_tokens))
	dense[-1] = np.float32(sum(1 for _ in _iter_int_features(row, "item_int_feats_")) / max(1, config.max_feature_tokens))
	return dense


def _context_tokens_from_row(row: dict[str, Any], config: DataConfig, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
	timestamp = int(row.get("timestamp", 0) or 0)
	hour_bucket = (timestamp // 3600) % 24
	signatures = [f"request_hour|{hour_bucket}"]
	for fid, value in _iter_int_features(row, "item_int_feats_"):
		signatures.append(_flat_feature_signature("context", fid, value))
	return _role_token_array(signatures, config.max_feature_tokens, "context", vocab_size)


def _candidate_tokens_from_row(
	row: dict[str, Any],
	config: DataConfig,
	vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	item_features = list(_iter_int_features(row, "item_int_feats_"))
	post_signatures = [f"item_id|{row.get('item_id', 0)}"]
	for fid, value in item_features[: max(0, config.max_event_features - 1)]:
		post_signatures.append(_flat_feature_signature("item_post", fid, value))
	candidate_post_tokens, candidate_post_mask = _role_token_array(
		post_signatures,
		max(1, config.max_event_features),
		"candidate_post",
		vocab_size,
	)

	ranked_features = sorted(
		item_features,
		key=lambda pair: (abs(_flat_numeric_value(pair[1])), pair[0]),
		reverse=True,
	)
	author_signatures = [_flat_feature_signature("item_author", fid, value) for fid, value in ranked_features[:AUTHOR_TOKEN_COUNT]]
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


def _sequence_events_from_arrays(
	sequence_name: str,
	sequence_index: int,
	arrays: dict[int, list[int]],
	config: DataConfig,
	vocab_size: int,
	reference_timestamp: int,
) -> tuple[list[_SequenceEvent], np.ndarray, np.ndarray]:
	sequence_tokens = np.zeros(config.max_seq_len, dtype=np.int64)
	sequence_mask = np.zeros(config.max_seq_len, dtype=np.bool_)
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
	merged_events: list[_SequenceEvent] = []

	for sequence_index, sequence_name in enumerate(config.sequence_names):
		arrays = _sequence_feature_arrays_from_row(row, sequence_name)
		events, sequence_row_tokens, sequence_row_mask = _sequence_events_from_arrays(
			sequence_name=sequence_name,
			sequence_index=sequence_index,
			arrays=arrays,
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
	label_type = int(row.get("label_type", 0) or 0)
	label = 1.0 if label_type == config.label_action_type else 0.0
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



def _collate_batch(samples: list[_EncodedSample], sequence_names: tuple[str, ...]) -> BatchTensors:
	candidate_tokens = torch.as_tensor(np.stack([sample.candidate_tokens for sample in samples]), dtype=torch.long)
	candidate_mask = torch.as_tensor(np.stack([sample.candidate_mask for sample in samples]), dtype=torch.bool)
	context_tokens = torch.as_tensor(np.stack([sample.context_tokens for sample in samples]), dtype=torch.long)
	context_mask = torch.as_tensor(np.stack([sample.context_mask for sample in samples]), dtype=torch.bool)
	history_tokens = torch.as_tensor(np.stack([sample.history_tokens for sample in samples]), dtype=torch.long)
	history_mask = torch.as_tensor(np.stack([sample.history_mask for sample in samples]), dtype=torch.bool)
	sequence_tokens = torch.as_tensor(np.stack([sample.sequence_tokens for sample in samples]), dtype=torch.long)
	sequence_mask = torch.as_tensor(np.stack([sample.sequence_mask for sample in samples]), dtype=torch.bool)
	dense_features = torch.as_tensor(np.stack([sample.dense_features for sample in samples]), dtype=torch.float32)
	labels = torch.as_tensor([sample.label for sample in samples], dtype=torch.float32)
	user_indices = torch.as_tensor([sample.user_index for sample in samples], dtype=torch.long)
	item_indices = torch.as_tensor([sample.item_index for sample in samples], dtype=torch.long)
	item_logq = torch.as_tensor([sample.item_logq for sample in samples], dtype=torch.float32)
	user_tokens = torch.as_tensor(np.stack([sample.user_tokens for sample in samples]), dtype=torch.long)
	user_mask = torch.as_tensor(np.stack([sample.user_mask for sample in samples]), dtype=torch.bool)
	candidate_post_tokens = torch.as_tensor(np.stack([sample.candidate_post_tokens for sample in samples]), dtype=torch.long)
	candidate_post_mask = torch.as_tensor(np.stack([sample.candidate_post_mask for sample in samples]), dtype=torch.bool)
	candidate_author_tokens = torch.as_tensor(np.stack([sample.candidate_author_tokens for sample in samples]), dtype=torch.long)
	candidate_author_mask = torch.as_tensor(np.stack([sample.candidate_author_mask for sample in samples]), dtype=torch.bool)
	history_post_tokens = torch.as_tensor(np.stack([sample.history_post_tokens for sample in samples]), dtype=torch.long)
	history_author_tokens = torch.as_tensor(np.stack([sample.history_author_tokens for sample in samples]), dtype=torch.long)
	history_action_tokens = torch.as_tensor(np.stack([sample.history_action_tokens for sample in samples]), dtype=torch.long)
	history_time_gap = torch.as_tensor(np.stack([sample.history_time_gap for sample in samples]), dtype=torch.long)
	history_group_ids = torch.as_tensor(np.stack([sample.history_group_ids for sample in samples]), dtype=torch.long)
	sparse_features, sequence_features = build_batch_torchrec_features(
		sequence_names=sequence_names,
		user_tokens=user_tokens,
		user_mask=user_mask,
		context_tokens=context_tokens,
		context_mask=context_mask,
		candidate_tokens=candidate_tokens,
		candidate_mask=candidate_mask,
		candidate_post_tokens=candidate_post_tokens,
		candidate_post_mask=candidate_post_mask,
		candidate_author_tokens=candidate_author_tokens,
		candidate_author_mask=candidate_author_mask,
		history_tokens=history_tokens,
		history_mask=history_mask,
		history_post_tokens=history_post_tokens,
		history_author_tokens=history_author_tokens,
		history_action_tokens=history_action_tokens,
		history_time_gap=history_time_gap,
		history_group_ids=history_group_ids,
		sequence_tokens=sequence_tokens,
		sequence_mask=sequence_mask,
	)
	return BatchTensors(
		candidate_tokens=candidate_tokens,
		candidate_mask=candidate_mask,
		context_tokens=context_tokens,
		context_mask=context_mask,
		history_tokens=history_tokens,
		history_mask=history_mask,
		sequence_tokens=sequence_tokens,
		sequence_mask=sequence_mask,
		dense_features=dense_features,
		labels=labels,
		user_indices=user_indices,
		item_indices=item_indices,
		item_logq=item_logq,
		user_tokens=user_tokens,
		user_mask=user_mask,
		candidate_post_tokens=candidate_post_tokens,
		candidate_post_mask=candidate_post_mask,
		candidate_author_tokens=candidate_author_tokens,
		candidate_author_mask=candidate_author_mask,
		history_post_tokens=history_post_tokens,
		history_author_tokens=history_author_tokens,
		history_action_tokens=history_action_tokens,
		history_time_gap=history_time_gap,
		history_group_ids=history_group_ids,
		sparse_features=sparse_features,
		sequence_features=sequence_features,
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
	sorted_rows = _sort_rows_by_timestamp(iter_dataset_rows(config.dataset_path))
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
	collate_batch = partial(_collate_batch, sequence_names=tuple(config.sequence_names))

	train_loader = DataLoader(
		_EncodedDataset(train_samples),
		batch_size=max(1, batch_size),
		shuffle=False,
		num_workers=num_workers,
		collate_fn=collate_batch,
	)
	val_loader = DataLoader(
		_EncodedDataset(val_samples),
		batch_size=max(1, eval_batch_size),
		shuffle=False,
		num_workers=num_workers,
		collate_fn=collate_batch,
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


__all__ = [
	"AUTHOR_TOKEN_COUNT",
	"DENSE_FEATURE_DIM",
	"DOMAIN_COLUMN_PREFIXES",
	"PADDING_TOKEN_ID",
	"SEQUENCE_ACTION_FEATURE_IDS",
	"SEQUENCE_AUTHOR_FEATURE_IDS",
	"SEQUENCE_POST_FEATURE_IDS",
	"SEQUENCE_TIMESTAMP_FEATURE_IDS",
	"TIME_GAP_BUCKET_COUNT",
	"TIMESTAMP_FEATURE_ID",
	"build_data_pipeline",
	"load_dataloaders",
]
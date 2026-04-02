from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_len(value: Any) -> int:
    return 0 if value is None else len(value)


def _summarize_numeric(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "count": float(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _label_from_entries(entries: Any, label_action_type: int) -> float:
    labels = entries if entries is not None else []
    return 1.0 if any(int(entry["action_type"]) == label_action_type for entry in labels) else 0.0


def _sequence_group_profile(features: Any) -> tuple[int, int | None]:
    feature_list = features if features is not None else []
    arrays: list[np.ndarray] = []
    timestamp_candidates: list[np.ndarray] = []

    for feature in feature_list:
        values = feature.get("int_array")
        if values is None:
            continue
        array = np.asarray(values)
        arrays.append(array)
        if array.size > 0 and float(np.median(array)) > 1_000_000_000:
            timestamp_candidates.append(array)

    if not arrays:
        return 0, None

    event_count = min(int(array.size) for array in arrays)
    if event_count == 0 or not timestamp_candidates:
        return event_count, None

    oldest_timestamp = min(int(array[:event_count].min()) for array in timestamp_candidates if array.size > 0)
    return event_count, oldest_timestamp


def _quantile_windows(
    timestamps: np.ndarray,
    labels: np.ndarray,
    total_events: np.ndarray,
    selected_events: np.ndarray,
    active_span_hours: np.ndarray,
    behavior_density: np.ndarray,
    user_feature_counts: np.ndarray,
    item_feature_counts: np.ndarray,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    if timestamps.size == 0:
        return []

    quantiles = np.quantile(timestamps, np.linspace(0.0, 1.0, 5))
    edges = [float(quantiles[0])]
    for edge in quantiles[1:]:
        edge_value = float(edge)
        if edge_value > edges[-1]:
            edges.append(edge_value)

    if len(edges) == 1:
        masks = [("all", timestamps >= edges[0], edges[0], edges[0])]
    else:
        masks = []
        total_segments = len(edges) - 1
        for index, (lower_bound, upper_bound) in enumerate(zip(edges, edges[1:]), start=1):
            if index == total_segments:
                mask = (timestamps >= lower_bound) & (timestamps <= upper_bound)
            else:
                mask = (timestamps >= lower_bound) & (timestamps < upper_bound)
            masks.append((f"q{index}", mask, lower_bound, upper_bound))

    results: list[dict[str, Any]] = []
    for bucket_name, mask, lower_bound, upper_bound in masks:
        if not np.any(mask):
            continue
        results.append(
            {
                "bucket": bucket_name,
                "timestamp_min": int(timestamps[mask].min()),
                "timestamp_max": int(timestamps[mask].max()),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "rows": int(mask.sum()),
                "positive_rate": float(labels[mask].mean()),
                "mean_total_event_count": float(total_events[mask].mean()),
                "mean_selected_event_count": float(selected_events[mask].mean()),
                "mean_active_span_hours": float(active_span_hours[mask].mean()),
                "mean_behavior_density": float(behavior_density[mask].mean()),
                "mean_user_feature_count": float(user_feature_counts[mask].mean()),
                "mean_item_feature_count": float(item_feature_counts[mask].mean()),
                "truncation_rate": float((total_events[mask] > max_seq_len).mean()),
            }
        )
    return results


def _frequency_bucket_summary(
    row_frequencies: np.ndarray,
    entity_frequencies: np.ndarray,
    labels: np.ndarray,
    total_events: np.ndarray,
    behavior_density: np.ndarray,
) -> list[dict[str, Any]]:
    bucket_specs = [
        ("1", 1, 1),
        ("2-4", 2, 4),
        ("5-9", 5, 9),
        ("10-49", 10, 49),
        ("50+", 50, None),
    ]
    results: list[dict[str, Any]] = []
    total_rows = max(int(row_frequencies.size), 1)

    for bucket_name, lower_bound, upper_bound in bucket_specs:
        if upper_bound is None:
            row_mask = row_frequencies >= lower_bound
            entity_mask = entity_frequencies >= lower_bound
        else:
            row_mask = (row_frequencies >= lower_bound) & (row_frequencies <= upper_bound)
            entity_mask = (entity_frequencies >= lower_bound) & (entity_frequencies <= upper_bound)
        if not np.any(row_mask):
            continue
        results.append(
            {
                "bucket": bucket_name,
                "rows": int(row_mask.sum()),
                "row_rate": float(row_mask.mean()),
                "entity_count": int(entity_mask.sum()),
                "positive_rate": float(labels[row_mask].mean()),
                "mean_total_event_count": float(total_events[row_mask].mean()),
                "mean_behavior_density": float(behavior_density[row_mask].mean()),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )

    if not results:
        return [
            {
                "bucket": "all",
                "rows": total_rows,
                "row_rate": 1.0,
                "entity_count": int(entity_frequencies.size),
                "positive_rate": float(labels.mean()) if labels.size else 0.0,
                "mean_total_event_count": float(total_events.mean()) if total_events.size else 0.0,
                "mean_behavior_density": float(behavior_density.mean()) if behavior_density.size else 0.0,
                "lower_bound": 0,
                "upper_bound": None,
            }
        ]
    return results


def build_dataset_profile_artifacts(
    dataset_path: str | Path,
    label_action_type: int = 2,
    max_seq_len: int = 256,
) -> dict[str, Any]:
    dataframe = pd.read_parquet(Path(dataset_path))

    timestamps: list[int] = []
    labels: list[float] = []
    total_event_counts: list[float] = []
    selected_event_counts: list[float] = []
    active_span_hours: list[float] = []
    behavior_density: list[float] = []
    non_empty_group_counts: list[float] = []
    user_feature_counts: list[float] = []
    item_feature_counts: list[float] = []
    per_group_lengths: dict[str, list[float]] = {}
    per_group_non_empty_rows: Counter[str] = Counter()
    per_group_timestamp_rows: Counter[str] = Counter()
    hour_of_day: Counter[int] = Counter()

    for row in dataframe.itertuples(index=False):
        row_timestamp = int(row.timestamp)
        timestamps.append(row_timestamp)
        hour_of_day[int((row_timestamp // 3600) % 24)] += 1
        labels.append(_label_from_entries(row.label, label_action_type))

        total_events = 0
        non_empty_groups = 0
        oldest_timestamps: list[int] = []
        sequence_groups = row.seq_feature if row.seq_feature is not None else {}

        for group_name, features in sequence_groups.items():
            group_length, oldest_timestamp = _sequence_group_profile(features)
            per_group_lengths.setdefault(group_name, []).append(float(group_length))
            total_events += group_length
            if group_length > 0:
                non_empty_groups += 1
                per_group_non_empty_rows[group_name] += 1
            if oldest_timestamp is not None:
                oldest_timestamps.append(oldest_timestamp)
                per_group_timestamp_rows[group_name] += 1

        span_hours = 0.0
        if oldest_timestamps:
            span_hours = max(row_timestamp - min(oldest_timestamps), 0) / 3600.0

        total_event_counts.append(float(total_events))
        selected_event_counts.append(float(min(total_events, max_seq_len)))
        active_span_hours.append(float(span_hours))
        behavior_density.append(float(total_events / max(span_hours, 1.0)) if total_events > 0 else 0.0)
        non_empty_group_counts.append(float(non_empty_groups))
        user_feature_counts.append(float(_safe_len(row.user_feature)))
        item_feature_counts.append(float(_safe_len(row.item_feature)))

    rows = int(dataframe.shape[0])
    labels_np = np.asarray(labels, dtype=np.float64)
    timestamps_np = np.asarray(timestamps, dtype=np.float64)
    total_events_np = np.asarray(total_event_counts, dtype=np.float64)
    selected_events_np = np.asarray(selected_event_counts, dtype=np.float64)
    active_span_hours_np = np.asarray(active_span_hours, dtype=np.float64)
    behavior_density_np = np.asarray(behavior_density, dtype=np.float64)
    user_feature_counts_np = np.asarray(user_feature_counts, dtype=np.float64)
    item_feature_counts_np = np.asarray(item_feature_counts, dtype=np.float64)

    user_frequency = dataframe["user_id"].astype(str).value_counts()
    item_frequency = dataframe["item_id"].astype(str).value_counts()
    row_user_frequency = dataframe["user_id"].astype(str).map(user_frequency).to_numpy(dtype=np.int64)
    row_item_frequency = dataframe["item_id"].astype(str).map(item_frequency).to_numpy(dtype=np.int64)

    profile = {
        "dataset": {
            "path": str(dataset_path),
            "rows": rows,
            "label_action_type": int(label_action_type),
            "max_seq_len": int(max_seq_len),
            "timestamp_min": int(min(timestamps)) if timestamps else 0,
            "timestamp_max": int(max(timestamps)) if timestamps else 0,
        },
        "label_summary": {
            "positive_count": int(labels_np.sum()) if labels_np.size else 0,
            "negative_count": int(rows - int(labels_np.sum())) if labels_np.size else 0,
            "positive_rate": float(labels_np.mean()) if labels_np.size else 0.0,
        },
        "sequence_profile": {
            "total_event_count": _summarize_numeric(total_event_counts),
            "selected_event_count": _summarize_numeric(selected_event_counts),
            "active_span_hours": _summarize_numeric(active_span_hours),
            "behavior_density": _summarize_numeric(behavior_density),
            "non_empty_group_count": _summarize_numeric(non_empty_group_counts),
            "rows_exceeding_max_seq_len": int((total_events_np > max_seq_len).sum()) if total_events_np.size else 0,
            "truncation_rate": float((total_events_np > max_seq_len).mean()) if total_events_np.size else 0.0,
            "per_group": {
                group_name: {
                    "length": _summarize_numeric(lengths),
                    "non_empty_rate": float(per_group_non_empty_rows[group_name] / max(rows, 1)),
                    "timestamp_coverage_rate": float(
                        per_group_timestamp_rows[group_name] / max(per_group_non_empty_rows[group_name], 1)
                    ),
                }
                for group_name, lengths in sorted(per_group_lengths.items())
            },
        },
        "temporal_drift": {
            "time_windows": _quantile_windows(
                timestamps_np,
                labels_np,
                total_events_np,
                selected_events_np,
                active_span_hours_np,
                behavior_density_np,
                user_feature_counts_np,
                item_feature_counts_np,
                max_seq_len,
            ),
            "hour_of_day_counts": {str(hour): int(count) for hour, count in sorted(hour_of_day.items())},
        },
        "hot_cold_profile": {
            "user_frequency": _summarize_numeric(user_frequency.to_numpy(dtype=np.float64).tolist()),
            "item_frequency": _summarize_numeric(item_frequency.to_numpy(dtype=np.float64).tolist()),
            "user_buckets": _frequency_bucket_summary(
                row_user_frequency,
                user_frequency.to_numpy(dtype=np.int64),
                labels_np,
                total_events_np,
                behavior_density_np,
            ),
            "item_buckets": _frequency_bucket_summary(
                row_item_frequency,
                item_frequency.to_numpy(dtype=np.int64),
                labels_np,
                total_events_np,
                behavior_density_np,
            ),
        },
    }
    return profile


def print_dataset_profile_summary(profile: dict[str, Any]) -> None:
    dataset = profile["dataset"]
    labels = profile["label_summary"]
    sequence = profile["sequence_profile"]
    print(f"rows={dataset['rows']} positive_rate={labels['positive_rate']:.4f} max_seq_len={dataset['max_seq_len']}")
    print(
        f"total_event_count_p50={sequence['total_event_count']['p50']:.2f} total_event_count_p95={sequence['total_event_count']['p95']:.2f} truncation_rate={sequence['truncation_rate']:.4f}"
    )
    for bucket in profile["temporal_drift"]["time_windows"]:
        print(
            f"time_window={bucket['bucket']} positive_rate={bucket['positive_rate']:.4f} mean_total_event_count={bucket['mean_total_event_count']:.2f} mean_behavior_density={bucket['mean_behavior_density']:.2f}"
        )


__all__ = ["build_dataset_profile_artifacts", "print_dataset_profile_summary"]
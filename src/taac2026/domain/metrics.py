"""Dependency-light metrics for validation and smoke tests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def safe_mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        return 0.0
    return float(sum(materialized) / len(materialized))


def percentile(values: Iterable[float], q: float) -> float:
    materialized = np.asarray(list(values), dtype=np.float64)
    if materialized.size == 0:
        return 0.0
    return float(np.percentile(materialized, q))


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels_array = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid_mask = np.isfinite(labels_array) & np.isfinite(scores_array)
    labels_array = labels_array[valid_mask]
    scores_array = scores_array[valid_mask]
    if labels_array.size == 0:
        return 0.5

    positives = labels_array > 0.5
    positive_count = int(positives.sum())
    negative_count = int(labels_array.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return 0.5

    order = np.argsort(scores_array, kind="mergesort")
    sorted_scores = scores_array[order]
    ranks = np.empty(labels_array.size, dtype=np.float64)
    position = 0
    while position < sorted_scores.size:
        next_position = position + 1
        while next_position < sorted_scores.size and sorted_scores[next_position] == sorted_scores[position]:
            next_position += 1
        average_rank = (position + 1 + next_position) / 2.0
        ranks[order[position:next_position]] = average_rank
        position = next_position

    positive_rank_sum = float(ranks[positives].sum())
    numerator = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(numerator / (positive_count * negative_count))


def binary_logloss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels_array = np.asarray(labels, dtype=np.float64).reshape(-1)
    probability_array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    valid_mask = np.isfinite(labels_array) & np.isfinite(probability_array)
    labels_array = labels_array[valid_mask]
    probability_array = np.clip(probability_array[valid_mask], 1.0e-7, 1.0 - 1.0e-7)
    if labels_array.size == 0:
        return 0.0
    loss = -(labels_array * np.log(probability_array) + (1.0 - labels_array) * np.log(1.0 - probability_array))
    return float(loss.mean())


def group_auc(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    labels_array = np.asarray(labels).reshape(-1)
    scores_array = np.asarray(scores).reshape(-1)
    groups_array = np.asarray(groups).reshape(-1)
    grouped_indices: dict[object, list[int]] = defaultdict(list)
    for index, group_value in enumerate(groups_array.tolist()):
        grouped_indices[group_value].append(index)

    auc_values: list[float] = []
    covered_samples = 0
    for indices in grouped_indices.values():
        group_labels = labels_array[indices]
        if len(np.unique(group_labels)) < 2:
            continue
        auc_values.append(binary_auc(group_labels, scores_array[indices]))
        covered_samples += len(indices)

    return {
        "value": safe_mean(auc_values) if auc_values else 0.5,
        "coverage": float(covered_samples / labels_array.size) if labels_array.size else 0.0,
    }


def compute_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict[str, object]:
    labels_array = np.asarray(labels, dtype=np.float64).reshape(-1)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    probabilities = score_array
    if probabilities.size and (probabilities.min() < 0.0 or probabilities.max() > 1.0):
        probabilities = sigmoid(probabilities)
    if groups is None:
        groups_array = np.arange(labels_array.size, dtype=np.int64)
    else:
        groups_array = np.asarray(groups).reshape(-1)
    brier = float(np.mean((probabilities - labels_array) ** 2)) if labels_array.size else 0.0
    return {
        "auc": binary_auc(labels_array, probabilities),
        "logloss": binary_logloss(labels_array, probabilities),
        "brier": brier,
        "gauc": group_auc(labels_array, probabilities, groups_array),
        "sample_count": int(labels_array.size),
    }

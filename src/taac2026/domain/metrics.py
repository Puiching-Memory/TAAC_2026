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


def binary_score_diagnostics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    labels_array = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid_mask = np.isfinite(labels_array) & np.isfinite(scores_array)
    labels_array = labels_array[valid_mask]
    scores_array = scores_array[valid_mask]
    sample_count = int(labels_array.size)
    positive_scores = scores_array[labels_array > 0.5]
    negative_scores = scores_array[labels_array <= 0.5]
    positive_count = int(positive_scores.size)
    negative_count = int(negative_scores.size)
    positive_mean = safe_mean(positive_scores)
    negative_mean = safe_mean(negative_scores)
    return {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "invalid_count": int(valid_mask.size - valid_mask.sum()),
        "positive_rate": float(positive_count / sample_count) if sample_count else 0.0,
        "score_mean": safe_mean(scores_array),
        "score_std": float(np.std(scores_array)) if sample_count else 0.0,
        "positive_score_mean": positive_mean,
        "negative_score_mean": negative_mean,
        "score_margin_mean": positive_mean - negative_mean if positive_count and negative_count else 0.0,
        "positive_score_p10": percentile(positive_scores, 10),
        "positive_score_p50": percentile(positive_scores, 50),
        "positive_score_p90": percentile(positive_scores, 90),
        "negative_score_p10": percentile(negative_scores, 10),
        "negative_score_p50": percentile(negative_scores, 50),
        "negative_score_p90": percentile(negative_scores, 90),
    }


def binary_auc_bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    samples: int = 200,
    seed: int = 42,
    confidence: float = 0.95,
    max_resample_size: int = 50_000,
) -> dict[str, float | int]:
    labels_array = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid_mask = np.isfinite(labels_array) & np.isfinite(scores_array)
    labels_array = labels_array[valid_mask]
    scores_array = scores_array[valid_mask]
    auc = binary_auc(labels_array, scores_array)
    positive_indices = np.flatnonzero(labels_array > 0.5)
    negative_indices = np.flatnonzero(labels_array <= 0.5)
    positive_count = int(positive_indices.size)
    negative_count = int(negative_indices.size)
    if samples <= 0 or positive_count == 0 or negative_count == 0:
        return {
            "samples": 0,
            "confidence": float(confidence),
            "low": auc,
            "high": auc,
            "std": 0.0,
            "positive_resample_count": positive_count,
            "negative_resample_count": negative_count,
        }

    total_count = positive_count + negative_count
    if max_resample_size > 0 and total_count > max_resample_size:
        positive_draw_count = max(1, round(max_resample_size * positive_count / total_count))
        negative_draw_count = max(1, max_resample_size - positive_draw_count)
    else:
        positive_draw_count = positive_count
        negative_draw_count = negative_count

    generator = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        sampled_positive = generator.choice(positive_indices, size=positive_draw_count, replace=True)
        sampled_negative = generator.choice(negative_indices, size=negative_draw_count, replace=True)
        sampled_indices = np.concatenate([sampled_positive, sampled_negative])
        values[sample_index] = binary_auc(labels_array[sampled_indices], scores_array[sampled_indices])

    tail = (1.0 - confidence) / 2.0
    return {
        "samples": int(samples),
        "confidence": float(confidence),
        "low": float(np.percentile(values, tail * 100.0)),
        "high": float(np.percentile(values, (1.0 - tail) * 100.0)),
        "std": float(np.std(values)),
        "positive_resample_count": int(positive_draw_count),
        "negative_resample_count": int(negative_draw_count),
    }


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
    *,
    auc_bootstrap_samples: int = 200,
    auc_bootstrap_seed: int = 42,
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
        "auc_ci": binary_auc_bootstrap_ci(
            labels_array,
            probabilities,
            samples=auc_bootstrap_samples,
            seed=auc_bootstrap_seed,
        ),
        "logloss": binary_logloss(labels_array, probabilities),
        "brier": brier,
        "gauc": group_auc(labels_array, probabilities, groups_array),
        "score_diagnostics": binary_score_diagnostics(labels_array, probabilities),
        "sample_count": int(labels_array.size),
    }

from __future__ import annotations

from collections import defaultdict

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.float64)
    scores = scores.astype(np.float64)
    positive_scores = scores[labels > 0.5]
    negative_scores = scores[labels <= 0.5]
    if positive_scores.size == 0 or negative_scores.size == 0:
        return 0.5
    margins = positive_scores[:, None] - negative_scores[None, :]
    wins = np.mean(margins > 0)
    ties = np.mean(margins == 0)
    return float(wins + 0.5 * ties)


def binary_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.float64)
    scores = scores.astype(np.float64)
    positive_total = np.sum(labels > 0.5)
    if positive_total == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(1.0 - sorted_labels)
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = true_positive / positive_total
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapezoid(precision, recall))


def binary_brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if labels.size == 0 or probabilities.size == 0:
        return 0.0
    return float(np.mean(np.square(probabilities - labels.astype(np.float64))))


def binary_logloss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if labels.size == 0 or probabilities.size == 0:
        return 0.0
    clipped = np.clip(probabilities.astype(np.float64), 1.0e-7, 1.0 - 1.0e-7)
    labels = labels.astype(np.float64)
    losses = -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    return float(np.mean(losses))


def group_auc(labels: np.ndarray, scores: np.ndarray, group_ids: np.ndarray) -> dict[str, float]:
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for label, score, group_id in zip(labels, scores, group_ids, strict=False):
        grouped[int(group_id)].append((float(label), float(score)))
    covered_groups = 0
    weighted_auc_sum = 0.0
    weighted_count = 0
    for rows in grouped.values():
        group_labels = np.asarray([row[0] for row in rows], dtype=np.float64)
        group_scores = np.asarray([row[1] for row in rows], dtype=np.float64)
        if np.sum(group_labels > 0.5) == 0 or np.sum(group_labels <= 0.5) == 0:
            continue
        covered_groups += 1
        weighted_auc_sum += binary_auc(group_labels, group_scores) * len(rows)
        weighted_count += len(rows)
    coverage = covered_groups / max(len(grouped), 1)
    value = weighted_auc_sum / weighted_count if weighted_count else 0.5
    return {"value": float(value), "coverage": float(coverage)}


def compute_classification_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    group_ids: np.ndarray,
) -> dict[str, float | dict[str, float]]:
    probabilities = sigmoid(logits)
    return {
        "auc": binary_auc(labels, logits),
        "pr_auc": binary_pr_auc(labels, logits),
        "brier": binary_brier(labels, probabilities),
        "logloss": binary_logloss(labels, probabilities),
        "gauc": group_auc(labels, logits, group_ids),
    }


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


__all__ = [
    "binary_auc",
    "binary_brier",
    "binary_logloss",
    "binary_pr_auc",
    "compute_classification_metrics",
    "group_auc",
    "percentile",
    "safe_mean",
    "sigmoid",
]

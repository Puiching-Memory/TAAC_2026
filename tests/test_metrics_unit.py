from __future__ import annotations

import random

import numpy as np
import pytest

from taac2026.domain.metrics import (
    binary_auc,
    binary_brier,
    binary_logloss,
    binary_pr_auc,
    compute_classification_metrics,
    group_auc,
    percentile,
    safe_mean,
    sigmoid,
)


pytestmark = pytest.mark.unit


def test_binary_auc_returns_default_for_single_class_inputs() -> None:
    assert binary_auc(np.asarray([1.0, 1.0]), np.asarray([0.2, 0.8])) == 0.5
    assert binary_auc(np.asarray([0.0, 0.0]), np.asarray([0.2, 0.8])) == 0.5


def test_binary_auc_counts_ties_as_half_wins() -> None:
    labels = np.asarray([1.0, 0.0], dtype=np.float64)
    scores = np.asarray([0.5, 0.5], dtype=np.float64)

    assert binary_auc(labels, scores) == 0.5


def test_empty_metric_inputs_return_stable_defaults() -> None:
    labels = np.asarray([], dtype=np.float64)
    scores = np.asarray([], dtype=np.float64)

    metrics = compute_classification_metrics(labels, scores, labels.astype(np.int64))

    assert metrics["auc"] == 0.5
    assert metrics["pr_auc"] == 0.0
    assert metrics["brier"] == 0.0
    assert metrics["logloss"] == 0.0
    assert metrics["gauc"] == {"value": 0.5, "coverage": 0.0}


def test_group_auc_reports_partial_group_coverage() -> None:
    labels = np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    scores = np.asarray([0.9, 0.1, 0.8, 0.7], dtype=np.float64)
    groups = np.asarray([1, 1, 2, 2], dtype=np.int64)

    result = group_auc(labels, scores, groups)

    assert result["value"] == 1.0
    assert result["coverage"] == 0.5


def test_numeric_helpers_handle_empty_inputs() -> None:
    assert safe_mean([]) == 0.0
    assert percentile([], 95.0) == 0.0
    assert binary_brier(np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)) == 0.0
    assert binary_logloss(np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)) == 0.0


def test_randomized_metric_invariants_hold_under_permutation() -> None:
    rng = random.Random(7)
    for _ in range(20):
        sample_count = rng.randint(4, 12)
        labels = np.asarray([rng.randint(0, 1) for _ in range(sample_count)], dtype=np.float64)
        scores = np.asarray([rng.uniform(-3.0, 3.0) for _ in range(sample_count)], dtype=np.float64)
        groups = np.asarray([rng.randint(1, 3) for _ in range(sample_count)], dtype=np.int64)

        metrics = compute_classification_metrics(labels, scores, groups)
        order = np.asarray(rng.sample(range(sample_count), sample_count), dtype=np.int64)
        shuffled = compute_classification_metrics(labels[order], scores[order], groups[order])

        assert metrics["auc"] == pytest.approx(shuffled["auc"])
        assert metrics["pr_auc"] == pytest.approx(shuffled["pr_auc"])
        assert metrics["brier"] == pytest.approx(shuffled["brier"])
        assert metrics["logloss"] == pytest.approx(shuffled["logloss"])
        assert metrics["gauc"]["value"] == pytest.approx(shuffled["gauc"]["value"])
        assert metrics["gauc"]["coverage"] == pytest.approx(shuffled["gauc"]["coverage"])
        assert 0.0 <= float(metrics["auc"]) <= 1.0
        assert 0.0 <= float(metrics["pr_auc"]) <= 1.0


def test_sigmoid_is_bounded_and_pr_auc_degenerates_without_positives() -> None:
    probabilities = sigmoid(np.asarray([-1_000.0, 0.0, 1_000.0], dtype=np.float64))

    assert 0.0 <= probabilities.min() < probabilities.max() <= 1.0
    assert binary_pr_auc(np.asarray([0.0, 0.0]), np.asarray([0.2, 0.8])) == 0.0

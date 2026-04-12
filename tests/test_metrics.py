from __future__ import annotations

import numpy as np
import pytest

from taac2026.domain.metrics import (
    binary_auc,
    compute_classification_metrics,
    group_auc,
    percentile,
    safe_mean,
)


def test_binary_auc_returns_half_for_single_class_labels() -> None:
    labels = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    scores = np.asarray([0.8, 0.2, 0.4], dtype=np.float32)

    assert binary_auc(labels, scores) == pytest.approx(0.5)
    assert binary_auc(1.0 - labels, scores) == pytest.approx(0.5)


def test_binary_auc_counts_ties_as_half_credit() -> None:
    labels = np.asarray([1.0, 0.0], dtype=np.float32)
    scores = np.asarray([0.3, 0.3], dtype=np.float32)

    assert binary_auc(labels, scores) == pytest.approx(0.5)


def test_group_auc_skips_single_class_groups_and_reports_coverage() -> None:
    result = group_auc(
        np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
        np.asarray([0.9, 0.1, 0.3, 0.2], dtype=np.float32),
        np.asarray([10, 10, 20, 30], dtype=np.int64),
    )

    assert result["value"] == pytest.approx(1.0)
    assert result["coverage"] == pytest.approx(1.0 / 3.0)


def test_compute_classification_metrics_handles_empty_inputs() -> None:
    metrics = compute_classification_metrics(
        np.asarray([], dtype=np.float32),
        np.asarray([], dtype=np.float32),
        np.asarray([], dtype=np.int64),
    )

    assert metrics["auc"] == pytest.approx(0.5)
    assert metrics["pr_auc"] == pytest.approx(0.0)
    assert metrics["brier"] == pytest.approx(0.0)
    assert metrics["logloss"] == pytest.approx(0.0)
    assert metrics["gauc"] == {"value": 0.5, "coverage": 0.0}


def test_safe_mean_and_percentile_cover_empty_and_boundary_inputs() -> None:
    assert safe_mean([]) == pytest.approx(0.0)
    assert percentile([], 50.0) == pytest.approx(0.0)
    assert percentile([1.0, 2.0, 3.0], 0.0) == pytest.approx(1.0)
    assert percentile([1.0, 2.0, 3.0], 100.0) == pytest.approx(3.0)


def test_classification_metrics_randomized_checks_stay_finite() -> None:
    rng = np.random.default_rng(7)

    for _ in range(24):
        labels = rng.integers(0, 2, size=32, dtype=np.int64).astype(np.float32)
        logits = rng.normal(size=32).astype(np.float32)
        groups = rng.integers(0, 6, size=32, dtype=np.int64)
        metrics = compute_classification_metrics(labels, logits, groups)

        assert 0.0 <= float(metrics["auc"]) <= 1.0
        assert 0.0 <= float(metrics["pr_auc"]) <= 1.0
        assert float(metrics["brier"]) >= 0.0
        assert float(metrics["logloss"]) >= 0.0
        assert 0.0 <= float(metrics["gauc"]["value"]) <= 1.0
        assert 0.0 <= float(metrics["gauc"]["coverage"]) <= 1.0

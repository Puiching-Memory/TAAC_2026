from __future__ import annotations

import numpy as np
import pytest

from taac2026.domain.metrics import binary_auc, binary_logloss, compute_classification_metrics


def test_binary_auc_counts_ties_as_half_credit() -> None:
    labels = np.asarray([1.0, 0.0], dtype=np.float32)
    scores = np.asarray([0.3, 0.3], dtype=np.float32)

    assert binary_auc(labels, scores) == pytest.approx(0.5)


def test_binary_auc_returns_half_for_single_class() -> None:
    labels = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    scores = np.asarray([0.2, 0.4, 0.8], dtype=np.float32)

    assert binary_auc(labels, scores) == pytest.approx(0.5)


def test_binary_logloss_stays_finite_for_extreme_probabilities() -> None:
    labels = np.asarray([1.0, 0.0], dtype=np.float32)
    scores = np.asarray([1.0, 0.0], dtype=np.float32)

    assert binary_logloss(labels, scores) >= 0.0


def test_classification_metrics_accepts_logits() -> None:
    labels = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    logits = np.asarray([-5.0, 5.0, 2.0, -1.0], dtype=np.float32)

    metrics = compute_classification_metrics(labels, logits)

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["sample_count"] == 4

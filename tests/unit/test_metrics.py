from __future__ import annotations

import numpy as np
import pytest

from taac2026.domain.metrics import binary_auc, binary_auc_bootstrap_ci, binary_logloss, binary_score_diagnostics, compute_classification_metrics


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
    assert metrics["auc_ci"]["low"] <= metrics["auc"] <= metrics["auc_ci"]["high"]
    assert metrics["sample_count"] == 4
    assert metrics["score_diagnostics"]["positive_score_mean"] > metrics["score_diagnostics"]["negative_score_mean"]


def test_binary_auc_bootstrap_ci_is_deterministic() -> None:
    labels = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    scores = np.asarray([0.1, 0.3, 0.7, 0.9], dtype=np.float32)

    first = binary_auc_bootstrap_ci(labels, scores, samples=64, seed=7)
    second = binary_auc_bootstrap_ci(labels, scores, samples=64, seed=7)

    assert first == second
    assert first["low"] <= 1.0 <= first["high"]


def test_binary_score_diagnostics_reports_class_margins() -> None:
    labels = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    scores = np.asarray([0.1, 0.9, 0.7, 0.2], dtype=np.float32)

    diagnostics = binary_score_diagnostics(labels, scores)

    assert diagnostics["sample_count"] == 4
    assert diagnostics["positive_count"] == 2
    assert diagnostics["negative_count"] == 2
    assert diagnostics["positive_rate"] == pytest.approx(0.5)
    assert diagnostics["positive_score_mean"] == pytest.approx(0.8)
    assert diagnostics["negative_score_mean"] == pytest.approx(0.15)
    assert diagnostics["score_margin_mean"] == pytest.approx(0.65)

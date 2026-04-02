from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _binary_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    auc_default: float = 0.5,
) -> dict[str, Any]:
    count = int(labels.size)
    if count == 0:
        return {
            "count": 0,
            "positive_count": 0,
            "positive_rate": 0.0,
            "auc": float(auc_default),
            "auc_defined": False,
            "pr_auc": 0.0,
            "pr_auc_defined": False,
            "brier": 0.0,
            "logloss": 0.0,
            "prediction_mean": 0.0,
            "prediction_std": 0.0,
            "prediction_min": 0.0,
            "prediction_max": 0.0,
        }

    positive_count = int(labels.sum())
    positive_rate = float(labels.mean())
    has_both_classes = 0 < positive_count < count

    if has_both_classes:
        auc = float(roc_auc_score(labels, predictions))
    else:
        auc = float(auc_default)

    if positive_count == 0:
        pr_auc = 0.0
        pr_auc_defined = False
    elif positive_count == count:
        pr_auc = 1.0
        pr_auc_defined = False
    else:
        pr_auc = float(average_precision_score(labels, predictions))
        pr_auc_defined = True

    clipped_predictions = np.clip(predictions, 1e-6, 1.0 - 1e-6)
    return {
        "count": count,
        "positive_count": positive_count,
        "positive_rate": positive_rate,
        "auc": auc,
        "auc_defined": has_both_classes,
        "pr_auc": pr_auc,
        "pr_auc_defined": pr_auc_defined,
        "brier": float(np.mean((predictions - labels) ** 2)),
        "logloss": float(log_loss(labels, clipped_predictions, labels=[0.0, 1.0])),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std()),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
    }


def _bucket_result(bucket_name: str, values: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    metrics = _binary_classification_metrics(labels, predictions)
    metrics.update(
        {
            "bucket": bucket_name,
            "value_min": float(values.min()) if values.size else 0.0,
            "value_max": float(values.max()) if values.size else 0.0,
        }
    )
    return metrics


def _bootstrap_metric_interval(
    labels: np.ndarray,
    predictions: np.ndarray,
    metric_name: str,
    sample_count: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    if labels.size == 0:
        return {
            "metric": metric_name,
            "defined": False,
            "bootstrap_samples": 0,
            "lower": 0.0,
            "upper": 0.0,
            "mean": 0.0,
        }

    rng = np.random.default_rng(seed)
    values: list[float] = []
    indices = np.arange(labels.size)
    for _ in range(sample_count):
        sampled_indices = rng.choice(indices, size=indices.size, replace=True)
        sampled_labels = labels[sampled_indices]
        sampled_predictions = predictions[sampled_indices]

        positive_count = int(sampled_labels.sum())
        count = int(sampled_labels.size)
        if metric_name == "auc":
            if positive_count == 0 or positive_count == count:
                continue
            values.append(float(roc_auc_score(sampled_labels, sampled_predictions)))
            continue

        if metric_name == "pr_auc":
            if positive_count == 0:
                values.append(0.0)
            elif positive_count == count:
                values.append(1.0)
            else:
                values.append(float(average_precision_score(sampled_labels, sampled_predictions)))
            continue

        raise ValueError(f"Unsupported bootstrap metric: {metric_name}")

    if not values:
        return {
            "metric": metric_name,
            "defined": False,
            "bootstrap_samples": 0,
            "lower": 0.0,
            "upper": 0.0,
            "mean": 0.0,
        }

    values_array = np.asarray(values, dtype=np.float64)
    return {
        "metric": metric_name,
        "defined": True,
        "bootstrap_samples": int(values_array.size),
        "lower": float(np.quantile(values_array, 0.025)),
        "upper": float(np.quantile(values_array, 0.975)),
        "mean": float(values_array.mean()),
    }


def _group_auc(group_ids: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    if group_ids.size == 0:
        return {
            "value": 0.5,
            "valid_group_count": 0,
            "ignored_group_count": 0,
            "weighted_example_count": 0,
            "coverage": 0.0,
        }

    unique_group_ids, inverse_indices = np.unique(group_ids, return_inverse=True)
    weighted_auc_sum = 0.0
    weighted_example_count = 0
    valid_group_count = 0

    for group_index in range(unique_group_ids.size):
        mask = inverse_indices == group_index
        group_labels = labels[mask]
        positive_count = int(group_labels.sum())
        if positive_count == 0 or positive_count == int(group_labels.size):
            continue
        group_predictions = predictions[mask]
        group_auc = float(roc_auc_score(group_labels, group_predictions))
        group_weight = int(mask.sum())
        weighted_auc_sum += group_auc * group_weight
        weighted_example_count += group_weight
        valid_group_count += 1

    ignored_group_count = int(unique_group_ids.size - valid_group_count)
    if weighted_example_count == 0:
        value = 0.5
    else:
        value = weighted_auc_sum / weighted_example_count

    return {
        "value": float(value),
        "valid_group_count": valid_group_count,
        "ignored_group_count": ignored_group_count,
        "weighted_example_count": weighted_example_count,
        "coverage": float(weighted_example_count / max(group_ids.size, 1)),
    }


def _frequency_buckets(
    entity_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    if entity_ids.size == 0:
        return []

    unique_ids, inverse_indices, counts = np.unique(entity_ids, return_inverse=True, return_counts=True)
    row_frequencies = counts[inverse_indices].astype(np.float64)
    bucket_specs = [
        ("1", 1.0, 1.0),
        ("2-4", 2.0, 4.0),
        ("5-9", 5.0, 9.0),
        ("10-49", 10.0, 49.0),
        ("50+", 50.0, None),
    ]
    results: list[dict[str, Any]] = []
    entity_counts = counts.astype(np.float64)

    for bucket_name, lower_bound, upper_bound in bucket_specs:
        if upper_bound is None:
            row_mask = row_frequencies >= lower_bound
            entity_mask = entity_counts >= lower_bound
        else:
            row_mask = (row_frequencies >= lower_bound) & (row_frequencies <= upper_bound)
            entity_mask = (entity_counts >= lower_bound) & (entity_counts <= upper_bound)
        if not np.any(row_mask):
            continue
        record = _bucket_result(bucket_name, row_frequencies[row_mask], labels[row_mask], predictions[row_mask])
        record.update(
            {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "entity_count": int(entity_mask.sum()),
                "row_rate": float(row_mask.mean()),
            }
        )
        results.append(record)
    return results


def _sequence_length_buckets(values: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> list[dict[str, Any]]:
    bucket_specs = [
        ("0", 0.0, 0.0),
        ("1-4", 1.0, 4.0),
        ("5-16", 5.0, 16.0),
        ("17-64", 17.0, 64.0),
        ("65+", 65.0, None),
    ]
    results: list[dict[str, Any]] = []
    for bucket_name, lower_bound, upper_bound in bucket_specs:
        if upper_bound is None:
            mask = values >= lower_bound
        else:
            mask = (values >= lower_bound) & (values <= upper_bound)
        if not np.any(mask):
            continue
        record = _bucket_result(bucket_name, values[mask], labels[mask], predictions[mask])
        record.update({"lower_bound": lower_bound, "upper_bound": upper_bound})
        results.append(record)
    return results


def _quantile_buckets(
    values: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    bucket_count: int = 4,
) -> list[dict[str, Any]]:
    if values.size == 0:
        return []

    quantiles = np.quantile(values, np.linspace(0.0, 1.0, bucket_count + 1))
    edges = [float(quantiles[0])]
    for edge in quantiles[1:]:
        edge_value = float(edge)
        if edge_value > edges[-1]:
            edges.append(edge_value)

    if len(edges) == 1:
        record = _bucket_result("all", values, labels, predictions)
        record.update(
            {
                "lower_bound": edges[0],
                "upper_bound": edges[0],
                "quantile_start": 0.0,
                "quantile_end": 1.0,
            }
        )
        return [record]

    results: list[dict[str, Any]] = []
    total_segments = len(edges) - 1
    for index, (lower_bound, upper_bound) in enumerate(zip(edges, edges[1:]), start=1):
        if index == total_segments:
            mask = (values >= lower_bound) & (values <= upper_bound)
        else:
            mask = (values >= lower_bound) & (values < upper_bound)
        if not np.any(mask):
            continue
        record = _bucket_result(f"q{index}", values[mask], labels[mask], predictions[mask])
        record.update(
            {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "quantile_start": float((index - 1) / total_segments),
                "quantile_end": float(index / total_segments),
            }
        )
        results.append(record)
    return results


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    include_bucket_metrics: bool = True,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions: list[float] = []
    labels: list[float] = []
    sequence_lengths: list[float] = []
    behavior_densities: list[float] = []
    timestamps: list[float] = []
    user_indices: list[int] = []
    item_indices: list[int] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["labels"])

        probabilities = torch.sigmoid(logits)
        predictions.extend(probabilities.detach().cpu().tolist())
        labels.extend(batch["labels"].detach().cpu().tolist())

        history_mask = batch["history_mask"].float()
        history_lengths = history_mask.sum(dim=1)
        max_history_gap = (batch["history_time_gaps"] * history_mask).amax(dim=1)
        behavior_density = torch.where(
            history_lengths > 0,
            history_lengths / max_history_gap.clamp_min(1.0),
            torch.zeros_like(history_lengths),
        )

        sequence_lengths.extend(history_lengths.detach().cpu().tolist())
        behavior_densities.extend(behavior_density.detach().cpu().tolist())
        timestamps.extend(batch["timestamps"].detach().cpu().tolist())
        user_indices.extend(batch["user_indices"].detach().cpu().tolist())
        item_indices.extend(batch["item_indices"].detach().cpu().tolist())

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    labels_np = np.asarray(labels, dtype=np.float64)
    predictions_np = np.asarray(predictions, dtype=np.float64)
    user_indices_np = np.asarray(user_indices, dtype=np.int64)
    item_indices_np = np.asarray(item_indices, dtype=np.int64)
    metrics = _binary_classification_metrics(labels_np, predictions_np, auc_default=0.5)
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["gauc"] = _group_auc(user_indices_np, labels_np, predictions_np)
    metrics["confidence_intervals"] = {
        "auc": _bootstrap_metric_interval(labels_np, predictions_np, metric_name="auc"),
        "pr_auc": _bootstrap_metric_interval(labels_np, predictions_np, metric_name="pr_auc"),
    }

    if include_bucket_metrics:
        sequence_lengths_np = np.asarray(sequence_lengths, dtype=np.float64)
        behavior_densities_np = np.asarray(behavior_densities, dtype=np.float64)
        timestamps_np = np.asarray(timestamps, dtype=np.float64)
        metrics["bucket_metrics"] = {
            "sequence_length": _sequence_length_buckets(sequence_lengths_np, labels_np, predictions_np),
            "behavior_density": _quantile_buckets(behavior_densities_np, labels_np, predictions_np),
            "time_window": _quantile_buckets(timestamps_np, labels_np, predictions_np),
            "user_frequency": _frequency_buckets(user_indices_np, labels_np, predictions_np),
            "item_frequency": _frequency_buckets(item_indices_np, labels_np, predictions_np),
        }

    return metrics


@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_steps: int = 10,
    measure_steps: int = 30,
) -> dict[str, float]:
    model.eval()
    iterator = iter(loader)
    timings: list[float] = []

    for step in range(warmup_steps + measure_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = move_batch_to_device(batch, device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - start

        if step >= warmup_steps:
            timings.append(duration / max(batch["labels"].size(0), 1))

    if not timings:
        return {"mean_ms_per_sample": 0.0, "p95_ms_per_sample": 0.0}

    timings_ms = torch.tensor(timings, dtype=torch.float32) * 1000.0
    return {
        "mean_ms_per_sample": float(timings_ms.mean().item()),
        "p95_ms_per_sample": float(torch.quantile(timings_ms, 0.95).item()),
    }


__all__ = ["benchmark_latency", "evaluate_model", "move_batch_to_device"]
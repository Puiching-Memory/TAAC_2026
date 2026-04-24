from __future__ import annotations

import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from ...domain.config import DataConfig, ModelConfig
from ...domain.experiment import ExperimentSpec
from ...domain.features import FeatureSchema, sync_feature_schema
from ...domain.metrics import percentile, safe_mean
from ...domain.types import BatchTensors
from ...infrastructure.io.sparse_collate import build_batch_torchrec_features, validate_default_feature_schema
from .runtime_optimization import RuntimeExecution, prepare_runtime_execution


PROFILE_SCHEMA_VERSION = 2
DEFAULT_OPERATOR_SUMMARY_LIMIT = 8


@dataclass(slots=True)
class TimingSummary:
    observation_count: int
    mean: float
    p50: float
    p95: float
    minimum: float
    maximum: float
    standard_deviation: float
    total_seconds: float


@dataclass(slots=True)
class OperatorSummary:
    name: str
    calls: int
    cpu_self_time_ms: float
    cpu_total_time_ms: float
    device_self_time_ms: float
    device_total_time_ms: float
    flops: float


@dataclass(slots=True)
class ProfilerTraceSummary:
    activities: list[str]
    operator_count: int
    sort_key: str
    top_operations: list[OperatorSummary]


def _timing_summary(observations: list[float], *, total_seconds: float = 0.0) -> TimingSummary:
    if not observations:
        return TimingSummary(
            observation_count=0,
            mean=0.0,
            p50=0.0,
            p95=0.0,
            minimum=0.0,
            maximum=0.0,
            standard_deviation=0.0,
            total_seconds=float(total_seconds),
        )

    values = np.asarray(observations, dtype=np.float64)
    return TimingSummary(
        observation_count=int(values.size),
        mean=safe_mean(observations),
        p50=percentile(observations, 50.0),
        p95=percentile(observations, 95.0),
        minimum=float(values.min()),
        maximum=float(values.max()),
        standard_deviation=float(values.std()),
        total_seconds=float(total_seconds),
    )


def _profiler_activity_names(device: torch.device) -> list[str]:
    device = torch.device(device)
    names = ["cpu"]
    if device.type == "cuda":
        names.append("cuda")
    return names


def _event_device_total_time(event: Any) -> float:
    return float(
        getattr(
            event,
            "device_time_total",
            getattr(event, "cuda_time_total", 0.0),
        )
        or 0.0
    )


def _event_self_device_total_time(event: Any) -> float:
    return float(
        getattr(
            event,
            "self_device_time_total",
            getattr(event, "self_cuda_time_total", 0.0),
        )
        or 0.0
    )


def _summarize_profiler_trace(
    profiler,
    device: torch.device,
    *,
    top_k: int = DEFAULT_OPERATOR_SUMMARY_LIMIT,
) -> dict[str, Any]:
    events = list(profiler.key_averages())
    use_device_sort = device.type == "cuda" and any(_event_device_total_time(event) > 0.0 for event in events)
    if use_device_sort:
        sort_key = "device_total_time_ms"
        sorted_events = sorted(events, key=_event_device_total_time, reverse=True)
    else:
        sort_key = "cpu_total_time_ms"
        sorted_events = sorted(events, key=lambda event: float(getattr(event, "cpu_time_total", 0.0) or 0.0), reverse=True)

    top_operations = [
        OperatorSummary(
            name=str(event.key),
            calls=int(getattr(event, "count", 0) or 0),
            cpu_self_time_ms=float(getattr(event, "self_cpu_time_total", 0.0) or 0.0) / 1000.0,
            cpu_total_time_ms=float(getattr(event, "cpu_time_total", 0.0) or 0.0) / 1000.0,
            device_self_time_ms=_event_self_device_total_time(event) / 1000.0,
            device_total_time_ms=_event_device_total_time(event) / 1000.0,
            flops=float(getattr(event, "flops", 0.0) or 0.0),
        )
        for event in sorted_events[:top_k]
    ]
    return asdict(
        ProfilerTraceSummary(
            activities=_profiler_activity_names(device),
            operator_count=len(events),
            sort_key=sort_key,
            top_operations=top_operations,
        )
    )


def _parameter_profile(model) -> tuple[int, int, int]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return int(total_parameters), int(trainable_parameters), int(parameter_bytes)


def _measure_model_profile_batch(
    profiled_model,
    batch,
    device: torch.device,
    runtime_execution: RuntimeExecution | None = None,
) -> dict[str, Any]:
    profile_batch = batch.to(device)
    batch_size = max(profile_batch.batch_size, 1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with profile(activities=_profiler_activities_for(device), with_flops=True, record_shapes=False, acc_events=True) as profiler:
        with torch.no_grad():
            autocast_context = runtime_execution.autocast_context() if runtime_execution is not None else nullcontext()
            with autocast_context:
                _ = profiled_model(profile_batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time_ms = (time.perf_counter() - start) * 1000.0
    total_flops = float(profiler.key_averages().total_average().flops or 0.0)

    return {
        "profile_batch_size": int(profile_batch.batch_size),
        "flops_per_batch": total_flops,
        "tflops_per_batch": total_flops / 1.0e12,
        "flops_per_sample": total_flops / float(batch_size),
        "profiled_wall_time_ms": wall_time_ms,
        "profiled_wall_time_ms_per_sample": wall_time_ms / float(batch_size),
        "operator_summary": _summarize_profiler_trace(profiler, device),
    }


def _synthetic_token_tensor(batch_size: int, length: int, num_embeddings: int, *, offset: int = 0) -> torch.Tensor:
    if length <= 0:
        return torch.zeros((batch_size, 0), dtype=torch.long)
    if num_embeddings <= 1:
        return torch.zeros((batch_size, length), dtype=torch.long)
    base = torch.arange(length, dtype=torch.long) + int(offset)
    values = (base % (int(num_embeddings) - 1)) + 1
    return values.unsqueeze(0).repeat(batch_size, 1)


def _synthetic_mask(batch_size: int, length: int) -> torch.Tensor:
    if length <= 0:
        return torch.zeros((batch_size, 0), dtype=torch.bool)
    return torch.ones((batch_size, length), dtype=torch.bool)


def _table_max_length(feature_schema: FeatureSchema, name: str, fallback: int) -> int:
    try:
        table = feature_schema.table(name)
    except KeyError:
        return int(fallback)
    if table.max_length is None:
        return int(fallback)
    return int(table.max_length)


def _table_num_embeddings(feature_schema: FeatureSchema, name: str, fallback: int = 2) -> int:
    try:
        return int(feature_schema.table(name).num_embeddings)
    except KeyError:
        return int(fallback)


def build_synthetic_profile_batch(
    data_config: DataConfig,
    model_config: ModelConfig,
    *,
    feature_schema: FeatureSchema | None = None,
    batch_size: int = 1,
    dense_dim_override: int | None = None,
) -> BatchTensors:
    resolved_feature_schema = sync_feature_schema(feature_schema, data_config, model_config)
    resolved_batch_size = max(1, int(batch_size))
    sequence_names = resolved_feature_schema.sequence_names
    validate_default_feature_schema(resolved_feature_schema, sequence_names)
    history_capacity = _table_max_length(
        resolved_feature_schema,
        "history_tokens",
        max(1, len(sequence_names)) * int(data_config.max_seq_len),
    )
    sequence_length = max(1, int(data_config.max_seq_len))
    dense_dim = int(resolved_feature_schema.dense_dim if dense_dim_override is None else dense_dim_override)

    user_tokens = _synthetic_token_tensor(
        resolved_batch_size,
        _table_max_length(resolved_feature_schema, "user_tokens", int(data_config.max_feature_tokens)),
        _table_num_embeddings(resolved_feature_schema, "user_tokens", max(2, int(model_config.vocab_size))),
        offset=1,
    )
    context_tokens = _synthetic_token_tensor(
        resolved_batch_size,
        _table_max_length(resolved_feature_schema, "context_tokens", int(data_config.max_feature_tokens)),
        _table_num_embeddings(resolved_feature_schema, "context_tokens", max(2, int(model_config.vocab_size))),
        offset=5,
    )
    candidate_tokens = _synthetic_token_tensor(
        resolved_batch_size,
        _table_max_length(resolved_feature_schema, "candidate_tokens", 1),
        _table_num_embeddings(resolved_feature_schema, "candidate_tokens", max(2, int(model_config.vocab_size))),
        offset=9,
    )
    candidate_post_tokens = _synthetic_token_tensor(
        resolved_batch_size,
        _table_max_length(resolved_feature_schema, "candidate_post_tokens", max(1, int(data_config.max_event_features))),
        _table_num_embeddings(resolved_feature_schema, "candidate_post_tokens", max(2, int(model_config.vocab_size))),
        offset=13,
    )
    candidate_author_tokens = _synthetic_token_tensor(
        resolved_batch_size,
        _table_max_length(resolved_feature_schema, "candidate_author_tokens", 2),
        _table_num_embeddings(resolved_feature_schema, "candidate_author_tokens", max(2, int(model_config.vocab_size))),
        offset=17,
    )
    history_tokens = _synthetic_token_tensor(
        resolved_batch_size,
        history_capacity,
        _table_num_embeddings(resolved_feature_schema, "history_tokens", max(2, int(model_config.vocab_size))),
        offset=21,
    )
    history_post_tokens = _synthetic_token_tensor(
        resolved_batch_size,
        history_capacity,
        _table_num_embeddings(resolved_feature_schema, "history_post_tokens", max(2, int(model_config.vocab_size))),
        offset=25,
    )
    history_author_tokens = _synthetic_token_tensor(
        resolved_batch_size,
        history_capacity,
        _table_num_embeddings(resolved_feature_schema, "history_author_tokens", max(2, int(model_config.vocab_size))),
        offset=29,
    )
    history_action_tokens = _synthetic_token_tensor(
        resolved_batch_size,
        history_capacity,
        _table_num_embeddings(resolved_feature_schema, "history_action_tokens", max(2, int(model_config.vocab_size))),
        offset=33,
    )
    history_time_gap = _synthetic_token_tensor(
        resolved_batch_size,
        history_capacity,
        _table_num_embeddings(resolved_feature_schema, "history_time_gap", 64),
        offset=37,
    )
    history_group_ids = _synthetic_token_tensor(
        resolved_batch_size,
        history_capacity,
        _table_num_embeddings(resolved_feature_schema, "history_group_ids", max(2, len(sequence_names) + 1)),
        offset=41,
    )
    if sequence_names:
        sequence_tokens = torch.stack(
            [
                _synthetic_token_tensor(
                    resolved_batch_size,
                    _table_max_length(resolved_feature_schema, f"sequence:{sequence_name}", sequence_length),
                    _table_num_embeddings(resolved_feature_schema, f"sequence:{sequence_name}", max(2, int(model_config.vocab_size))),
                    offset=43 + sequence_index,
                )
                for sequence_index, sequence_name in enumerate(sequence_names)
            ],
            dim=1,
        )
        sequence_mask = torch.ones(sequence_tokens.shape, dtype=torch.bool)
    else:
        sequence_tokens = torch.zeros((resolved_batch_size, 0, sequence_length), dtype=torch.long)
        sequence_mask = torch.zeros((resolved_batch_size, 0, sequence_length), dtype=torch.bool)
    user_mask = _synthetic_mask(resolved_batch_size, int(user_tokens.shape[1]))
    context_mask = _synthetic_mask(resolved_batch_size, int(context_tokens.shape[1]))
    candidate_mask = _synthetic_mask(resolved_batch_size, int(candidate_tokens.shape[1]))
    candidate_post_mask = _synthetic_mask(resolved_batch_size, int(candidate_post_tokens.shape[1]))
    candidate_author_mask = _synthetic_mask(resolved_batch_size, int(candidate_author_tokens.shape[1]))
    history_mask = _synthetic_mask(resolved_batch_size, history_capacity)
    sparse_features, sequence_features = build_batch_torchrec_features(
        sequence_names=sequence_names,
        feature_schema=resolved_feature_schema,
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
    if dense_dim <= 0:
        dense_features = torch.zeros((resolved_batch_size, 0), dtype=torch.float32)
    else:
        dense_features = torch.linspace(0.0, 1.0, steps=dense_dim, dtype=torch.float32).unsqueeze(0).repeat(resolved_batch_size, 1)
    return BatchTensors(
        dense_features=dense_features,
        labels=torch.zeros(resolved_batch_size, dtype=torch.float32),
        user_indices=torch.arange(resolved_batch_size, dtype=torch.long),
        item_indices=torch.arange(1, resolved_batch_size + 1, dtype=torch.long),
        item_logq=torch.zeros(resolved_batch_size, dtype=torch.float32),
        sparse_features=sparse_features,
        sequence_features=sequence_features,
    )


def collect_synthetic_model_profile(
    model,
    data_config: DataConfig,
    model_config: ModelConfig,
    device,
    *,
    feature_schema: FeatureSchema | None = None,
    runtime_execution: RuntimeExecution | None = None,
    profile_batch_size: int = 1,
    dense_dim_override: int | None = None,
) -> dict[str, float | int | str]:
    device = torch.device(device)
    total_parameters, trainable_parameters, parameter_bytes = _parameter_profile(model)
    profiled_model = runtime_execution.execution_model if runtime_execution is not None else model
    synthetic_batch = build_synthetic_profile_batch(
        data_config,
        model_config,
        feature_schema=feature_schema,
        batch_size=profile_batch_size,
        dense_dim_override=dense_dim_override,
    )
    was_training = profiled_model.training
    profiled_model.eval()
    try:
        profile = _measure_model_profile_batch(
            profiled_model,
            synthetic_batch,
            device,
            runtime_execution=runtime_execution,
        )
    finally:
        if was_training:
            profiled_model.train()

    return {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_scope": "synthetic_fixed_forward",
        "profile_input_kind": "synthetic_fixed_batch",
        "device": str(device),
        "profiled_batches": 1,
        "selected_batch_index": 0,
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "parameter_size_bytes": int(parameter_bytes),
        "parameter_size_mb": parameter_bytes / (1024.0 * 1024.0),
        "flops_profile_status": "measured" if float(profile["flops_per_sample"]) > 0.0 else "unavailable",
        **profile,
    }


def collect_experiment_model_profile(
    experiment: ExperimentSpec,
    model,
    device,
    *,
    runtime_execution: RuntimeExecution | None = None,
    profile_batch_size: int = 1,
    dense_dim: int | None = None,
) -> dict[str, float | int | str]:
    return collect_synthetic_model_profile(
        model,
        experiment.data,
        experiment.model,
        device,
        feature_schema=experiment.feature_schema,
        runtime_execution=runtime_execution,
        profile_batch_size=profile_batch_size,
        dense_dim_override=dense_dim,
    )


def build_profiling_report(
    *,
    device: torch.device | str,
    latency: dict[str, Any],
    model_profile: dict[str, Any],
    inference_profile: dict[str, Any],
    compute_profile: dict[str, Any],
    external_profilers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "device": str(device),
        "latency": latency,
        "model_profile": model_profile,
        "inference_profile": inference_profile,
        "compute_profile": compute_profile,
    }
    if external_profilers is not None:
        report["external_profilers"] = external_profilers
    return report


def select_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_prediction_records(batch: BatchTensors, logits: torch.Tensor) -> list[dict[str, Any]]:
    metadata = batch.metadata
    if metadata is None:
        raise RuntimeError("Batch is missing required metadata for validation prediction export")

    scores = logits.detach().float().cpu().reshape(-1).tolist()
    targets = batch.labels.detach().cpu().reshape(-1).tolist()
    batch_size = int(batch.batch_size)
    expected_size = len(metadata.sample_indices)
    if expected_size != batch_size:
        raise RuntimeError(
            f"Batch metadata length mismatch: expected {batch_size} records, got {expected_size}"
        )

    records: list[dict[str, Any]] = []
    for index, score in enumerate(scores):
        records.append(
            {
                "sample_index": int(metadata.sample_indices[index]),
                "user_id": metadata.user_ids[index],
                "item_id": metadata.item_ids[index],
                "timestamp": int(metadata.timestamps[index]),
                "raw_label": int(metadata.raw_labels[index]),
                "target": int(round(float(targets[index]))),
                "score": float(score),
            }
        )
    return records


def _collect_loader_outputs_impl(
    model,
    loader,
    device,
    loss_fn=None,
    runtime_execution: RuntimeExecution | None = None,
    *,
    include_prediction_records: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[dict[str, Any]]]:
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    group_list: list[np.ndarray] = []
    losses: list[float] = []
    prediction_records: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            autocast_context = runtime_execution.autocast_context() if runtime_execution is not None else nullcontext()
            with autocast_context:
                logits = model(batch)
                if loss_fn is not None:
                    losses.append(float(loss_fn(logits, batch.labels).detach().cpu().item()))
            logits_list.append(logits.detach().float().cpu().numpy())
            labels_list.append(batch.labels.detach().cpu().numpy())
            group_list.append(batch.user_indices.detach().cpu().numpy())
            if include_prediction_records:
                prediction_records.extend(_batch_prediction_records(batch, logits))
    if not logits_list:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty, 0.0, prediction_records
    return (
        np.concatenate(logits_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(group_list, axis=0),
        safe_mean(losses),
        prediction_records,
    )


def collect_loader_outputs(
    model,
    loader,
    device,
    loss_fn=None,
    runtime_execution: RuntimeExecution | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    logits, labels, groups, loss, _ = _collect_loader_outputs_impl(
        model,
        loader,
        device,
        loss_fn,
        runtime_execution,
    )
    return logits, labels, groups, loss


def collect_loader_outputs_with_predictions(
    model,
    loader,
    device,
    loss_fn=None,
    runtime_execution: RuntimeExecution | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[dict[str, Any]]]:
    return _collect_loader_outputs_impl(
        model,
        loader,
        device,
        loss_fn,
        runtime_execution,
        include_prediction_records=True,
    )


def measure_latency(
    model,
    loader,
    device,
    warmup_steps: int,
    measure_steps: int,
    runtime_execution: RuntimeExecution | None = None,
) -> dict[str, float]:
    device = torch.device(device)
    durations: list[float] = []
    warmup_batches = 0
    warmup_samples = 0
    measured_samples = 0
    total_elapsed_seconds = 0.0
    model.eval()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if step < warmup_steps:
            warmup_batches += 1
            warmup_samples += int(batch.batch_size)
            with torch.no_grad():
                autocast_context = runtime_execution.autocast_context() if runtime_execution is not None else nullcontext()
                with autocast_context:
                    _ = model(batch)
            continue
        if measure_steps > 0 and len(durations) >= measure_steps:
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            autocast_context = runtime_execution.autocast_context() if runtime_execution is not None else nullcontext()
            with autocast_context:
                _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        total_elapsed_seconds += elapsed
        durations.append((elapsed * 1000.0) / max(batch.batch_size, 1))
        measured_samples += int(batch.batch_size)
    timing = _timing_summary(durations, total_seconds=total_elapsed_seconds)
    return {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_scope": "loader_eval_forward",
        "device": str(device),
        "latency_unit": "ms_per_sample",
        "warmup_steps": int(max(warmup_steps, 0)),
        "measure_steps": int(max(measure_steps, 0)),
        "warmup_batches": warmup_batches,
        "warmup_samples": warmup_samples,
        "measured_batches": int(timing.observation_count),
        "measured_samples": measured_samples,
        "profiled_batches": warmup_batches + int(timing.observation_count),
        "profiled_samples": warmup_samples + measured_samples,
        "total_measured_seconds": float(total_elapsed_seconds),
        "mean_latency_ms_per_sample": timing.mean,
        "p50_latency_ms_per_sample": timing.p50,
        "p95_latency_ms_per_sample": timing.p95,
        "min_latency_ms_per_sample": timing.minimum,
        "max_latency_ms_per_sample": timing.maximum,
        "latency_std_ms_per_sample": timing.standard_deviation,
    }


def _profiler_activities_for(device: torch.device) -> list[ProfilerActivity]:
    device = torch.device(device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def _loader_num_batches(loader) -> int:
    try:
        return len(loader)
    except TypeError:
        return 0


def _loader_num_samples(loader, max_batches: int | None = None) -> int:
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        sample_count += int(batch.batch_size)
    return sample_count


def _count_latency_probe_batches(total_batches: int, warmup_steps: int, measure_steps: int) -> int:
    if total_batches <= 0:
        return 0
    warmup_batches = min(total_batches, max(warmup_steps, 0))
    remaining_batches = max(total_batches - warmup_batches, 0)
    measured_batches = remaining_batches if measure_steps <= 0 else min(remaining_batches, measure_steps)
    return warmup_batches + measured_batches


def collect_inference_profile(
    experiment: ExperimentSpec,
    val_loader_or_sample_count,
    latency: dict[str, float],
) -> dict[str, float | int]:
    if isinstance(val_loader_or_sample_count, int):
        val_sample_count = int(val_loader_or_sample_count)
    else:
        val_sample_count = _loader_num_samples(val_loader_or_sample_count)
    return {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_scope": "scaled_eval_forward_latency",
        "estimation_method": "measured_eval_forward_latency_scaled_by_validation_samples",
        "device": str(latency.get("device", "unknown")),
        "val_sample_count": int(val_sample_count),
        "latency_warmup_steps": int(experiment.train.latency_warmup_steps),
        "latency_measure_steps": int(experiment.train.latency_measure_steps),
        "latency_observed_batches": int(latency.get("measured_batches", 0)),
        "latency_observed_samples": int(latency.get("measured_samples", 0)),
    }


def collect_compute_profile(
    experiment: ExperimentSpec,
    model,
    loss_fn,
    train_loader,
    val_loader,
    data_stats,
    device,
    model_profile: dict[str, float | int | str],
    latency: dict[str, Any] | None = None,
    runtime_execution: RuntimeExecution | None = None,
) -> dict[str, float | int | str]:
    from ...infrastructure.nn.defaults import resolve_experiment_builders

    del model
    del val_loader
    del model_profile
    del latency

    device = torch.device(device)
    train_profile_batch = next(iter(train_loader), None)
    if train_profile_batch is None:
        train_step_flops = 0.0
        train_profile_batch_size = 0
        train_step_flops_per_sample = 0.0
        train_step_wall_time_ms = 0.0
        train_operator_summary = asdict(
            ProfilerTraceSummary(
                activities=_profiler_activity_names(device),
                operator_count=0,
                sort_key="cpu_total_time_ms",
                top_operations=[],
            )
        )
    else:
        train_profile_batch = train_profile_batch.to(device)
        train_profile_batch_size = int(train_profile_batch.batch_size)
        profile_model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
        profile_model = profile_model.to(device)
        profile_runtime = prepare_runtime_execution(profile_model, experiment.train, device)
        profile_execution_model = profile_runtime.execution_model
        profile_optimizer = resolve_experiment_builders(experiment).build_optimizer_component(
            profile_model,
            experiment.train,
        )
        profile_execution_model.train()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with profile(
            activities=_profiler_activities_for(device),
            with_flops=True,
            record_shapes=False,
            acc_events=True,
        ) as profiler:
            profile_optimizer.zero_grad(set_to_none=True)
            with profile_runtime.autocast_context():
                logits = profile_execution_model(train_profile_batch)
                loss = loss_fn(logits, train_profile_batch.labels)
            profile_runtime.backward_and_step(
                loss,
                profile_optimizer,
                model_parameters=profile_model.parameters(),
                grad_clip_norm=experiment.train.grad_clip_norm,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_step_wall_time_ms = (time.perf_counter() - start) * 1000.0

        train_step_flops = float(profiler.key_averages().total_average().flops or 0.0)
        train_step_flops_per_sample = train_step_flops / float(max(train_profile_batch_size, 1))
        train_operator_summary = _summarize_profiler_trace(profiler, device)

        del profile_optimizer
        del profile_model
        del train_profile_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_scope": "single_train_step_profile",
        "device": str(device),
        "train_profile_scope": "single_train_step_forward_backward_optimizer",
        "train_profile_batch_size": train_profile_batch_size,
        "train_step_wall_time_ms": train_step_wall_time_ms,
        "train_step_wall_time_ms_per_sample": train_step_wall_time_ms / float(max(train_profile_batch_size, 1)),
        "train_step_flops": train_step_flops,
        "train_step_tflops": train_step_flops / 1.0e12,
        "train_step_flops_per_sample": train_step_flops_per_sample,
        "train_operator_summary": train_operator_summary,
    }


__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "build_profiling_report",
    "build_synthetic_profile_batch",
    "collect_compute_profile",
    "collect_inference_profile",
    "collect_loader_outputs",
    "collect_synthetic_model_profile",
    "measure_latency",
    "select_device",
    "set_random_seed",
]

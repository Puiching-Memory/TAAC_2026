from __future__ import annotations

import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from ...domain.experiment import ExperimentSpec
from ...domain.metrics import percentile, safe_mean
from ...infrastructure.nn.defaults import resolve_experiment_builders
from .runtime_optimization import RuntimeExecution, prepare_runtime_execution


PROFILE_SCHEMA_VERSION = 1
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


def collect_loader_outputs(
    model,
    loader,
    device,
    loss_fn=None,
    runtime_execution: RuntimeExecution | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    group_list: list[np.ndarray] = []
    losses: list[float] = []
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
    if not logits_list:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty, 0.0
    return (
        np.concatenate(logits_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(group_list, axis=0),
        safe_mean(losses),
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
        return int(len(loader))
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
    mean_latency_ms_per_sample = float(latency.get("mean_latency_ms_per_sample", 0.0))
    p50_latency_ms_per_sample = float(latency.get("p50_latency_ms_per_sample", 0.0))
    p95_latency_ms_per_sample = float(latency.get("p95_latency_ms_per_sample", 0.0))
    estimated_end_to_end_inference_seconds = (mean_latency_ms_per_sample * float(val_sample_count)) / 1000.0
    estimated_end_to_end_inference_seconds_p50 = (p50_latency_ms_per_sample * float(val_sample_count)) / 1000.0
    estimated_end_to_end_inference_seconds_p95 = (p95_latency_ms_per_sample * float(val_sample_count)) / 1000.0
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
        "estimated_end_to_end_inference_seconds": estimated_end_to_end_inference_seconds,
        "estimated_end_to_end_inference_minutes": estimated_end_to_end_inference_seconds / 60.0,
        "estimated_end_to_end_inference_seconds_p50": estimated_end_to_end_inference_seconds_p50,
        "estimated_end_to_end_inference_minutes_p50": estimated_end_to_end_inference_seconds_p50 / 60.0,
        "estimated_end_to_end_inference_seconds_p95": estimated_end_to_end_inference_seconds_p95,
        "estimated_end_to_end_inference_minutes_p95": estimated_end_to_end_inference_seconds_p95 / 60.0,
    }


def collect_model_profile(
    model,
    loader,
    device,
    runtime_execution: RuntimeExecution | None = None,
) -> dict[str, float | int | str]:
    device = torch.device(device)
    total_parameters, trainable_parameters, parameter_bytes = _parameter_profile(model)

    profile_batch = next(iter(loader), None)
    if profile_batch is None:
        return {
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
            "profile_scope": "single_eval_forward",
            "device": str(device),
            "profile_batch_size": 0,
            "total_parameters": int(total_parameters),
            "trainable_parameters": int(trainable_parameters),
            "parameter_size_bytes": int(parameter_bytes),
            "parameter_size_mb": parameter_bytes / (1024.0 * 1024.0),
            "flops_per_batch": 0.0,
            "tflops_per_batch": 0.0,
            "flops_per_sample": 0.0,
            "profiled_wall_time_ms": 0.0,
            "profiled_wall_time_ms_per_sample": 0.0,
            "operator_summary": asdict(
                ProfilerTraceSummary(
                    activities=_profiler_activity_names(device),
                    operator_count=0,
                    sort_key="cpu_total_time_ms",
                    top_operations=[],
                )
            ),
        }

    profile_batch = profile_batch.to(device)
    batch_size = max(profile_batch.batch_size, 1)
    profiled_model = runtime_execution.execution_model if runtime_execution is not None else model
    was_training = profiled_model.training
    profiled_model.eval()

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

    if was_training:
        profiled_model.train()

    total_flops = float(profiler.key_averages().total_average().flops or 0.0)
    return {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_scope": "single_eval_forward",
        "device": str(device),
        "profile_batch_size": int(profile_batch.batch_size),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "parameter_size_bytes": int(parameter_bytes),
        "parameter_size_mb": parameter_bytes / (1024.0 * 1024.0),
        "flops_per_batch": total_flops,
        "tflops_per_batch": total_flops / 1.0e12,
        "flops_per_sample": total_flops / float(batch_size),
        "profiled_wall_time_ms": wall_time_ms,
        "profiled_wall_time_ms_per_sample": wall_time_ms / float(batch_size),
        "operator_summary": _summarize_profiler_trace(profiler, device),
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
    device = torch.device(device)
    train_batches_per_epoch = _loader_num_batches(train_loader)
    val_batches_per_epoch = _loader_num_batches(val_loader)
    planned_latency_probe_batches = _count_latency_probe_batches(
        total_batches=val_batches_per_epoch,
        warmup_steps=experiment.train.latency_warmup_steps,
        measure_steps=experiment.train.latency_measure_steps,
    )
    if latency is None:
        latency_probe_batches = planned_latency_probe_batches
        latency_probe_samples = _loader_num_samples(val_loader, max_batches=latency_probe_batches)
        latency_probe_source = "loader_scan"
    else:
        latency_probe_batches = int(latency.get("profiled_batches", planned_latency_probe_batches))
        latency_probe_samples = int(latency.get("profiled_samples", 0))
        latency_probe_source = "latency_profile"

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
            if profile_runtime.gradient_scaler is not None:
                profile_runtime.gradient_scaler.scale(loss).backward()
                if experiment.train.grad_clip_norm and experiment.train.grad_clip_norm > 0:
                    profile_runtime.gradient_scaler.unscale_(profile_optimizer)
                    torch.nn.utils.clip_grad_norm_(profile_model.parameters(), experiment.train.grad_clip_norm)
                profile_runtime.gradient_scaler.step(profile_optimizer)
                profile_runtime.gradient_scaler.update()
            else:
                loss.backward()
                if experiment.train.grad_clip_norm and experiment.train.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(profile_model.parameters(), experiment.train.grad_clip_norm)
                profile_optimizer.step()
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

    eval_flops_per_sample = float(model_profile.get("flops_per_sample", 0.0))
    train_samples_per_epoch = int(data_stats.train_size)
    val_samples_per_epoch = int(data_stats.val_size)

    estimated_train_flops_total = train_step_flops_per_sample * float(train_samples_per_epoch) * float(experiment.train.epochs)
    estimated_eval_flops_total = eval_flops_per_sample * float(val_samples_per_epoch) * float(experiment.train.epochs)
    estimated_latency_probe_flops_total = eval_flops_per_sample * float(latency_probe_samples)
    estimated_end_to_end_flops_total = (
        estimated_train_flops_total
        + estimated_eval_flops_total
        + estimated_latency_probe_flops_total
    )

    return {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_scope": "single_train_step_scaled",
        "device": str(device),
        "estimation_method": "profiled_single_step_scaled_by_observed_sample_counts",
        "epochs": int(experiment.train.epochs),
        "train_batches_per_epoch": train_batches_per_epoch,
        "val_batches_per_epoch": val_batches_per_epoch,
        "planned_latency_probe_batches": planned_latency_probe_batches,
        "train_samples_per_epoch": train_samples_per_epoch,
        "val_samples_per_epoch": val_samples_per_epoch,
        "latency_probe_batches": latency_probe_batches,
        "latency_probe_samples": latency_probe_samples,
        "latency_probe_source": latency_probe_source,
        "train_profile_scope": "single_train_step_forward_backward_optimizer",
        "train_profile_batch_size": train_profile_batch_size,
        "train_step_wall_time_ms": train_step_wall_time_ms,
        "train_step_wall_time_ms_per_sample": train_step_wall_time_ms / float(max(train_profile_batch_size, 1)),
        "train_step_flops": train_step_flops,
        "train_step_tflops": train_step_flops / 1.0e12,
        "train_step_flops_per_sample": train_step_flops_per_sample,
        "estimated_train_flops_total": estimated_train_flops_total,
        "estimated_train_tflops_total": estimated_train_flops_total / 1.0e12,
        "estimated_eval_flops_total": estimated_eval_flops_total,
        "estimated_eval_tflops_total": estimated_eval_flops_total / 1.0e12,
        "estimated_latency_probe_flops_total": estimated_latency_probe_flops_total,
        "estimated_latency_probe_tflops_total": estimated_latency_probe_flops_total / 1.0e12,
        "estimated_end_to_end_flops_total": estimated_end_to_end_flops_total,
        "estimated_end_to_end_tflops_total": estimated_end_to_end_flops_total / 1.0e12,
        "train_operator_summary": train_operator_summary,
    }


__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "build_profiling_report",
    "collect_compute_profile",
    "collect_inference_profile",
    "collect_loader_outputs",
    "collect_model_profile",
    "measure_latency",
    "select_device",
    "set_random_seed",
]

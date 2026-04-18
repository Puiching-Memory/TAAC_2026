from __future__ import annotations

from pathlib import Path
from typing import Any

from ...domain.config import SearchConfig
from ...domain.experiment import ExperimentSpec
from ...infrastructure.experiments.payload import apply_serialized_experiment, serialize_experiment
from ...infrastructure.nn.defaults import resolve_experiment_builders
from ..training.profiling import (
    collect_inference_profile,
    collect_model_profile,
    measure_latency,
    select_device,
)
from ..training.runtime_optimization import prepare_runtime_execution
from ..training.service import run_training


def resolve_metric(summary: dict[str, Any], metric_name: str) -> float:
    current: Any = summary
    for part in metric_name.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Metric '{metric_name}' is not present in summary")
        current = current[part]
    return float(current)


def budget_status(
    model_profile: dict[str, Any],
    inference_profile: dict[str, Any],
    search_config: SearchConfig,
) -> dict[str, Any]:
    parameter_bytes = float(model_profile.get("parameter_size_mb", 0.0)) * 1024.0 * 1024.0
    estimated_inference_seconds = float(inference_profile.get("estimated_end_to_end_inference_seconds", 0.0))
    parameter_budget_met = parameter_bytes <= float(search_config.max_parameter_bytes)
    inference_budget_met = estimated_inference_seconds <= float(search_config.max_end_to_end_inference_seconds)
    return {
        "parameter_budget_met": parameter_budget_met,
        "inference_budget_met": inference_budget_met,
        "constraints_met": parameter_budget_met and inference_budget_met,
        "parameter_bytes": parameter_bytes,
        "parameter_gib": parameter_bytes / float(1024**3),
        "max_parameter_bytes": int(search_config.max_parameter_bytes),
        "max_parameter_gib": float(search_config.max_parameter_bytes) / float(1024**3),
        "estimated_end_to_end_inference_seconds": estimated_inference_seconds,
        "estimated_end_to_end_inference_minutes": estimated_inference_seconds / 60.0,
        "max_end_to_end_inference_seconds": float(search_config.max_end_to_end_inference_seconds),
        "max_end_to_end_inference_minutes": float(search_config.max_end_to_end_inference_seconds) / 60.0,
    }


def profile_trial_budget(experiment: ExperimentSpec) -> dict[str, Any]:
    device = select_device(experiment.train.device)
    builders = resolve_experiment_builders(experiment)
    train_loader, val_loader, data_stats = builders.build_data_pipeline(
        experiment.data,
        experiment.model,
        experiment.train,
    )
    del train_loader
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    model = model.to(device)
    runtime_execution = prepare_runtime_execution(model, experiment.train, device)
    execution_model = runtime_execution.execution_model

    try:
        model_profile = collect_model_profile(model, val_loader, device, runtime_execution=runtime_execution)
        latency = measure_latency(
            execution_model,
            val_loader,
            device,
            warmup_steps=experiment.train.latency_warmup_steps,
            measure_steps=experiment.train.latency_measure_steps,
            runtime_execution=runtime_execution,
        )
        inference_profile = collect_inference_profile(experiment, int(data_stats.val_size), latency)
        return {
            "model_profile": model_profile,
            "latency": latency,
            "inference_profile": inference_profile,
            "budget_status": budget_status(model_profile, inference_profile, experiment.search),
        }
    finally:
        del model
        if device.type == "cuda":
            import torch

            torch.cuda.empty_cache()


def execute_search_trial(experiment: ExperimentSpec) -> dict[str, Any]:
    budget_probe = profile_trial_budget(experiment)
    result: dict[str, Any] = {
        "status": "pruned",
        "budget_probe": budget_probe,
        "summary_path": None,
        "final_budget_status": None,
        "objective_value": None,
        "prune_reason": None,
    }
    if not budget_probe["budget_status"]["constraints_met"]:
        result["prune_reason"] = "trial exceeds search budget before training"
        return result

    summary = run_training(experiment)
    summary_path = Path(experiment.train.output_dir) / "summary.json"
    final_budget = budget_status(summary["model_profile"], summary["inference_profile"], experiment.search)
    result["summary_path"] = str(summary_path)
    result["final_budget_status"] = final_budget

    if not final_budget["constraints_met"]:
        result["prune_reason"] = "trial exceeds search budget after training"
        return result

    result["status"] = "complete"
    result["objective_value"] = resolve_metric(summary, experiment.search.metric_name)
    return result


__all__ = [
    "apply_serialized_experiment",
    "budget_status",
    "execute_search_trial",
    "profile_trial_budget",
    "resolve_metric",
    "serialize_experiment",
]

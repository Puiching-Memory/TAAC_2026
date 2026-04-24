from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...domain.experiment import ExperimentSpec
from ...domain.metrics import compute_classification_metrics
from ...infrastructure.experiments.loader import load_experiment_package
from ...infrastructure.io.console import logger
from ...infrastructure.io.files import write_json
from ...infrastructure.nn.defaults import resolve_experiment_builders
from ...infrastructure.nn.quantization import normalize_quantization_mode
from ..training.artifacts import write_validation_predictions
from ..training.external_profilers import (
    build_evaluation_external_profiler_plan,
    write_external_profiler_plan_artifacts,
)
from ..training.profiling import PROFILE_SCHEMA_VERSION, collect_loader_outputs_with_predictions, measure_latency, select_device
from .inference import (
    export_model_for_inference,
    normalize_inference_export_mode,
    prepare_evaluation_inference,
)


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def record_bucket(record: dict[str, Any]) -> int:
        if bool(record.get("latency_budget_met")) and float(record.get("latency_budget_ms_per_sample", 0.0)) > 0.0:
            return 0
        if bool(record.get("latency_budget_met")):
            return 1
        return 2

    return sorted(
        records,
        key=lambda record: (
            record_bucket(record),
            -float(record.get("auc", 0.0)),
            -float(record.get("pr_auc", 0.0)),
            float(record.get("mean_latency_ms_per_sample", float("inf"))),
            str(record.get("experiment_id", "")),
        ),
    )


def _resolve_validation_predictions_path(
    output_path: str | Path | None,
    run_dir: str | Path,
) -> Path:
    if output_path is None:
        return Path(run_dir) / "validation_predictions.jsonl"
    report_path = Path(output_path)
    return report_path.with_name(f"{report_path.stem}.validation_predictions.jsonl")


def evaluate_checkpoint(
    experiment_path: str | Path,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    experiment: ExperimentSpec | None = None,
    quantization_mode: str | None = None,
    export_mode: str | None = None,
    export_path: str | Path | None = None,
) -> dict[str, Any]:
    experiment = experiment.clone() if experiment is not None else load_experiment_package(experiment_path)
    resolved_quantization_mode = normalize_quantization_mode(quantization_mode)
    resolved_export_mode = normalize_inference_export_mode(export_mode)
    device = torch.device("cpu") if resolved_quantization_mode != "none" else select_device(experiment.train.device)
    logger.info(
        "evaluate start: experiment={} device={} quantization={} export={}",
        experiment_path,
        device,
        resolved_quantization_mode,
        resolved_export_mode,
    )
    builders = resolve_experiment_builders(experiment)
    train_loader, val_loader, data_stats = builders.build_data_pipeline(
        experiment.data,
        experiment.model,
        experiment.train,
    )
    del train_loader
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    model = model.to(device)
    loss_fn, _ = builders.build_loss_stack(
        experiment.data,
        experiment.model,
        experiment.train,
        data_stats,
        device,
    )

    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else Path(experiment.train.output_dir) / "best.pt"
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

    payload = torch.load(resolved_checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict", payload)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"incompatible checkpoint: {exc}") from exc

    runtime_execution, quantization_summary, inference_train_config = prepare_evaluation_inference(
        model,
        experiment.train,
        device,
        quantization_mode=resolved_quantization_mode,
    )
    execution_model = runtime_execution.execution_model
    external_profiler_output_dir = Path(output_path).parent if output_path is not None else Path(experiment.train.output_dir)

    if resolved_export_mode != "none" and resolved_quantization_mode != "none":
        raise ValueError("Inference export currently requires quantization mode 'none'")

    example_batch = None
    resolved_export_path = Path(export_path) if export_path is not None else external_profiler_output_dir / "inference_export.pt2"
    if resolved_export_mode != "none":
        example_batch = next(iter(val_loader)).to(runtime_execution.device)
    export_summary = export_model_for_inference(
        runtime_execution.base_model,
        example_batch,
        mode=resolved_export_mode,
        output_path=resolved_export_path,
    )

    logits, labels, groups, loss, prediction_records = collect_loader_outputs_with_predictions(
        execution_model,
        val_loader,
        device,
        loss_fn,
        runtime_execution=runtime_execution,
    )
    metrics = compute_classification_metrics(labels, logits, groups)
    latency = measure_latency(
        execution_model,
        val_loader,
        device,
        warmup_steps=experiment.train.latency_warmup_steps,
        measure_steps=experiment.train.latency_measure_steps,
        runtime_execution=runtime_execution,
    )
    external_profilers = build_evaluation_external_profiler_plan(
        device=str(device),
        output_dir=external_profiler_output_dir,
        experiment_path=experiment_path,
        checkpoint_path=resolved_checkpoint,
        output_path=output_path,
        run_dir=experiment.train.output_dir,
        train_config=inference_train_config,
    )
    write_external_profiler_plan_artifacts(external_profilers)
    validation_predictions_path = _resolve_validation_predictions_path(output_path, experiment.train.output_dir)
    write_validation_predictions(validation_predictions_path, prediction_records)
    report = {
        "experiment": experiment.name,
        "experiment_path": str(experiment_path),
        "model_name": experiment.model.name,
        "device": str(device),
        "checkpoint_path": str(resolved_checkpoint),
        "validation_predictions_path": str(validation_predictions_path),
        "loss": loss,
        "auc": float(metrics.get("auc", 0.0)),
        "pr_auc": float(metrics.get("pr_auc", 0.0)),
        "metrics": metrics,
        "quantization": quantization_summary,
        "export": export_summary,
        "runtime_optimization": runtime_execution.summary(),
        "profiling": {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "device": str(device),
            "latency": latency,
            "external_profilers": external_profilers,
        },
        **latency,
    }
    if output_path is not None:
        write_json(output_path, report)
    logger.info(
        "evaluate complete: experiment={} auc={:.6f} pr_auc={:.6f} latency_ms={:.4f} quantization={} export={}",
        experiment_path,
        float(metrics.get("auc", 0.0)),
        float(metrics.get("pr_auc", 0.0)),
        float(report["mean_latency_ms_per_sample"]),
        quantization_summary["mode"],
        export_summary["mode"],
    )
    return report


__all__ = ["_sort_records", "evaluate_checkpoint"]

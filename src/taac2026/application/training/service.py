from __future__ import annotations

from typing import Any

import torch

from ...domain.experiment import ExperimentSpec
from ...domain.metrics import compute_classification_metrics, safe_mean
from ...infrastructure.io.console import create_progress_bar, logger
from ...infrastructure.io.files import ensure_dir, replace_file, write_json
from ...infrastructure.nn.defaults import resolve_experiment_builders
from .artifacts import write_training_curve_artifacts
from .external_profilers import (
    build_training_external_profiler_plan,
    write_external_profiler_plan_artifacts,
)
from .profiling import (
    build_profiling_report,
    collect_compute_profile,
    collect_inference_profile,
    collect_loader_outputs,
    collect_model_profile,
    measure_latency,
    select_device,
    set_random_seed,
)
from .runtime_optimization import prepare_runtime_execution


def _write_best_checkpoint(path: str, payload: dict[str, Any]) -> None:
    replace_file(path, lambda staged_path: torch.save(payload, staged_path))


def run_training(
    experiment: ExperimentSpec,
    *,
    experiment_path: str | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    output_dir = ensure_dir(experiment.train.output_dir)
    set_random_seed(experiment.train.seed)
    device = select_device(experiment.train.device)
    logger.info(
        "training start: experiment={} output_dir={} device={} epochs={} batch_size={}",
        experiment.name,
        output_dir,
        device,
        experiment.train.epochs,
        experiment.train.batch_size,
    )

    builders = resolve_experiment_builders(experiment)
    train_loader, val_loader, data_stats = builders.build_data_pipeline(
        experiment.data,
        experiment.model,
        experiment.train,
    )
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    model = model.to(device)
    runtime_execution = prepare_runtime_execution(model, experiment.train, device)
    execution_model = runtime_execution.execution_model
    loss_fn, auxiliary_loss = builders.build_loss_stack(
        experiment.data,
        experiment.model,
        experiment.train,
        data_stats,
        device,
    )
    optimizer = builders.build_optimizer_component(model, experiment.train)
    model_profile = collect_model_profile(model, val_loader, device, runtime_execution=runtime_execution)
    logger.info(
        "runtime optimization: compile_active={} amp_active={} amp_dtype={}",
        runtime_execution.compile_active,
        runtime_execution.amp_active,
        runtime_execution.amp_resolved_dtype,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_aucs: list[float] = []
    best_auc = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    progress_bar = None
    if show_progress:
        progress_bar = create_progress_bar(
            total=experiment.train.epochs,
            description=f"taac-train[{experiment.name}]",
        )

    try:
        for epoch in range(1, experiment.train.epochs + 1):
            execution_model.train()
            batch_losses: list[float] = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with runtime_execution.autocast_context():
                    logits = execution_model(batch)
                    loss = loss_fn(logits, batch.labels)
                if getattr(auxiliary_loss, "enabled", False) and getattr(auxiliary_loss, "requires_aux", False):
                    raise RuntimeError("Auxiliary losses requiring extra tensors are not implemented")
                runtime_execution.backward_and_step(
                    loss,
                    optimizer,
                    model_parameters=model.parameters(),
                    grad_clip_norm=experiment.train.grad_clip_norm,
                )
                batch_losses.append(float(loss.detach().cpu().item()))

            train_loss = safe_mean(batch_losses)
            val_logits, val_labels, val_groups, val_loss = collect_loader_outputs(
                execution_model,
                val_loader,
                device,
                loss_fn,
                runtime_execution=runtime_execution,
            )
            val_metrics = compute_classification_metrics(val_labels, val_logits, val_groups)
            val_auc = float(val_metrics["auc"])

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_aucs.append(val_auc)

            improved = val_auc >= best_auc
            if improved:
                best_auc = val_auc
                best_epoch = epoch
                best_metrics = dict(val_metrics)
                _write_best_checkpoint(
                    output_dir / "best.pt",
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": val_metrics,
                        "runtime_optimization": runtime_execution.summary(),
                    },
                )

            write_training_curve_artifacts(
                output_dir=output_dir,
                train_losses=train_losses,
                val_losses=val_losses,
                val_aucs=val_aucs,
                best_epoch=best_epoch,
            )

            epoch_status = {
                "epoch": f"{epoch}/{experiment.train.epochs}",
                "train_loss": f"{train_loss:.4f}",
                "val_auc": f"{val_auc:.4f}",
                "best": f"{best_auc:.4f}",
            }
            if progress_bar is not None:
                progress_bar.update()
                progress_bar.set_postfix(epoch_status, refresh=False)
            else:
                logger.info(
                    "epoch {}/{} train_loss={:.4f} val_loss={:.4f} val_auc={:.4f} best_auc={:.4f} improved={}",
                    epoch,
                    experiment.train.epochs,
                    train_loss,
                    val_loss,
                    val_auc,
                    best_auc,
                    improved,
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    latency = measure_latency(
        execution_model,
        val_loader,
        device,
        warmup_steps=experiment.train.latency_warmup_steps,
        measure_steps=experiment.train.latency_measure_steps,
        runtime_execution=runtime_execution,
    )
    inference_profile = collect_inference_profile(experiment, int(data_stats.val_size), latency)
    compute_profile = collect_compute_profile(
        experiment=experiment,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        data_stats=data_stats,
        device=device,
        model_profile=model_profile,
        latency=latency,
        runtime_execution=runtime_execution,
    )
    external_profilers = build_training_external_profiler_plan(
        device=str(device),
        output_dir=output_dir,
        experiment_path=experiment_path,
        train_config=experiment.train,
    )
    write_external_profiler_plan_artifacts(external_profilers)
    profiling = build_profiling_report(
        device=device,
        latency=latency,
        model_profile=model_profile,
        inference_profile=inference_profile,
        compute_profile=compute_profile,
        external_profilers=external_profilers,
    )

    summary = {
        "model_name": experiment.model.name,
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "metrics": best_metrics,
        "profiling": profiling,
        "runtime_optimization": runtime_execution.summary(),
        "model_profile": model_profile,
        "compute_profile": compute_profile,
        "inference_profile": inference_profile,
        **latency,
    }

    write_json(output_dir / "summary.json", summary)
    logger.info(
        "training complete: experiment={} best_epoch={} best_val_auc={:.6f} latency_ms={:.4f}",
        experiment.name,
        best_epoch,
        best_auc,
        float(summary["mean_latency_ms_per_sample"]),
    )
    return summary


__all__ = ["run_training"]

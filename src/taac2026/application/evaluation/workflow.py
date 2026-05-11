"""Composable PCVR prediction stack hooks."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader

import taac2026.infrastructure.data.dataset as pcvr_data
from taac2026.domain.runtime_config import RuntimeExecutionConfig
from taac2026.infrastructure.logging import logger
from taac2026.infrastructure.checkpoints import load_checkpoint_state_dict
from taac2026.infrastructure.modeling.model_contract import batch_to_model_input, build_pcvr_model, parse_seq_max_lens
from taac2026.infrastructure.modeling.sequence import configure_flash_attention_runtime as configure_shared_flash_attention_runtime
from taac2026.infrastructure.modeling.tensors import sigmoid_probabilities_numpy
from taac2026.infrastructure.runtime.execution import (
    maybe_compile_callable,
    maybe_prepare_internal_compile,
    runtime_autocast_context,
    runtime_execution_summary,
)


_PREDICTION_PROGRESS_LOG_EVERY_ROWS = 50_000


def _log_prediction_progress(
    *,
    mode: str,
    processed_rows: int,
    total_rows: int,
    batch_index: int,
    total_batches: int,
    elapsed_seconds: float,
) -> None:
    progress = 100.0 * processed_rows / total_rows if total_rows > 0 else 0.0
    logger.info(
        "PCVR {} progress: {}/{} rows ({:.1f}%), batch {}/{}, elapsed={:.1f}s",
        mode,
        processed_rows,
        total_rows,
        progress,
        batch_index,
        total_batches,
        elapsed_seconds,
    )


@dataclass(slots=True)
class PCVRPredictionContext:
    model_module: Any
    model_class_name: str
    package_dir: Path
    dataset_path: Path
    schema_path: Path
    checkpoint_path: Path
    batch_size: int
    num_workers: int
    device: str
    is_training_data: bool
    dataset_role: str
    config: dict[str, Any]
    runtime_execution: RuntimeExecutionConfig

    @property
    def mode(self) -> str:
        return "evaluation" if self.is_training_data else "inference"

    @property
    def runtime_device(self) -> torch.device:
        return torch.device(self.device)


@dataclass(frozen=True, slots=True)
class PCVRPredictionDataBundle:
    dataset: Any
    loader: Any
    data_module: Any = pcvr_data


@dataclass(frozen=True, slots=True)
class PCVRPredictionRunner:
    model: Any
    predict_fn: Any


def default_build_prediction_data(
    context: PCVRPredictionContext,
) -> PCVRPredictionDataBundle:
    pcvr_data.ensure_torch_file_system_sharing_strategy()
    seq_max_lens = parse_seq_max_lens(str(context.config["seq_max_lens"]))
    dataset = pcvr_data.PCVRParquetDataset(
        parquet_path=str(context.dataset_path.expanduser().resolve()),
        schema_path=str(context.schema_path),
        batch_size=context.batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        clip_vocab=True,
        is_training=context.is_training_data,
        dataset_role=context.dataset_role,
    )
    use_cuda_pinning = context.device.startswith("cuda") and torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=context.num_workers,
        pin_memory=use_cuda_pinning,
    )
    return PCVRPredictionDataBundle(dataset=dataset, loader=loader)


def default_build_prediction_model(
    context: PCVRPredictionContext,
    data_bundle: PCVRPredictionDataBundle,
) -> Any:
    configure_shared_flash_attention_runtime(
        backend=str(context.config.get("flash_attention_backend", "torch")),
    )
    return build_pcvr_model(
        model_module=context.model_module,
        model_class_name=context.model_class_name,
        data_module=data_bundle.data_module,
        dataset=data_bundle.dataset,
        config=context.config,
        package_dir=context.package_dir,
        checkpoint_dir=context.checkpoint_path.parent,
    )


def default_prepare_prediction_runner(
    context: PCVRPredictionContext,
    data_bundle: PCVRPredictionDataBundle,
    model: Any,
) -> PCVRPredictionRunner:
    del data_bundle
    model.to(context.runtime_device)
    state_dict = load_checkpoint_state_dict(
        context.checkpoint_path,
        map_location=context.runtime_device,
    )
    model.load_state_dict(state_dict)
    model.eval()
    uses_internal_compile = maybe_prepare_internal_compile(
        model,
        enabled=context.runtime_execution.compile,
        label=f"PCVR {context.mode} model",
    )
    predict_fn = maybe_compile_callable(
        model.predict,
        enabled=context.runtime_execution.compile and not uses_internal_compile,
        label=f"PCVR {context.mode} predict",
    )
    return PCVRPredictionRunner(model=model, predict_fn=predict_fn)


def default_run_prediction_loop(
    context: PCVRPredictionContext,
    data_bundle: PCVRPredictionDataBundle,
    runner: PCVRPredictionRunner,
) -> dict[str, Any]:
    total_rows = int(getattr(data_bundle.dataset, "num_rows", 0))
    total_batches = (total_rows + context.batch_size - 1) // context.batch_size if total_rows > 0 else 0
    logger.info(
        "PCVR {} loop starting: checkpoint={}, rows={}, estimated_batches={}, batch_size={}, num_workers={}, device={}, runtime={}",
        context.mode,
        context.checkpoint_path,
        total_rows,
        total_batches,
        context.batch_size,
        context.num_workers,
        context.device,
        runtime_execution_summary(context.runtime_execution, context.runtime_device),
    )

    labels: list[float] = []
    probabilities: list[float] = []
    records: list[dict[str, Any]] = []
    processed_rows = 0
    batch_count = 0
    progress_log_every_rows = max(_PREDICTION_PROGRESS_LOG_EVERY_ROWS, context.batch_size)
    next_progress_log_rows = progress_log_every_rows
    started_at = time.perf_counter()
    with torch.inference_mode():
        for batch_count, batch in enumerate(data_bundle.loader, start=1):
            model_input = batch_to_model_input(batch, context.model_module.ModelInput, context.runtime_device)
            with runtime_autocast_context(context.runtime_execution, context.runtime_device):
                logits, _embeddings = runner.predict_fn(model_input)
            batch_probabilities = sigmoid_probabilities_numpy(logits.squeeze(-1))
            batch_labels = batch["label"].detach().cpu().numpy() if "label" in batch else np.zeros_like(batch_probabilities)
            batch_user_ids = batch.get("user_id", list(range(len(batch_probabilities))))
            batch_timestamps = batch.get("timestamp")
            if isinstance(batch_timestamps, torch.Tensor):
                timestamp_values = batch_timestamps.detach().cpu().numpy().tolist()
            else:
                timestamp_values = [None] * len(batch_probabilities)
            for row_index, probability in enumerate(batch_probabilities.tolist()):
                label = float(batch_labels[row_index])
                user_id = batch_user_ids[row_index]
                labels.append(label)
                probabilities.append(float(probability))
                records.append(
                    {
                        "sample_index": len(records),
                        "user_id": str(user_id),
                        "score": float(probability),
                        "target": label,
                        "timestamp": timestamp_values[row_index],
                    }
                )
            processed_rows += len(batch_probabilities)
            if total_rows > 0 and processed_rows >= next_progress_log_rows:
                _log_prediction_progress(
                    mode=context.mode,
                    processed_rows=processed_rows,
                    total_rows=total_rows,
                    batch_index=batch_count,
                    total_batches=total_batches,
                    elapsed_seconds=time.perf_counter() - started_at,
                )
                while next_progress_log_rows <= processed_rows:
                    next_progress_log_rows += progress_log_every_rows
    logger.info(
        "PCVR {} loop completed: rows={}, batches={}, elapsed={:.1f}s",
        context.mode,
        processed_rows,
        batch_count,
        time.perf_counter() - started_at,
    )
    return {
        "labels": labels,
        "probabilities": probabilities,
        "records": records,
        "processed_rows": processed_rows,
        "batch_count": batch_count,
        "loop_elapsed_sec": time.perf_counter() - started_at,
    }


class BuildPredictionDataHook(Protocol):
    def __call__(self, context: PCVRPredictionContext) -> PCVRPredictionDataBundle:
        ...


class BuildPredictionModelHook(Protocol):
    def __call__(self, context: PCVRPredictionContext, data_bundle: PCVRPredictionDataBundle) -> Any:
        ...


class PreparePredictionRunnerHook(Protocol):
    def __call__(
        self,
        context: PCVRPredictionContext,
        data_bundle: PCVRPredictionDataBundle,
        model: Any,
    ) -> PCVRPredictionRunner:
        ...


class RunPredictionLoopHook(Protocol):
    def __call__(
        self,
        context: PCVRPredictionContext,
        data_bundle: PCVRPredictionDataBundle,
        runner: PCVRPredictionRunner,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class PCVRPredictionHooks:
    build_data: BuildPredictionDataHook = default_build_prediction_data
    build_model: BuildPredictionModelHook = default_build_prediction_model
    prepare_predictor: PreparePredictionRunnerHook = default_prepare_prediction_runner
    run_loop: RunPredictionLoopHook = default_run_prediction_loop


DEFAULT_PCVR_PREDICTION_HOOKS = PCVRPredictionHooks()


def build_pcvr_prediction_hooks(**overrides: Any) -> PCVRPredictionHooks:
    return replace(DEFAULT_PCVR_PREDICTION_HOOKS, **overrides)


__all__ = [
    "DEFAULT_PCVR_PREDICTION_HOOKS",
    "BuildPredictionDataHook",
    "BuildPredictionModelHook",
    "PCVRPredictionContext",
    "PCVRPredictionDataBundle",
    "PCVRPredictionHooks",
    "PCVRPredictionRunner",
    "PreparePredictionRunnerHook",
    "RunPredictionLoopHook",
    "_log_prediction_progress",
    "build_pcvr_prediction_hooks",
    "default_build_prediction_data",
    "default_build_prediction_model",
    "default_prepare_prediction_runner",
    "default_run_prediction_loop",
]

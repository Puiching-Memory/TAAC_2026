"""PCVR experiment adapter for plugin packages."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from taac2026.domain.requests import EvalRequest, InferRequest, TrainRequest
from taac2026.domain.metrics import compute_classification_metrics
from taac2026.infrastructure.io.files import write_json
from taac2026.infrastructure.io.json import dump_bytes
from taac2026.domain.config import PCVRTrainConfig
from taac2026.application.experiments.runtime import PCVRExperimentRuntimeMixin
from taac2026.application.evaluation.workflow import PCVRPredictionHooks, _log_prediction_progress
from taac2026.application.evaluation.runtime import PCVRRuntimeHooks
from taac2026.infrastructure.data.sample_dataset import resolve_default_pcvr_sample_paths
from taac2026.infrastructure.logging import logger
from taac2026.infrastructure.runtime.telemetry import RuntimeTelemetry, file_size_mb
from taac2026.application.training.workflow import PCVRTrainHooks
from taac2026.application.training.args import train_pcvr_model


_EVAL_AUC_BOOTSTRAP_SAMPLES = 200


def _callable_name(value: Any) -> str:
    return getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))


@dataclass(slots=True)
class PCVRExperiment(PCVRExperimentRuntimeMixin):
    name: str
    package_dir: Path
    model_class_name: str
    train_defaults: PCVRTrainConfig
    train_arg_parser: Callable[[Sequence[str] | None], Any]
    train_hooks: PCVRTrainHooks
    prediction_hooks: PCVRPredictionHooks
    runtime_hooks: PCVRRuntimeHooks

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "kind": "pcvr",
            "model_class": self.model_class_name,
            "source": str(self.package_dir),
            "train_arg_parser": _callable_name(self.train_arg_parser),
            "train_build_data": _callable_name(self.train_hooks.build_data),
            "train_build_model": _callable_name(self.train_hooks.build_model),
            "train_build_trainer": _callable_name(self.train_hooks.build_trainer),
            "train_run_training": _callable_name(self.train_hooks.run_training),
            "prediction_build_data": _callable_name(self.prediction_hooks.build_data),
            "prediction_build_model": _callable_name(self.prediction_hooks.build_model),
            "prediction_prepare_predictor": _callable_name(self.prediction_hooks.prepare_predictor),
            "prediction_run_loop": _callable_name(self.prediction_hooks.run_loop),
            "runtime_resolve_evaluation_checkpoint": _callable_name(self.runtime_hooks.resolve_evaluation_checkpoint),
            "runtime_resolve_inference_checkpoint": _callable_name(self.runtime_hooks.resolve_inference_checkpoint),
            "runtime_load_train_config": _callable_name(self.runtime_hooks.load_train_config),
            "runtime_load_runtime_schema": _callable_name(self.runtime_hooks.load_runtime_schema),
            "runtime_build_evaluation_data_diagnostics": _callable_name(self.runtime_hooks.build_evaluation_data_diagnostics),
            "runtime_write_observed_schema_report": _callable_name(self.runtime_hooks.write_observed_schema_report),
            "runtime_write_train_split_observed_schema_reports": _callable_name(self.runtime_hooks.write_train_split_observed_schema_reports),
        }

    @contextmanager
    def _module_context(self) -> Iterator[None]:
        yield

    def train(self, request: TrainRequest) -> Mapping[str, Any]:
        resolved_dataset_path, resolved_schema_override = resolve_default_pcvr_sample_paths(
            request.dataset_path,
            request.schema_path,
        )
        run_dir = request.run_dir.expanduser().resolve()
        train_log_dir = Path(os.environ.get("TRAIN_LOG_PATH", str(run_dir / "logs"))).expanduser().resolve()
        tensorboard_dir = Path(os.environ.get("TRAIN_TF_EVENTS_PATH", str(run_dir / "tensorboard"))).expanduser().resolve()

        forwarded_args = [
            "--data_dir",
            str(resolved_dataset_path),
            "--ckpt_dir",
            str(run_dir),
            "--log_dir",
            str(train_log_dir),
            "--tf_events_dir",
            str(tensorboard_dir),
        ]
        if resolved_schema_override is not None:
            forwarded_args.extend(["--schema_path", str(resolved_schema_override)])
        forwarded_args.extend(request.extra_args)

        with self._module_context():
            model_module = self._load_model_module()

            summary = dict(train_pcvr_model(
                model_module=model_module,
                model_class_name=self.model_class_name,
                package_dir=self.package_dir,
                defaults=self.train_defaults,
                arg_parser=self.train_arg_parser,
                train_hooks=self.train_hooks,
                argv=forwarded_args,
            ) or {})

        resolved_schema_path = Path(summary["schema_path"]).expanduser().resolve()

        observed_schema_payload = self.runtime_hooks.write_train_split_observed_schema_reports(
            self,
            dataset_path=resolved_dataset_path,
            schema_path=resolved_schema_path,
            run_dir=run_dir,
            valid_ratio=float(summary["valid_ratio"]),
            train_ratio=float(summary["train_ratio"]),
            split_strategy=str(summary.get("split_strategy", "row_group_tail")),
            train_timestamp_start=int(summary.get("train_timestamp_start", 0) or 0),
            train_timestamp_end=int(summary.get("train_timestamp_end", 0) or 0),
            valid_timestamp_start=int(summary.get("valid_timestamp_start", 0) or 0),
            valid_timestamp_end=int(summary.get("valid_timestamp_end", 0) or 0),
        )

        payload = dict(summary)
        payload["experiment_name"] = self.name
        payload["run_dir"] = str(run_dir)
        payload["checkpoint_root"] = str(run_dir)
        payload["schema_path"] = str(resolved_schema_path)
        payload.update(observed_schema_payload)
        return payload

    def evaluate(self, request: EvalRequest) -> Mapping[str, Any]:
        resolved_dataset_path, resolved_schema_override = resolve_default_pcvr_sample_paths(
            request.dataset_path,
            request.schema_path,
        )
        checkpoint = self.runtime_hooks.resolve_evaluation_checkpoint(self, request)
        output_path = request.output_path or (request.run_dir / "evaluation.json")
        predictions_path = request.predictions_path or (request.run_dir / "validation_predictions.jsonl")
        config = self.runtime_hooks.load_train_config(self, checkpoint.parent)
        resolved_schema_path, resolved_schema = self.runtime_hooks.load_runtime_schema(
            self,
            dataset_path=resolved_dataset_path,
            schema_path=resolved_schema_override,
            checkpoint_dir=checkpoint.parent,
            mode="evaluation",
        )
        effective_batch_size, batch_size_source, effective_num_workers, num_workers_source = self._resolve_prediction_runtime_settings(
            request,
            config,
        )
        runtime_execution, amp_source, amp_dtype_source, compile_source = self._resolve_prediction_runtime_execution(
            request,
            config,
        )
        logger.info(
            "Resolved PCVR evaluation runtime: experiment={}, checkpoint={}, batch_size={} ({}), num_workers={} ({}), amp={} ({}), amp_dtype={} ({}), compile={} ({})",
            self.name,
            checkpoint,
            effective_batch_size,
            batch_size_source,
            effective_num_workers,
            num_workers_source,
            runtime_execution.amp,
            amp_source,
            runtime_execution.normalized_amp_dtype(),
            amp_dtype_source,
            runtime_execution.compile,
            compile_source,
        )

        telemetry = RuntimeTelemetry(
            label="evaluation",
            device=request.device,
            metadata={
                "experiment_name": self.name,
                "checkpoint_path": str(checkpoint),
                "dataset_path": str(resolved_dataset_path),
            },
        ).start()
        with self._module_context():
            evaluation = self._run_prediction_loop(
                dataset_path=resolved_dataset_path,
                schema_path=resolved_schema_path,
                checkpoint_path=checkpoint,
                batch_size=effective_batch_size,
                num_workers=effective_num_workers,
                device=request.device,
                is_training_data=request.is_training_data,
                dataset_role="evaluation",
                config=config,
                runtime_execution=runtime_execution,
            )

        labels = np.asarray(evaluation["labels"], dtype=np.float64)
        probabilities = np.asarray(evaluation["probabilities"], dtype=np.float64)
        metrics = compute_classification_metrics(
            labels,
            probabilities,
            auc_bootstrap_samples=_EVAL_AUC_BOOTSTRAP_SAMPLES,
        )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("wb") as handle:
            for record in evaluation["records"]:
                handle.write(dump_bytes(record))
                handle.write(b"\n")
        payload = {
            "experiment_name": self.name,
            "checkpoint_path": str(checkpoint),
            "schema_path": str(resolved_schema_path),
            "schema": resolved_schema,
            "metrics": metrics,
            "data_diagnostics": self.runtime_hooks.build_evaluation_data_diagnostics(self, resolved_dataset_path),
            "validation_predictions_path": str(predictions_path),
            "batch_size": effective_batch_size,
            "num_workers": effective_num_workers,
        }
        telemetry_payload = telemetry.finish(
            rows=int(evaluation.get("processed_rows", len(probabilities))),
            batches=int(evaluation.get("batch_count", 0) or 0),
            prediction_file_mb=file_size_mb(predictions_path),
            checkpoint_file_mb=file_size_mb(checkpoint),
        )
        payload["telemetry"] = telemetry_payload
        observed_schema_path = output_path.with_name("evaluation_observed_schema.json")
        self.runtime_hooks.write_observed_schema_report(
            self,
            dataset_path=resolved_dataset_path,
            schema_path=resolved_schema_path,
            output_path=observed_schema_path,
            dataset_role="eval",
        )
        payload["observed_schema_paths"] = {"eval": str(observed_schema_path)}
        write_json(output_path, payload)
        write_json(output_path.with_name("evaluation_telemetry.json"), telemetry_payload)
        return payload

    def infer(self, request: InferRequest) -> Mapping[str, Any]:
        resolved_dataset_path, resolved_schema_override = resolve_default_pcvr_sample_paths(
            request.dataset_path,
            request.schema_path,
        )
        checkpoint = self.runtime_hooks.resolve_inference_checkpoint(self, request)
        config = self.runtime_hooks.load_train_config(self, checkpoint.parent)
        resolved_schema_path, resolved_schema = self.runtime_hooks.load_runtime_schema(
            self,
            dataset_path=resolved_dataset_path,
            schema_path=resolved_schema_override,
            checkpoint_dir=checkpoint.parent,
            mode="inference",
        )
        effective_batch_size, batch_size_source, effective_num_workers, num_workers_source = self._resolve_prediction_runtime_settings(
            request,
            config,
        )
        runtime_execution, amp_source, amp_dtype_source, compile_source = self._resolve_prediction_runtime_execution(
            request,
            config,
        )
        logger.info(
            "Resolved PCVR inference runtime: experiment={}, checkpoint={}, batch_size={} ({}), num_workers={} ({}), amp={} ({}), amp_dtype={} ({}), compile={} ({})",
            self.name,
            checkpoint,
            effective_batch_size,
            batch_size_source,
            effective_num_workers,
            num_workers_source,
            runtime_execution.amp,
            amp_source,
            runtime_execution.normalized_amp_dtype(),
            amp_dtype_source,
            runtime_execution.compile,
            compile_source,
        )

        telemetry = RuntimeTelemetry(
            label="inference",
            device=request.device,
            metadata={
                "experiment_name": self.name,
                "checkpoint_path": str(checkpoint),
                "dataset_path": str(resolved_dataset_path),
            },
        ).start()
        with self._module_context():
            evaluation = self._run_prediction_loop(
                dataset_path=resolved_dataset_path,
                schema_path=resolved_schema_path,
                checkpoint_path=checkpoint,
                batch_size=effective_batch_size,
                num_workers=effective_num_workers,
                device=request.device,
                is_training_data=False,
                dataset_role="inference",
                config=config,
                runtime_execution=runtime_execution,
            )

        prediction_map = {
            str(record["user_id"]): float(record["score"])
            for record in evaluation["records"]
        }
        request.result_dir.mkdir(parents=True, exist_ok=True)
        output_path = request.result_dir / "predictions.json"
        write_json(output_path, {"predictions": prediction_map})
        telemetry_payload = telemetry.finish(
            rows=int(evaluation.get("processed_rows", len(prediction_map))),
            batches=int(evaluation.get("batch_count", 0) or 0),
            prediction_file_mb=file_size_mb(output_path),
            checkpoint_file_mb=file_size_mb(checkpoint),
        )
        write_json(request.result_dir / "inference_telemetry.json", telemetry_payload)
        return {
            "checkpoint_path": str(checkpoint),
            "schema_path": str(resolved_schema_path),
            "schema": resolved_schema,
            "predictions_path": str(output_path),
            "prediction_count": len(prediction_map),
            "batch_size": effective_batch_size,
            "num_workers": effective_num_workers,
            "telemetry": telemetry_payload,
        }


__all__ = ["PCVRExperiment", "_log_prediction_progress"]

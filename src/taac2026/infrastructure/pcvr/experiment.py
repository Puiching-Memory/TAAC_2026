"""PCVR experiment adapter for plugin packages."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from taac2026.domain.config import EvalRequest, InferRequest, TrainRequest
from taac2026.domain.metrics import compute_classification_metrics
from taac2026.infrastructure.checkpoints import resolve_checkpoint_path
from taac2026.infrastructure.io.files import read_json, write_json
import taac2026.infrastructure.pcvr.data as pcvr_data
from taac2026.infrastructure.pcvr.config import PCVRTrainConfig, REQUIRED_PCVR_TRAIN_CONFIG_KEYS
from taac2026.infrastructure.pcvr.protocol import (
    batch_to_model_input,
    build_pcvr_model,
    parse_seq_max_lens,
    resolve_schema_path,
)
from taac2026.infrastructure.pcvr.tensors import sigmoid_probabilities_numpy
from taac2026.infrastructure.pcvr.training import train_pcvr_model
from taac2026.infrastructure.training.runtime import RuntimeExecutionConfig, maybe_compile_callable, normalize_amp_dtype


_PLUGIN_MODULE_NAMES = ("utils", "model")
_INFER_REQUEST_DEFAULT_BATCH_SIZE = int(InferRequest.__dataclass_fields__["batch_size"].default)
_INFER_REQUEST_DEFAULT_NUM_WORKERS = int(InferRequest.__dataclass_fields__["num_workers"].default)
_PREDICTION_PROGRESS_LOG_EVERY_ROWS = 50_000
_EVAL_AUC_BOOTSTRAP_SAMPLES = 200


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _required_config_value(config: dict[str, Any], config_key: str) -> Any:
    try:
        return config[config_key]
    except KeyError as error:
        raise KeyError(f"PCVR train_config is missing required key: {config_key}") from error


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
    logging.info(
        "PCVR %s progress: %d/%d rows (%.1f%%), batch %d/%d, elapsed=%.1fs",
        mode,
        processed_rows,
        total_rows,
        progress,
        batch_index,
        total_batches,
        elapsed_seconds,
    )


@dataclass(slots=True)
class PCVRExperiment:
    name: str
    package_dir: Path
    model_class_name: str
    train_defaults: PCVRTrainConfig = field(default_factory=PCVRTrainConfig)

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "kind": "pcvr",
            "model_class": self.model_class_name,
            "source": str(self.package_dir),
        }

    @contextmanager
    def _module_context(self) -> Iterator[None]:
        package_path = str(self.package_dir)
        previous_path = list(sys.path)
        previous_modules = {name: sys.modules.get(name) for name in _PLUGIN_MODULE_NAMES}
        for module_name in _PLUGIN_MODULE_NAMES:
            sys.modules.pop(module_name, None)
        sys.path.insert(0, package_path)
        try:
            yield
        finally:
            sys.path[:] = previous_path
            for module_name in _PLUGIN_MODULE_NAMES:
                sys.modules.pop(module_name, None)
            for module_name, module in previous_modules.items():
                if module is not None:
                    sys.modules[module_name] = module

    def train(self, request: TrainRequest) -> Mapping[str, Any]:
        run_dir = request.run_dir.expanduser().resolve()
        train_log_dir = Path(os.environ.get("TRAIN_LOG_PATH", str(run_dir / "logs"))).expanduser().resolve()
        tensorboard_dir = Path(os.environ.get("TRAIN_TF_EVENTS_PATH", str(run_dir / "tensorboard"))).expanduser().resolve()

        forwarded_args = [
            "--data_dir",
            str(request.dataset_path.expanduser().resolve()),
            "--ckpt_dir",
            str(run_dir),
            "--log_dir",
            str(train_log_dir),
            "--tf_events_dir",
            str(tensorboard_dir),
        ]
        if request.schema_path is not None:
            forwarded_args.extend(["--schema_path", str(request.schema_path.expanduser().resolve())])
        forwarded_args.extend(request.extra_args)

        with self._module_context():
            import model as model_module

            train_pcvr_model(
                model_module=model_module,
                model_class_name=self.model_class_name,
                package_dir=self.package_dir,
                defaults=self.train_defaults,
                argv=forwarded_args,
            )

        return {
            "experiment_name": self.name,
            "run_dir": str(run_dir),
            "checkpoint_root": str(run_dir),
        }

    def evaluate(self, request: EvalRequest) -> Mapping[str, Any]:
        checkpoint = resolve_checkpoint_path(request.run_dir, request.checkpoint_path)
        output_path = request.output_path or (request.run_dir / "evaluation.json")
        predictions_path = request.predictions_path or (request.run_dir / "validation_predictions.jsonl")
        config = self._load_train_config(checkpoint.parent)
        effective_batch_size, batch_size_source, effective_num_workers, num_workers_source = self._resolve_prediction_runtime_settings(
            request,
            config,
        )
        runtime_execution, amp_source, amp_dtype_source, compile_source = self._resolve_prediction_runtime_execution(
            request,
            config,
        )
        logging.info(
            "Resolved PCVR evaluation runtime: experiment=%s, checkpoint=%s, batch_size=%d (%s), num_workers=%d (%s), amp=%s (%s), amp_dtype=%s (%s), compile=%s (%s)",
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

        with self._module_context():
            evaluation = self._run_prediction_loop(
                dataset_path=request.dataset_path,
                schema_path=request.schema_path,
                checkpoint_path=checkpoint,
                batch_size=effective_batch_size,
                num_workers=effective_num_workers,
                device=request.device,
                is_training_data=request.is_training_data,
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
        rows = [
            json.dumps(record, ensure_ascii=False)
            for record in evaluation["records"]
        ]
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        payload = {
            "experiment_name": self.name,
            "checkpoint_path": str(checkpoint),
            "metrics": metrics,
            "data_diagnostics": self._build_evaluation_data_diagnostics(request.dataset_path),
            "validation_predictions_path": str(predictions_path),
        }
        write_json(output_path, payload)
        return payload

    def _build_evaluation_data_diagnostics(self, dataset_path: Path) -> dict[str, Any]:
        resolved_dataset_path = dataset_path.expanduser()
        warnings: list[str] = []
        try:
            rg_info = pcvr_data.collect_pcvr_row_groups(resolved_dataset_path)
            split_plan = pcvr_data.plan_pcvr_row_group_split(rg_info)
        except (FileNotFoundError, OSError, ValueError) as error:
            return {
                "dataset_path": str(resolved_dataset_path),
                "warnings": [f"row group diagnostics unavailable: {error}"],
            }

        files = sorted({path for path, _index, _rows in rg_info})
        if split_plan.reuse_train_for_valid:
            warnings.append("single Row Group dataset would reuse train rows for validation; treat as L0 smoke only")
        if not split_plan.is_l1_ready:
            warnings.append("row group split is not suitable for L1 model comparison")

        return {
            "dataset_path": str(resolved_dataset_path.resolve()),
            "file_count": len(files),
            "total_row_groups": split_plan.total_row_groups,
            "total_rows": int(sum(rows for _path, _index, rows in rg_info)),
            "row_group_split": {
                "train_row_groups": split_plan.train_row_groups,
                "valid_row_groups": split_plan.valid_row_groups,
                "train_row_group_range": list(split_plan.train_row_group_range),
                "valid_row_group_range": list(split_plan.valid_row_group_range),
                "train_rows": split_plan.train_rows,
                "valid_rows": split_plan.valid_rows,
                "reuse_train_for_valid": split_plan.reuse_train_for_valid,
                "is_disjoint": split_plan.is_disjoint,
                "is_l1_ready": split_plan.is_l1_ready,
            },
            "warnings": warnings,
        }

    def infer(self, request: InferRequest) -> Mapping[str, Any]:
        checkpoint_root = Path(os.environ.get("MODEL_OUTPUT_PATH", "")).expanduser()
        checkpoint = resolve_checkpoint_path(Path.cwd(), request.checkpoint_path) if request.checkpoint_path else None
        if checkpoint is None and str(checkpoint_root) not in ("", ".") and checkpoint_root.exists():
            checkpoint = resolve_checkpoint_path(checkpoint_root)
        if checkpoint is None:
            checkpoint = resolve_checkpoint_path(Path.cwd())
        config = self._load_train_config(checkpoint.parent)
        effective_batch_size, batch_size_source, effective_num_workers, num_workers_source = self._resolve_prediction_runtime_settings(
            request,
            config,
        )
        runtime_execution, amp_source, amp_dtype_source, compile_source = self._resolve_prediction_runtime_execution(
            request,
            config,
        )
        logging.info(
            "Resolved PCVR inference runtime: experiment=%s, checkpoint=%s, batch_size=%d (%s), num_workers=%d (%s), amp=%s (%s), amp_dtype=%s (%s), compile=%s (%s)",
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

        with self._module_context():
            evaluation = self._run_prediction_loop(
                dataset_path=request.dataset_path,
                schema_path=request.schema_path,
                checkpoint_path=checkpoint,
                batch_size=effective_batch_size,
                num_workers=effective_num_workers,
                device=request.device,
                is_training_data=False,
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
        return {
            "checkpoint_path": str(checkpoint),
            "predictions_path": str(output_path),
            "prediction_count": len(prediction_map),
            "batch_size": effective_batch_size,
            "num_workers": effective_num_workers,
        }

    def _configured_infer_runtime_value(
        self,
        config: dict[str, Any],
        *,
        config_key: str,
        minimum: int,
    ) -> tuple[int, str]:
        configured_value = _coerce_optional_int(_required_config_value(config, config_key))
        if configured_value is None or configured_value < minimum:
            raise ValueError(f"PCVR train_config key {config_key!r} must be >= {minimum}, got {config.get(config_key)!r}")
        return configured_value, "train_config"

    def _resolve_prediction_runtime_settings(
        self,
        request: EvalRequest | InferRequest,
        config: dict[str, Any],
    ) -> tuple[int, str, int, str]:
        batch_size = int(request.batch_size)
        batch_size_source = "request" if request.batch_size != _INFER_REQUEST_DEFAULT_BATCH_SIZE else "cli_default"
        if batch_size_source == "cli_default":
            configured_batch_size, configured_batch_size_source = self._configured_infer_runtime_value(
                config,
                config_key="batch_size",
                minimum=1,
            )
            batch_size = configured_batch_size
            batch_size_source = configured_batch_size_source

        num_workers = int(request.num_workers)
        num_workers_source = "request" if request.num_workers != _INFER_REQUEST_DEFAULT_NUM_WORKERS else "cli_default"
        if num_workers_source == "cli_default":
            configured_num_workers, configured_num_workers_source = self._configured_infer_runtime_value(
                config,
                config_key="num_workers",
                minimum=0,
            )
            num_workers = configured_num_workers
            num_workers_source = configured_num_workers_source

        return batch_size, batch_size_source, num_workers, num_workers_source

    def _resolve_infer_runtime_settings(
        self,
        request: InferRequest,
        config: dict[str, Any],
    ) -> tuple[int, str, int, str]:
        return self._resolve_prediction_runtime_settings(request, config)

    def _configured_runtime_bool(
        self,
        request_value: bool | None,
        config: dict[str, Any],
        *,
        config_key: str,
    ) -> tuple[bool, str]:
        if request_value is not None:
            return bool(request_value), "request"

        configured_value = _required_config_value(config, config_key)
        if not isinstance(configured_value, bool):
            raise TypeError(f"PCVR train_config key {config_key!r} must be bool, got {type(configured_value).__name__}")
        return configured_value, "train_config"

    def _configured_runtime_string(
        self,
        request_value: str | None,
        config: dict[str, Any],
        *,
        config_key: str,
    ) -> tuple[str, str]:
        if request_value not in (None, ""):
            return normalize_amp_dtype(request_value), "request"

        configured_value = _required_config_value(config, config_key)
        if not isinstance(configured_value, str) or not configured_value.strip():
            raise TypeError(f"PCVR train_config key {config_key!r} must be a non-empty string")
        return normalize_amp_dtype(configured_value), "train_config"

    def _resolve_prediction_runtime_execution(
        self,
        request: EvalRequest | InferRequest,
        config: dict[str, Any],
    ) -> tuple[RuntimeExecutionConfig, str, str, str]:
        amp, amp_source = self._configured_runtime_bool(
            getattr(request, "amp", None),
            config,
            config_key="amp",
        )
        amp_dtype, amp_dtype_source = self._configured_runtime_string(
            getattr(request, "amp_dtype", None),
            config,
            config_key="amp_dtype",
        )
        compile_enabled, compile_source = self._configured_runtime_bool(
            getattr(request, "compile", None),
            config,
            config_key="compile",
        )
        return RuntimeExecutionConfig(amp=amp, amp_dtype=amp_dtype, compile=compile_enabled), amp_source, amp_dtype_source, compile_source

    def _run_prediction_loop(
        self,
        *,
        dataset_path: Path,
        schema_path: Path | None,
        checkpoint_path: Path,
        batch_size: int,
        num_workers: int,
        device: str,
        is_training_data: bool,
        config: dict[str, Any] | None = None,
        runtime_execution: RuntimeExecutionConfig | None = None,
    ) -> dict[str, Any]:
        import model as model_module

        resolved_schema_path = self._resolve_schema_path(dataset_path, schema_path, checkpoint_path.parent)
        resolved_config = config if config is not None else self._load_train_config(checkpoint_path.parent)
        seq_max_lens = parse_seq_max_lens(str(resolved_config["seq_max_lens"]))
        dataset = pcvr_data.PCVRParquetDataset(
            parquet_path=str(dataset_path.expanduser().resolve()),
            schema_path=str(resolved_schema_path),
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=False,
            buffer_batches=0,
            clip_vocab=True,
            is_training=is_training_data,
        )
        use_cuda_pinning = device.startswith("cuda") and torch.cuda.is_available()
        loader = DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=use_cuda_pinning)
        model = build_pcvr_model(
            model_module=model_module,
            model_class_name=self.model_class_name,
            data_module=pcvr_data,
            dataset=dataset,
            config=resolved_config,
            package_dir=self.package_dir,
            checkpoint_dir=checkpoint_path.parent,
        )
        runtime_device = torch.device(device)
        resolved_runtime_execution = runtime_execution or RuntimeExecutionConfig()
        model.to(runtime_device)
        state_dict = torch.load(checkpoint_path, map_location=runtime_device)
        model.load_state_dict(state_dict)
        model.eval()
        predict_fn = maybe_compile_callable(
            model.predict,
            enabled=resolved_runtime_execution.compile,
            label=f"PCVR {'evaluation' if is_training_data else 'inference'} predict",
        )

        mode = "evaluation" if is_training_data else "inference"
        total_rows = int(getattr(dataset, "num_rows", 0))
        total_batches = (total_rows + batch_size - 1) // batch_size if total_rows > 0 else 0
        logging.info(
            "PCVR %s loop starting: checkpoint=%s, rows=%d, estimated_batches=%d, batch_size=%d, num_workers=%d, device=%s, runtime=%s",
            mode,
            checkpoint_path,
            total_rows,
            total_batches,
            batch_size,
            num_workers,
            device,
            resolved_runtime_execution.summary(runtime_device),
        )

        labels: list[float] = []
        probabilities: list[float] = []
        records: list[dict[str, Any]] = []
        processed_rows = 0
        batch_count = 0
        progress_log_every_rows = max(_PREDICTION_PROGRESS_LOG_EVERY_ROWS, batch_size)
        next_progress_log_rows = progress_log_every_rows
        started_at = time.perf_counter()
        with torch.no_grad():
            for batch_count, batch in enumerate(loader, start=1):
                model_input = batch_to_model_input(batch, model_module.ModelInput, runtime_device)
                with resolved_runtime_execution.autocast_context(runtime_device):
                    logits, _embeddings = predict_fn(model_input)
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
                        mode=mode,
                        processed_rows=processed_rows,
                        total_rows=total_rows,
                        batch_index=batch_count,
                        total_batches=total_batches,
                        elapsed_seconds=time.perf_counter() - started_at,
                    )
                    while next_progress_log_rows <= processed_rows:
                        next_progress_log_rows += progress_log_every_rows
        logging.info(
            "PCVR %s loop completed: rows=%d, batches=%d, elapsed=%.1fs",
            mode,
            processed_rows,
            batch_count,
            time.perf_counter() - started_at,
        )
        return {"labels": labels, "probabilities": probabilities, "records": records}

    def _load_train_config(self, checkpoint_dir: Path) -> dict[str, Any]:
        config_path = checkpoint_dir / "train_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"PCVR train_config.json not found in checkpoint directory: {checkpoint_dir}")
        config = read_json(config_path)
        missing_keys = sorted(REQUIRED_PCVR_TRAIN_CONFIG_KEYS - set(config))
        if missing_keys:
            joined = ", ".join(missing_keys)
            raise KeyError(f"PCVR train_config.json is missing required key(s): {joined}")
        return config

    def _resolve_schema_path(self, dataset_path: Path, schema_path: Path | None, checkpoint_dir: Path) -> Path:
        return resolve_schema_path(dataset_path, schema_path, checkpoint_dir)

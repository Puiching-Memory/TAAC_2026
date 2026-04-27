"""PCVR experiment adapter for plugin packages."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from taac2026.domain.config import EvalRequest, InferRequest, TrainRequest
from taac2026.domain.metrics import binary_auc, binary_logloss
from taac2026.infrastructure.checkpoints import resolve_checkpoint_path
from taac2026.infrastructure.io.files import read_json, write_json
import taac2026.infrastructure.pcvr.data as pcvr_data
from taac2026.infrastructure.pcvr.protocol import (
    DEFAULT_PCVR_MODEL_CONFIG,
    batch_to_model_input,
    build_pcvr_model,
    parse_seq_max_lens,
    resolve_schema_path,
)
from taac2026.infrastructure.pcvr.training import train_pcvr_model
from taac2026.infrastructure.training.runtime import RuntimeExecutionConfig, maybe_compile_callable, normalize_amp_dtype


_PLUGIN_MODULE_NAMES = ("utils", "model")
_INFER_REQUEST_DEFAULT_BATCH_SIZE = int(InferRequest.__dataclass_fields__["batch_size"].default)
_INFER_REQUEST_DEFAULT_NUM_WORKERS = int(InferRequest.__dataclass_fields__["num_workers"].default)
_PREDICTION_PROGRESS_LOG_EVERY_ROWS = 50_000


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_flag_value(args: tuple[str, ...], flag_names: tuple[str, ...]) -> str | None:
    index = 0
    while index < len(args):
        token = args[index]
        has_value = index + 1 < len(args) and not args[index + 1].startswith("--")
        if token in flag_names:
            return args[index + 1] if has_value else None
        if token.startswith("--") and has_value:
            index += 2
        else:
            index += 1
    return None


def _read_bool_flag_value(
    args: tuple[str, ...],
    enabled_flags: tuple[str, ...],
    disabled_flags: tuple[str, ...],
) -> bool | None:
    resolved: bool | None = None
    for token in args:
        if token in enabled_flags:
            resolved = True
        elif token in disabled_flags:
            resolved = False
    return resolved


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
    default_train_args: tuple[str, ...] = ()

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
            *self.default_train_args,
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
        metrics = {
            "auc": binary_auc(labels, probabilities),
            "logloss": binary_logloss(labels, probabilities),
            "sample_count": int(labels.size),
        }
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
            "validation_predictions_path": str(predictions_path),
        }
        write_json(output_path, payload)
        return payload

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
        flag_names: tuple[str, ...],
        minimum: int,
    ) -> tuple[int | None, str | None]:
        configured_value = _coerce_optional_int(config.get(config_key))
        if configured_value is not None and configured_value >= minimum:
            return configured_value, "train_config"

        default_arg_value = _coerce_optional_int(_read_flag_value(self.default_train_args, flag_names))
        if default_arg_value is not None and default_arg_value >= minimum:
            return default_arg_value, "experiment_default_train_args"

        return None, None

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
                flag_names=("--batch_size", "--batch-size"),
                minimum=1,
            )
            if configured_batch_size is not None and configured_batch_size_source is not None:
                batch_size = configured_batch_size
                batch_size_source = configured_batch_size_source

        num_workers = int(request.num_workers)
        num_workers_source = "request" if request.num_workers != _INFER_REQUEST_DEFAULT_NUM_WORKERS else "cli_default"
        if num_workers_source == "cli_default":
            configured_num_workers, configured_num_workers_source = self._configured_infer_runtime_value(
                config,
                config_key="num_workers",
                flag_names=("--num_workers", "--num-workers"),
                minimum=0,
            )
            if configured_num_workers is not None and configured_num_workers_source is not None:
                num_workers = configured_num_workers
                num_workers_source = configured_num_workers_source

        return batch_size, batch_size_source, num_workers, num_workers_source

    def _resolve_infer_runtime_settings(
        self,
        request: InferRequest,
        config: dict[str, Any],
    ) -> tuple[int, str, int, str]:
        return self._resolve_prediction_runtime_settings(request, config)

    def _configured_runtime_flag(
        self,
        request_value: bool | None,
        config: dict[str, Any],
        *,
        config_key: str,
        enabled_flags: tuple[str, ...],
        disabled_flags: tuple[str, ...],
        default: bool,
    ) -> tuple[bool, str]:
        if request_value is not None:
            return bool(request_value), "request"

        configured_value = config.get(config_key)
        if isinstance(configured_value, bool):
            return configured_value, "train_config"

        default_arg_value = _read_bool_flag_value(self.default_train_args, enabled_flags, disabled_flags)
        if default_arg_value is not None:
            return default_arg_value, "experiment_default_train_args"

        return default, "default"

    def _configured_runtime_string(
        self,
        request_value: str | None,
        config: dict[str, Any],
        *,
        config_key: str,
        flag_names: tuple[str, ...],
        default: str,
    ) -> tuple[str, str]:
        if request_value not in (None, ""):
            return normalize_amp_dtype(request_value), "request"

        configured_value = config.get(config_key)
        if isinstance(configured_value, str) and configured_value.strip():
            return normalize_amp_dtype(configured_value), "train_config"

        default_arg_value = _read_flag_value(self.default_train_args, flag_names)
        if default_arg_value not in (None, ""):
            return normalize_amp_dtype(default_arg_value), "experiment_default_train_args"

        return normalize_amp_dtype(default), "default"

    def _resolve_prediction_runtime_execution(
        self,
        request: EvalRequest | InferRequest,
        config: dict[str, Any],
    ) -> tuple[RuntimeExecutionConfig, str, str, str]:
        amp, amp_source = self._configured_runtime_flag(
            getattr(request, "amp", None),
            config,
            config_key="amp",
            enabled_flags=("--amp",),
            disabled_flags=("--no-amp",),
            default=False,
        )
        amp_dtype, amp_dtype_source = self._configured_runtime_string(
            getattr(request, "amp_dtype", None),
            config,
            config_key="amp_dtype",
            flag_names=("--amp_dtype", "--amp-dtype"),
            default="bfloat16",
        )
        compile_enabled, compile_source = self._configured_runtime_flag(
            getattr(request, "compile", None),
            config,
            config_key="compile",
            enabled_flags=("--compile",),
            disabled_flags=("--no-compile",),
            default=False,
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
        seq_max_lens = parse_seq_max_lens(str(resolved_config.get("seq_max_lens", "")))
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
                batch_probabilities = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
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
        config = dict(DEFAULT_PCVR_MODEL_CONFIG)
        config_path = checkpoint_dir / "train_config.json"
        if config_path.exists():
            stored_config = read_json(config_path)
            config.update(stored_config)
        return config

    def _resolve_schema_path(self, dataset_path: Path, schema_path: Path | None, checkpoint_dir: Path) -> Path:
        return resolve_schema_path(dataset_path, schema_path, checkpoint_dir)

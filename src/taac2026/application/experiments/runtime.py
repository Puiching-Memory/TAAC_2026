"""Runtime helpers extracted from the PCVR experiment adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from taac2026.domain.requests import EvalRequest, InferRequest
from taac2026.application.evaluation.workflow import PCVRPredictionContext
from taac2026.infrastructure.runtime.execution import RuntimeExecutionConfig, normalize_amp_dtype


_INFER_REQUEST_DEFAULT_BATCH_SIZE = int(InferRequest.__dataclass_fields__["batch_size"].default)
_INFER_REQUEST_DEFAULT_NUM_WORKERS = int(InferRequest.__dataclass_fields__["num_workers"].default)


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


class PCVRExperimentRuntimeMixin:
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
        dataset_role: str,
        config: dict[str, Any] | None = None,
        runtime_execution: RuntimeExecutionConfig | None = None,
    ) -> dict[str, Any]:
        model_module = importlib.import_module("model")

        if schema_path is None:
            resolved_schema_path, _resolved_schema = self.runtime_hooks.load_runtime_schema(
                self,
                dataset_path=dataset_path,
                schema_path=None,
                checkpoint_dir=checkpoint_path.parent,
                mode="evaluation" if is_training_data else "inference",
            )
        else:
            resolved_schema_path = schema_path.expanduser().resolve()
        resolved_config = config if config is not None else self.runtime_hooks.load_train_config(self, checkpoint_path.parent)
        resolved_runtime_execution = runtime_execution or RuntimeExecutionConfig()
        context = PCVRPredictionContext(
            model_module=model_module,
            model_class_name=self.model_class_name,
            package_dir=self.package_dir,
            dataset_path=dataset_path,
            schema_path=resolved_schema_path,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            is_training_data=is_training_data,
            dataset_role=dataset_role,
            config=resolved_config,
            runtime_execution=resolved_runtime_execution,
        )
        data_bundle = self.prediction_hooks.build_data(context)
        model = self.prediction_hooks.build_model(context, data_bundle)
        runner = self.prediction_hooks.prepare_predictor(context, data_bundle, model)
        return self.prediction_hooks.run_loop(context, data_bundle, runner)

    def _load_train_config(self, checkpoint_dir: Path) -> dict[str, Any]:
        return self.runtime_hooks.load_train_config(self, checkpoint_dir)

    def _load_resolved_schema(
        self,
        *,
        dataset_path: Path,
        schema_path: Path | None,
        checkpoint_dir: Path,
        mode: str,
    ) -> tuple[Path, Any]:
        return self.runtime_hooks.load_runtime_schema(
            self,
            dataset_path=dataset_path,
            schema_path=schema_path,
            checkpoint_dir=checkpoint_dir,
            mode=mode,
        )

    def _write_observed_schema_report(
        self,
        *,
        dataset_path: Path,
        schema_path: Path,
        output_path: Path,
        dataset_role: str,
        row_group_range: tuple[int, int] | None = None,
    ) -> Path:
        return self.runtime_hooks.write_observed_schema_report(
            self,
            dataset_path=dataset_path,
            schema_path=schema_path,
            output_path=output_path,
            dataset_role=dataset_role,
            row_group_range=row_group_range,
        )

    def _write_train_split_observed_schema_reports(
        self,
        *,
        dataset_path: Path,
        schema_path: Path,
        run_dir: Path,
        valid_ratio: float,
        train_ratio: float,
    ) -> dict[str, Any]:
        return self.runtime_hooks.write_train_split_observed_schema_reports(
            self,
            dataset_path=dataset_path,
            schema_path=schema_path,
            run_dir=run_dir,
            valid_ratio=valid_ratio,
            train_ratio=train_ratio,
        )

    def _resolve_schema_path(self, dataset_path: Path, schema_path: Path | None, checkpoint_dir: Path) -> Path:
        resolved_schema_path, _schema_payload = self.runtime_hooks.load_runtime_schema(
            self,
            dataset_path=dataset_path,
            schema_path=schema_path,
            checkpoint_dir=checkpoint_dir,
            mode="runtime",
        )
        return resolved_schema_path


__all__ = ["PCVRExperimentRuntimeMixin"]
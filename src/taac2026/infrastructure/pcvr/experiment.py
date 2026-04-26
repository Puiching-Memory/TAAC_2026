"""PCVR experiment adapter for plugin packages."""

from __future__ import annotations

import json
import os
import sys
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


_PLUGIN_MODULE_NAMES = ("utils", "model")


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

        with self._module_context():
            evaluation = self._run_prediction_loop(
                dataset_path=request.dataset_path,
                schema_path=request.schema_path,
                checkpoint_path=checkpoint,
                batch_size=request.batch_size,
                num_workers=request.num_workers,
                device=request.device,
                is_training_data=request.is_training_data,
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

        with self._module_context():
            evaluation = self._run_prediction_loop(
                dataset_path=request.dataset_path,
                schema_path=request.schema_path,
                checkpoint_path=checkpoint,
                batch_size=request.batch_size,
                num_workers=request.num_workers,
                device=request.device,
                is_training_data=False,
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
        }

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
    ) -> dict[str, Any]:
        import model as model_module

        resolved_schema_path = self._resolve_schema_path(dataset_path, schema_path, checkpoint_path.parent)
        config = self._load_train_config(checkpoint_path.parent)
        seq_max_lens = parse_seq_max_lens(str(config.get("seq_max_lens", "")))
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
            config=config,
            package_dir=self.package_dir,
            checkpoint_dir=checkpoint_path.parent,
        )
        runtime_device = torch.device(device)
        model.to(runtime_device)
        state_dict = torch.load(checkpoint_path, map_location=runtime_device)
        model.load_state_dict(state_dict)
        model.eval()

        labels: list[float] = []
        probabilities: list[float] = []
        records: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                model_input = batch_to_model_input(batch, model_module.ModelInput, runtime_device)
                logits, _embeddings = model.predict(model_input)
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

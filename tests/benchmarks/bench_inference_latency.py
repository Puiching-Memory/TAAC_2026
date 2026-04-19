from __future__ import annotations

import torch
from torch import nn

from taac2026.domain.config import ModelConfig
from taac2026.domain.features import build_default_feature_schema
from taac2026.infrastructure.io.default_data_pipeline import load_dataloaders
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter
from taac2026.infrastructure.nn.quantization import quantize_model_for_inference
from tests.support import TinyExperimentModel


SPARSE_TABLE_NAMES = (
    "user_tokens",
    "context_tokens",
    "candidate_tokens",
    "candidate_post_tokens",
    "candidate_author_tokens",
)


def _estimate_state_dict_memory_mb(model: nn.Module) -> float:
    total_bytes = 0
    for value in model.state_dict().values():
        if torch.is_tensor(value):
            total_bytes += value.numel() * value.element_size()
    return float(total_bytes) / (1024.0 * 1024.0)


class TinyTorchRecInferenceModel(nn.Module):
    def __init__(self, model_config: ModelConfig, benchmark_workspace) -> None:
        super().__init__()
        feature_schema = build_default_feature_schema(
            benchmark_workspace.data_config,
            model_config,
        )
        self.embedding = TorchRecEmbeddingBagAdapter(
            feature_schema=feature_schema,
            table_names=SPARSE_TABLE_NAMES,
        )
        self.output = nn.Sequential(
            nn.Linear(self.embedding.output_dim + benchmark_workspace.data_config.dense_feature_dim, model_config.hidden_dim),
            nn.SiLU(),
            nn.Linear(model_config.hidden_dim, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        if batch.sparse_features is None:
            raise RuntimeError("Benchmark batch is missing sparse_features")
        pooled = self.embedding(batch.sparse_features)
        fused = torch.cat([pooled, batch.dense_features], dim=-1)
        return self.output(fused).squeeze(-1)


def test_inference_latency_baseline(benchmark, benchmark_device, benchmark_workspace, cuda_timer, performance_recorder) -> None:
    model_config = ModelConfig(name="tiny_benchmark", **benchmark_workspace.model_kwargs)
    model = TinyExperimentModel(benchmark_workspace.data_config, model_config, benchmark_workspace.data_config.dense_feature_dim)
    model = model.to(benchmark_device)
    model.eval()

    train_loader, _, _ = load_dataloaders(
        config=benchmark_workspace.data_config,
        vocab_size=model_config.vocab_size,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        seed=7,
    )
    batch = next(iter(train_loader)).to(benchmark_device)

    def run() -> torch.Tensor:
        with torch.inference_mode():
            return model(batch)

    benchmark.extra_info.update({
        "component": "inference",
        "phase": "baseline",
        "label": "phase-0",
        "model": "tiny_baseline",
        "metric": "latency",
    })
    benchmark(run)

    stats = cuda_timer(run)
    stats.update({
        "name": "inference_latency",
        "component": "inference",
        "phase": "baseline",
        "label": "phase-0",
        "model": "tiny_baseline",
        "throughput": float(batch.batch_size) / max(stats["median_ms"] / 1e3, 1e-9),
    })
    performance_recorder("inference_latency", stats)


def test_inference_latency_int8_quantized_torchrec(benchmark, benchmark_workspace, cuda_timer, performance_recorder) -> None:
    model_config = ModelConfig(name="tiny_torchrec_benchmark", **benchmark_workspace.model_kwargs)
    model = TinyTorchRecInferenceModel(model_config, benchmark_workspace)
    model.eval()

    train_loader, _, _ = load_dataloaders(
        config=benchmark_workspace.data_config,
        vocab_size=model_config.vocab_size,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        seed=7,
    )
    batch = next(iter(train_loader)).to("cpu")
    quantized_model, quantization_summary = quantize_model_for_inference(model, "int8")
    assert quantization_summary["active"] is True

    def run() -> torch.Tensor:
        with torch.inference_mode():
            return quantized_model(batch)

    benchmark.extra_info.update({
        "component": "quantization",
        "phase": "phase-6",
        "label": "phase-6",
        "model": "tiny_torchrec_int8",
        "metric": "latency",
    })
    benchmark(run)

    stats = cuda_timer(run)
    stats.update({
        "name": "quantized_inference_latency",
        "component": "quantization",
        "phase": "phase-6",
        "label": "phase-6",
        "model": "tiny_torchrec_int8",
        "throughput": float(batch.batch_size) / max(stats["median_ms"] / 1e3, 1e-9),
        "memory_mb": _estimate_state_dict_memory_mb(quantized_model),
    })
    performance_recorder("quantized_inference_latency", stats)
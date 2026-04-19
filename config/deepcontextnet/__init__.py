from __future__ import annotations

from pathlib import Path

from taac2026.domain.config import DataConfig, ModelConfig, TrainConfig
from taac2026.domain.experiment import ExperimentSpec
from taac2026.domain.features import build_default_feature_schema

from .model import build_model_component


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "datasets--TAAC2026--data_sample_1000"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "config" / "deepcontextnet"


EXPERIMENT = ExperimentSpec(
    name="deepcontextnet",
    data=DataConfig(
        dataset_path=str(DEFAULT_DATASET),
        max_seq_len=32,
        max_feature_tokens=16,
        max_event_features=4,
        stream_batch_rows=256,
        val_ratio=0.2,
        label_action_type=2,
        dense_feature_dim=16,
    ),
    model=ModelConfig(
        name="deepcontextnet",
        vocab_size=131072,
        embedding_dim=128,
        hidden_dim=128,
        dropout=0.1,
        num_layers=4,
        num_heads=8,
        recent_seq_len=32,
        memory_slots=0,
        ffn_multiplier=4.0,
        feature_cross_layers=0,
        sequence_layers=0,
        static_layers=0,
        query_decoder_layers=0,
        fusion_layers=0,
        num_queries=1,
        head_hidden_dim=128,
        segment_count=0,
        attention_dropout=0.1,
    ),
    train=TrainConfig(
        seed=42,
        epochs=10,
        batch_size=32,
        eval_batch_size=32,
        num_workers=0,
        output_dir=str(DEFAULT_OUTPUT_DIR),
        learning_rate=2.0e-4,
        weight_decay=1.0e-5,
        pairwise_weight=0.0,
        latency_warmup_steps=2,
        latency_measure_steps=8,
    ),
    build_data_pipeline=None,
    build_model_component=build_model_component,
    build_loss_stack=None,
    build_optimizer_component=None,
    switches={"logging": True, "visualization": True},
)

EXPERIMENT.feature_schema = build_default_feature_schema(EXPERIMENT.data, EXPERIMENT.model)
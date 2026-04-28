"""Symbiosis PCVR experiment package."""

from __future__ import annotations

from pathlib import Path

from taac2026.infrastructure.pcvr.config import (
    PCVRDataConfig,
    PCVRModelConfig,
    PCVRNSConfig,
    PCVROptimizerConfig,
    PCVRSymbiosisConfig,
    PCVRTrainConfig,
)
from taac2026.infrastructure.pcvr.experiment import PCVRExperiment
from taac2026.infrastructure.training.runtime import BinaryClassificationLossConfig, RuntimeExecutionConfig


EXPERIMENT = PCVRExperiment(
    name="pcvr_symbiosis",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVRSymbiosis",
    train_defaults=PCVRTrainConfig(
        data=PCVRDataConfig(batch_size=128, num_workers=8),
        optimizer=PCVROptimizerConfig(dense_optimizer_type="orthogonal_adamw"),
        runtime=RuntimeExecutionConfig(amp=True, amp_dtype="bfloat16", compile=True),
        loss=BinaryClassificationLossConfig(pairwise_auc_weight=0.05, pairwise_auc_temperature=1.0),
        model=PCVRModelConfig(
            num_blocks=3,
            num_heads=4,
            use_rope=True,
            rope_base=1_000_000.0,
            hidden_mult=4,
            dropout_rate=0.02,
            emb_skip_threshold=1_000_000,
        ),
        ns=PCVRNSConfig(
            tokenizer_type="rankmixer",
            user_tokens=5,
            item_tokens=2,
            groups_json="ns_groups.json",
        ),
        symbiosis=PCVRSymbiosisConfig(
            use_user_item_graph=True,
            use_fourier_time=True,
            use_context_exchange=True,
            use_multi_scale=True,
            use_domain_gate=True,
            use_candidate_decoder=True,
            use_action_conditioning=True,
            use_compressed_memory=True,
            use_attention_sink=True,
            use_lane_mixing=True,
            use_semantic_id=True,
            memory_block_size=16,
            memory_top_k=8,
            recent_tokens=64,
        ),
    ),
)

__all__ = ["EXPERIMENT"]
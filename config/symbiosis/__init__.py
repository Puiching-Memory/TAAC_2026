"""Symbiosis PCVR experiment package."""

from __future__ import annotations

from pathlib import Path

from taac2026.infrastructure.pcvr.config import PCVRDataConfig, PCVRModelConfig, PCVRNSConfig, PCVRTrainConfig
from taac2026.infrastructure.pcvr.experiment import PCVRExperiment
from taac2026.infrastructure.training.runtime import RuntimeExecutionConfig


EXPERIMENT = PCVRExperiment(
    name="pcvr_symbiosis",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVRSymbiosis",
    train_defaults=PCVRTrainConfig(
        data=PCVRDataConfig(batch_size=128, num_workers=8),
        runtime=RuntimeExecutionConfig(amp=True, amp_dtype="bfloat16", compile=True),
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
    ),
)

__all__ = ["EXPERIMENT"]
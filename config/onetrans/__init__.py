"""OneTrans-inspired PCVR experiment package."""

from __future__ import annotations

from pathlib import Path

from taac2026.infrastructure.pcvr.config import PCVRDataConfig, PCVRModelConfig, PCVRNSConfig, PCVRTrainConfig
from taac2026.infrastructure.pcvr.experiment import PCVRExperiment


EXPERIMENT = PCVRExperiment(
    name="pcvr_onetrans",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVROneTrans",
    train_defaults=PCVRTrainConfig(
        data=PCVRDataConfig(num_workers=8),
        model=PCVRModelConfig(
            num_blocks=2,
            num_heads=4,
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
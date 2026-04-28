"""PCVR HyFormer experiment package."""

from __future__ import annotations

from pathlib import Path

from taac2026.infrastructure.pcvr.config import PCVRDataConfig, PCVRModelConfig, PCVRNSConfig, PCVRTrainConfig
from taac2026.infrastructure.pcvr.experiment import PCVRExperiment


EXPERIMENT = PCVRExperiment(
    name="pcvr_hyformer",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVRHyFormer",
    train_defaults=PCVRTrainConfig(
        data=PCVRDataConfig(num_workers=8),
        model=PCVRModelConfig(num_queries=2, emb_skip_threshold=1_000_000),
        ns=PCVRNSConfig(
            tokenizer_type="rankmixer",
            user_tokens=5,
            item_tokens=2,
            groups_json="ns_groups.json",
        ),
    ),
)

__all__ = ["EXPERIMENT"]

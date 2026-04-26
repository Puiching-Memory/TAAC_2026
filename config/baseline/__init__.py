"""PCVR HyFormer experiment package."""

from __future__ import annotations

from pathlib import Path

from taac2026.infrastructure.pcvr.experiment import PCVRExperiment


EXPERIMENT = PCVRExperiment(
    name="pcvr_hyformer",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVRHyFormer",
    default_train_args=(
        "--ns_tokenizer_type",
        "rankmixer",
        "--user_ns_tokens",
        "5",
        "--item_ns_tokens",
        "2",
        "--num_queries",
        "2",
        "--ns_groups_json",
        "ns_groups.json",
        "--emb_skip_threshold",
        "1000000",
        "--num_workers",
        "8",
    ),
)

__all__ = ["EXPERIMENT"]

"""DeepContextNet PCVR experiment package."""

from __future__ import annotations

from pathlib import Path

from taac2026.infrastructure.pcvr.experiment import PCVRExperiment


EXPERIMENT = PCVRExperiment(
    name="pcvr_deepcontextnet",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVRDeepContextNet",
    default_train_args=(
        "--ns_tokenizer_type",
        "group",
        "--ns_groups_json",
        "ns_groups.json",
        "--num_blocks",
        "3",
        "--num_heads",
        "4",
        "--hidden_mult",
        "4",
        "--dropout_rate",
        "0.02",
        "--emb_skip_threshold",
        "1000000",
        "--num_workers",
        "8",
    ),
)

__all__ = ["EXPERIMENT"]
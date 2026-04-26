"""HyFormer PCVR experiment package."""

from __future__ import annotations

from pathlib import Path

from taac2026.infrastructure.pcvr.experiment import PCVRExperiment


EXPERIMENT = PCVRExperiment(
    name="pcvr_hyformer_paper",
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
        "--num_blocks",
        "2",
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
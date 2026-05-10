"""Shared PCVR data, model-input, and model construction helpers."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
from pathlib import Path
from typing import Any, NamedTuple

import torch


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict[str, torch.Tensor]
    seq_lens: dict[str, torch.Tensor]
    seq_time_buckets: dict[str, torch.Tensor]


def parse_seq_max_lens(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    if not value:
        return result
    for pair in value.split(","):
        if not pair.strip():
            continue
        name, raw_length = pair.split(":", 1)
        result[name.strip()] = int(raw_length.strip())
    return result


def build_feature_specs(schema: Any, per_position_vocab_sizes: list[int]) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    for _feature_id, offset, length in schema.entries:
        vocab_size = max(per_position_vocab_sizes[offset : offset + length])
        specs.append((vocab_size, offset, length))
    return specs


def resolve_schema_path(dataset_path: Path, schema_path: Path | None, checkpoint_dir: Path) -> Path:
    candidates: list[Path] = []
    if schema_path is not None:
        candidates.append(schema_path)
    candidates.append(checkpoint_dir / "schema.json")
    resolved_dataset_path = dataset_path.expanduser().resolve()
    if resolved_dataset_path.is_dir():
        candidates.append(resolved_dataset_path / "schema.json")
    else:
        candidates.append(resolved_dataset_path.parent / "schema.json")
    for candidate in candidates:
        expanded = candidate.expanduser().resolve()
        if expanded.exists():
            return expanded
    raise FileNotFoundError("schema.json not found from CLI, checkpoint sidecar, or dataset directory")


def load_ns_groups(dataset: Any, config: dict[str, Any], package_dir: Path, checkpoint_dir: Path) -> tuple[list[list[int]], list[list[int]]]:
    del package_dir, checkpoint_dir
    if config["ns_grouping_strategy"] == "singleton":
        return (
            [[index] for index in range(len(dataset.user_int_schema.entries))],
            [[index] for index in range(len(dataset.item_int_schema.entries))],
        )
    ns_groups_config = {
        "user_ns_groups": config["user_ns_groups"],
        "item_ns_groups": config["item_ns_groups"],
    }
    user_feature_to_index = {
        feature_id: index for index, (feature_id, _offset, _length) in enumerate(dataset.user_int_schema.entries)
    }
    item_feature_to_index = {
        feature_id: index for index, (feature_id, _offset, _length) in enumerate(dataset.item_int_schema.entries)
    }
    user_groups = [
        [user_feature_to_index[feature_id] for feature_id in feature_ids]
        for feature_ids in ns_groups_config["user_ns_groups"].values()
    ]
    item_groups = [
        [item_feature_to_index[feature_id] for feature_id in feature_ids]
        for feature_ids in ns_groups_config["item_ns_groups"].values()
    ]
    return user_groups, item_groups


def num_time_buckets(config: dict[str, Any], data_module: Any) -> int:
    if not bool(config["use_time_buckets"]):
        return 0
    return int(data_module.NUM_TIME_BUCKETS)


def build_pcvr_model(
    *,
    model_module: Any,
    model_class_name: str,
    data_module: Any,
    dataset: Any,
    config: dict[str, Any],
    package_dir: Path,
    checkpoint_dir: Path,
    extra_model_kwargs: Mapping[str, Any] | None = None,
) -> torch.nn.Module:
    user_ns_groups, item_ns_groups = load_ns_groups(dataset, config, package_dir, checkpoint_dir)
    user_int_feature_specs = build_feature_specs(dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(dataset.item_int_schema, dataset.item_int_vocab_sizes)
    configure_flash_attention_runtime = getattr(model_module, "configure_flash_attention_runtime", None)
    if callable(configure_flash_attention_runtime):
        configure_flash_attention_runtime(
            flash_attention_backend=str(config.get("flash_attention_backend", "torch")),
        )
    configure_rms_norm_runtime = getattr(model_module, "configure_rms_norm_runtime", None)
    if callable(configure_rms_norm_runtime):
        rms_norm_block_rows = int(config.get("rms_norm_block_rows", 1))
        if rms_norm_block_rows < 1:
            raise ValueError("rms_norm_block_rows must be positive")
        configure_rms_norm_runtime(
            rms_norm_backend=str(config.get("rms_norm_backend", "torch")),
            rms_norm_block_rows=rms_norm_block_rows,
        )
    model_class = getattr(model_module, model_class_name)
    model_kwargs = dict(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        d_model=int(config["d_model"]),
        emb_dim=int(config["emb_dim"]),
        num_queries=int(config["num_queries"]),
        num_blocks=int(config["num_blocks"]),
        num_heads=int(config["num_heads"]),
        seq_encoder_type=str(config["seq_encoder_type"]),
        hidden_mult=int(config["hidden_mult"]),
        dropout_rate=float(config["dropout_rate"]),
        seq_top_k=int(config["seq_top_k"]),
        seq_causal=bool(config["seq_causal"]),
        action_num=int(config["action_num"]),
        num_time_buckets=num_time_buckets(config, data_module),
        rank_mixer_mode=str(config["rank_mixer_mode"]),
        use_rope=bool(config["use_rope"]),
        rope_base=float(config["rope_base"]),
        emb_skip_threshold=int(config["emb_skip_threshold"]),
        seq_id_threshold=int(config["seq_id_threshold"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        ns_tokenizer_type=str(config["ns_tokenizer_type"]),
        user_ns_tokens=int(config["user_ns_tokens"]),
        item_ns_tokens=int(config["item_ns_tokens"]),
    )
    if extra_model_kwargs:
        model_signature = inspect.signature(model_class)
        for key, value in extra_model_kwargs.items():
            if key in model_signature.parameters:
                model_kwargs[key] = value
    return model_class(**model_kwargs)


def batch_to_model_input(batch: dict[str, Any], model_input_type: Any, device: torch.device) -> Any:
    device_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            device_batch[key] = value.to(device, non_blocking=True)
        else:
            device_batch[key] = value
    sequence_domains = device_batch["_seq_domains"]
    sequence_data: dict[str, torch.Tensor] = {}
    sequence_lengths: dict[str, torch.Tensor] = {}
    sequence_time_buckets: dict[str, torch.Tensor] = {}
    for domain in sequence_domains:
        sequence_data[domain] = device_batch[domain]
        sequence_lengths[domain] = device_batch[f"{domain}_len"]
        batch_size = device_batch[domain].shape[0]
        max_length = device_batch[domain].shape[2]
        sequence_time_buckets[domain] = device_batch.get(
            f"{domain}_time_bucket",
            torch.zeros(batch_size, max_length, dtype=torch.long, device=device),
        )
    return model_input_type(
        user_int_feats=device_batch["user_int_feats"],
        item_int_feats=device_batch["item_int_feats"],
        user_dense_feats=device_batch["user_dense_feats"],
        item_dense_feats=device_batch["item_dense_feats"],
        seq_data=sequence_data,
        seq_lens=sequence_lengths,
        seq_time_buckets=sequence_time_buckets,
    )


__all__ = [
    "ModelInput",
    "batch_to_model_input",
    "build_feature_specs",
    "build_pcvr_model",
    "load_ns_groups",
    "num_time_buckets",
    "parse_seq_max_lens",
    "resolve_schema_path",
]
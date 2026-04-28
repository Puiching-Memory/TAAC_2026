"""Shared PCVR data, model-input, and model construction helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch

from taac2026.infrastructure.io.files import read_json


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


def resolve_ns_groups_path(value: str, package_dir: Path, checkpoint_dir: Path) -> Path | None:
    if not value:
        return None
    candidates: list[Path] = []
    raw_path = Path(value)
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([checkpoint_dir / raw_path, package_dir / raw_path, Path.cwd() / raw_path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"NS groups JSON not found: {value}")


def load_ns_groups(dataset: Any, config: dict[str, Any], package_dir: Path, checkpoint_dir: Path) -> tuple[list[list[int]], list[list[int]]]:
    ns_groups_path = resolve_ns_groups_path(str(config["ns_groups_json"]), package_dir, checkpoint_dir)
    if ns_groups_path is None:
        return (
            [[index] for index in range(len(dataset.user_int_schema.entries))],
            [[index] for index in range(len(dataset.item_int_schema.entries))],
        )
    ns_groups_config = read_json(ns_groups_path)
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
) -> torch.nn.Module:
    user_ns_groups, item_ns_groups = load_ns_groups(dataset, config, package_dir, checkpoint_dir)
    user_int_feature_specs = build_feature_specs(dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(dataset.item_int_schema, dataset.item_int_vocab_sizes)
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
        ns_tokenizer_type=str(config["ns_tokenizer_type"]),
        user_ns_tokens=int(config["user_ns_tokens"]),
        item_ns_tokens=int(config["item_ns_tokens"]),
    )
    model_signature = inspect.signature(model_class)
    for key in (
        "symbiosis_use_user_item_graph",
        "symbiosis_use_fourier_time",
        "symbiosis_use_context_exchange",
        "symbiosis_use_multi_scale",
        "symbiosis_use_domain_gate",
        "symbiosis_use_candidate_decoder",
        "symbiosis_use_action_conditioning",
        "symbiosis_use_compressed_memory",
        "symbiosis_use_attention_sink",
        "symbiosis_use_lane_mixing",
        "symbiosis_use_semantic_id",
    ):
        if key in model_signature.parameters:
            model_kwargs[key] = bool(config[key])
    for key in (
        "symbiosis_memory_block_size",
        "symbiosis_memory_top_k",
        "symbiosis_recent_tokens",
    ):
        if key in model_signature.parameters:
            model_kwargs[key] = int(config[key])
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
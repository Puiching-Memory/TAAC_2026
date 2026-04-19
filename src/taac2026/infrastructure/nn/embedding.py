from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable

from torch import nn

from ...domain.features import FeatureSchema

if TYPE_CHECKING:
    from torch import Tensor, device
    from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
    from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


@lru_cache(maxsize=1)
def _torchrec_embedding_types():
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(max(previous_level, logging.ERROR))
    try:
        from torchrec import EmbeddingBagCollection
        from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
    finally:
        root_logger.setLevel(previous_level)
    return EmbeddingBagCollection, EmbeddingBagConfig, PoolingType


@lru_cache(maxsize=1)
def _torchrec_lookup_types():
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(max(previous_level, logging.ERROR))
    try:
        from torchrec import EmbeddingCollection
        from torchrec.modules.embedding_configs import EmbeddingConfig
    finally:
        root_logger.setLevel(previous_level)
    return EmbeddingCollection, EmbeddingConfig


def _selected_tables(
    feature_schema: FeatureSchema,
    table_names: Iterable[str] | None = None,
) -> tuple[object, ...]:
    selected_table_names = None if table_names is None else frozenset(table_names)
    selected = []
    for table in feature_schema.tables:
        if selected_table_names is not None and table.name not in selected_table_names:
            continue
        selected.append(table)
    if not selected:
        raise ValueError("No embedding configs were selected from the feature schema")
    return tuple(selected)


def _resolve_pooling_type(pooling_name: str):
    _, _, pooling_type_enum = _torchrec_embedding_types()
    normalized = pooling_name.strip().lower()
    if normalized == "mean":
        return pooling_type_enum.MEAN
    if normalized == "sum":
        return pooling_type_enum.SUM
    raise ValueError(f"Unsupported TorchRec pooling type: {pooling_name}")


def build_embedding_bag_configs(
    feature_schema: FeatureSchema,
    table_names: Iterable[str] | None = None,
) -> list[EmbeddingBagConfig]:
    _, embedding_bag_config, _ = _torchrec_embedding_types()
    configs: list[EmbeddingBagConfig] = []
    for table in _selected_tables(feature_schema, table_names):
        configs.append(
            embedding_bag_config(
                name=table.name,
                num_embeddings=table.num_embeddings,
                embedding_dim=table.embedding_dim,
                feature_names=[table.name],
                pooling=_resolve_pooling_type(table.pooling_type),
            )
        )
    return configs


def build_embedding_configs(
    feature_schema: FeatureSchema,
    table_names: Iterable[str] | None = None,
) -> list[EmbeddingConfig]:
    _, embedding_config = _torchrec_lookup_types()
    configs: list[EmbeddingConfig] = []
    for table in _selected_tables(feature_schema, table_names):
        configs.append(
            embedding_config(
                name=table.name,
                num_embeddings=table.num_embeddings,
                embedding_dim=table.embedding_dim,
                feature_names=[table.name],
            )
        )
    return configs


class TorchRecEmbeddingBagAdapter(nn.Module):
    """Thin wrapper around TorchRec EmbeddingBagCollection for schema-driven pooling."""

    def __init__(
        self,
        feature_schema: FeatureSchema,
        table_names: Iterable[str] | None = None,
        device: device | None = None,
    ) -> None:
        super().__init__()
        embedding_bag_collection, _, _ = _torchrec_embedding_types()
        self.configs = tuple(build_embedding_bag_configs(feature_schema, table_names=table_names))
        self.table_names = tuple(config.name for config in self.configs)
        self.output_dim = sum(int(config.embedding_dim) for config in self.configs)
        self.collection = embedding_bag_collection(tables=list(self.configs), device=device)

    def forward_keyed(self, features: KeyedJaggedTensor) -> KeyedTensor:
        return self.collection(features)

    def forward_dict(self, features: KeyedJaggedTensor) -> dict[str, Tensor]:
        return self.forward_keyed(features).to_dict()

    def forward(self, features: KeyedJaggedTensor) -> Tensor:
        return self.forward_keyed(features).values()


class TorchRecEmbeddingCollectionAdapter(nn.Module):
    """Schema-driven TorchRec EmbeddingCollection for per-token jagged lookups."""

    def __init__(
        self,
        feature_schema: FeatureSchema,
        table_names: Iterable[str] | None = None,
        device: device | None = None,
    ) -> None:
        super().__init__()
        embedding_collection, _ = _torchrec_lookup_types()
        self.configs = tuple(build_embedding_configs(feature_schema, table_names=table_names))
        self.table_names = tuple(config.name for config in self.configs)
        self.output_dim = sum(int(config.embedding_dim) for config in self.configs)
        self.collection = embedding_collection(tables=list(self.configs), device=device)

    def forward(self, features: KeyedJaggedTensor) -> dict[str, JaggedTensor]:
        return self.collection(features)

    def forward_dict(self, features: KeyedJaggedTensor) -> dict[str, JaggedTensor]:
        return self.forward(features)


__all__ = [
    "TorchRecEmbeddingCollectionAdapter",
    "TorchRecEmbeddingBagAdapter",
    "build_embedding_configs",
    "build_embedding_bag_configs",
]
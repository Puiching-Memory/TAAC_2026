from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable

from torch import nn

from ...domain.features import FeatureSchema

if TYPE_CHECKING:
    from torch import Tensor, device
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


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
    selected_table_names = None if table_names is None else frozenset(table_names)
    configs: list[EmbeddingBagConfig] = []
    for table in feature_schema.tables:
        if selected_table_names is not None and table.name not in selected_table_names:
            continue
        configs.append(
            embedding_bag_config(
                name=table.name,
                num_embeddings=table.num_embeddings,
                embedding_dim=table.embedding_dim,
                feature_names=[table.name],
                pooling=_resolve_pooling_type(table.pooling_type),
            )
        )
    if not configs:
        raise ValueError("No embedding bag configs were selected from the feature schema")
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


__all__ = [
    "TorchRecEmbeddingBagAdapter",
    "build_embedding_bag_configs",
]
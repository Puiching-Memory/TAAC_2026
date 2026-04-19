from __future__ import annotations

from dataclasses import dataclass

from .config import DataConfig, ModelConfig


@dataclass(frozen=True, slots=True)
class FeatureTableSpec:
    name: str
    family: str
    num_embeddings: int
    embedding_dim: int
    pooling_type: str = "mean"
    is_sequence: bool = False
    max_length: int | None = None


@dataclass(slots=True)
class FeatureSchema:
    tables: tuple[FeatureTableSpec, ...]
    dense_dim: int
    sequence_names: tuple[str, ...] = ()
    variant: str = "legacy_batch_v1"
    auto_sync: bool = False

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(table.name for table in self.tables)

    def table(self, name: str) -> FeatureTableSpec:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(f"Feature schema has no table named '{name}'")


def build_default_feature_schema(data_config: DataConfig, model_config: ModelConfig) -> FeatureSchema:
    vocab_size = max(int(model_config.vocab_size), 2)
    embedding_dim = int(model_config.embedding_dim)
    sequence_names = tuple(data_config.sequence_names)
    sequence_table_specs = tuple(
        FeatureTableSpec(
            name=f"sequence:{sequence_name}",
            family="sequence",
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            pooling_type="mean",
            is_sequence=True,
            max_length=int(data_config.max_seq_len),
        )
        for sequence_name in sequence_names
    )
    tables = (
        FeatureTableSpec(
            name="user_tokens",
            family="user",
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            pooling_type="mean",
            max_length=int(data_config.max_feature_tokens),
        ),
        FeatureTableSpec(
            name="candidate_tokens",
            family="candidate",
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            pooling_type="mean",
            max_length=int(data_config.max_feature_tokens),
        ),
        FeatureTableSpec(
            name="context_tokens",
            family="context",
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            pooling_type="mean",
            max_length=int(data_config.max_feature_tokens),
        ),
        FeatureTableSpec(
            name="history_tokens",
            family="history",
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            pooling_type="mean",
            is_sequence=True,
            max_length=int(data_config.max_seq_len),
        ),
        FeatureTableSpec(
            name="candidate_post_tokens",
            family="candidate_post",
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            pooling_type="mean",
            max_length=int(data_config.max_feature_tokens),
        ),
        FeatureTableSpec(
            name="candidate_author_tokens",
            family="candidate_author",
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            pooling_type="mean",
            max_length=int(data_config.max_feature_tokens),
        ),
        FeatureTableSpec(
            name="history_post_tokens",
            family="history_post",
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            pooling_type="mean",
            is_sequence=True,
            max_length=int(data_config.max_seq_len),
        ),
        FeatureTableSpec(
            name="history_author_tokens",
            family="history_author",
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            pooling_type="mean",
            is_sequence=True,
            max_length=int(data_config.max_seq_len),
        ),
        FeatureTableSpec(
            name="history_action_tokens",
            family="history_action",
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            pooling_type="mean",
            is_sequence=True,
            max_length=int(data_config.max_seq_len),
        ),
        FeatureTableSpec(
            name="history_time_gap",
            family="history_time_gap",
            num_embeddings=64,
            embedding_dim=embedding_dim,
            pooling_type="mean",
            is_sequence=True,
            max_length=int(data_config.max_seq_len),
        ),
        FeatureTableSpec(
            name="history_group_ids",
            family="history_group",
            num_embeddings=max(2, len(sequence_names) + 1),
            embedding_dim=embedding_dim,
            pooling_type="mean",
            is_sequence=True,
            max_length=int(data_config.max_seq_len),
        ),
        *sequence_table_specs,
    )
    return FeatureSchema(
        tables=tables,
        dense_dim=int(data_config.dense_feature_dim),
        sequence_names=sequence_names,
        variant="legacy_batch_v1",
        auto_sync=True,
    )


def sync_feature_schema(
    feature_schema: FeatureSchema | None,
    data_config: DataConfig,
    model_config: ModelConfig,
) -> FeatureSchema:
    if feature_schema is None or feature_schema.auto_sync:
        return build_default_feature_schema(data_config, model_config)
    return feature_schema


__all__ = [
    "FeatureSchema",
    "FeatureTableSpec",
    "build_default_feature_schema",
    "sync_feature_schema",
]
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taac2026.domain.schema import FeatureSchema
from taac2026.infrastructure.io.json import read_path


@dataclass(frozen=True, slots=True)
class PCVRSequenceLayout:
    domain: str
    prefix: str
    timestamp_fid: int | None
    feature_ids: tuple[int, ...]
    sideinfo_fids: tuple[int, ...]
    vocab_sizes: dict[int, int]
    max_len: int

    @property
    def sideinfo_vocab_sizes(self) -> list[int]:
        return [self.vocab_sizes[fid] for fid in self.sideinfo_fids]


@dataclass(frozen=True, slots=True)
class PCVRSchemaLayout:
    schema_path: Path
    raw_payload: dict[str, Any]
    user_int_cols: tuple[tuple[int, int, int], ...]
    item_int_cols: tuple[tuple[int, int, int], ...]
    user_dense_cols: tuple[tuple[int, int], ...]
    user_int_schema: FeatureSchema
    item_int_schema: FeatureSchema
    user_dense_schema: FeatureSchema
    item_dense_schema: FeatureSchema
    user_int_vocab_sizes: list[int]
    item_int_vocab_sizes: list[int]
    sequences: dict[str, PCVRSequenceLayout]

    @property
    def seq_domains(self) -> list[str]:
        return sorted(self.sequences)

    @property
    def seq_feature_ids(self) -> dict[str, list[int]]:
        return {domain: list(layout.feature_ids) for domain, layout in self.sequences.items()}

    @property
    def seq_vocab_sizes(self) -> dict[str, dict[int, int]]:
        return {domain: dict(layout.vocab_sizes) for domain, layout in self.sequences.items()}

    @property
    def seq_domain_vocab_sizes(self) -> dict[str, list[int]]:
        return {
            domain: layout.sideinfo_vocab_sizes
            for domain, layout in self.sequences.items()
        }

    @property
    def ts_fids(self) -> dict[str, int | None]:
        return {domain: layout.timestamp_fid for domain, layout in self.sequences.items()}

    @property
    def sideinfo_fids(self) -> dict[str, list[int]]:
        return {domain: list(layout.sideinfo_fids) for domain, layout in self.sequences.items()}

    @property
    def seq_prefix(self) -> dict[str, str]:
        return {domain: layout.prefix for domain, layout in self.sequences.items()}

    @property
    def seq_maxlen(self) -> dict[str, int]:
        return {domain: layout.max_len for domain, layout in self.sequences.items()}

    def required_column_names(self, parquet_schema_names: list[str]) -> tuple[str, ...]:
        available = set(parquet_schema_names)
        names: list[str] = ["timestamp", "label_type", "user_id"]
        names.extend(f"user_int_feats_{fid}" for fid, _vocab_size, _dim in self.user_int_cols)
        names.extend(f"item_int_feats_{fid}" for fid, _vocab_size, _dim in self.item_int_cols)
        names.extend(f"user_dense_feats_{fid}" for fid, _dim in self.user_dense_cols)
        for layout in self.sequences.values():
            names.extend(f"{layout.prefix}_{fid}" for fid in layout.feature_ids)
        return tuple(dict.fromkeys(name for name in names if name in available))


def _feature_schema_from_int_columns(
    columns: tuple[tuple[int, int, int], ...]
) -> tuple[FeatureSchema, list[int]]:
    schema = FeatureSchema()
    vocab_sizes: list[int] = []
    for fid, vocab_size, dim in columns:
        schema.add(fid, dim)
        vocab_sizes.extend([vocab_size] * dim)
    return schema, vocab_sizes


def _feature_schema_from_dense_columns(
    columns: tuple[tuple[int, int], ...]
) -> FeatureSchema:
    schema = FeatureSchema()
    for fid, dim in columns:
        schema.add(fid, dim)
    return schema


def _tuple_columns(raw_columns: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in column) for column in raw_columns)


def load_pcvr_schema_layout(
    schema_path: str | Path,
    seq_max_lens: dict[str, int] | None = None,
) -> PCVRSchemaLayout:
    resolved_path = Path(schema_path).expanduser().resolve()
    raw = read_path(resolved_path)
    max_lens = seq_max_lens or {}

    user_int_cols = _tuple_columns(raw["user_int"])
    item_int_cols = _tuple_columns(raw["item_int"])
    user_dense_cols = _tuple_columns(raw["user_dense"])

    user_int_schema, user_int_vocab_sizes = _feature_schema_from_int_columns(user_int_cols)
    item_int_schema, item_int_vocab_sizes = _feature_schema_from_int_columns(item_int_cols)
    user_dense_schema = _feature_schema_from_dense_columns(user_dense_cols)

    sequences: dict[str, PCVRSequenceLayout] = {}
    for domain in sorted(raw["seq"]):
        config = raw["seq"][domain]
        timestamp_fid = config["ts_fid"]
        features = _tuple_columns(config["features"])
        feature_ids = tuple(fid for fid, _vocab_size in features)
        vocab_sizes = {fid: vocab_size for fid, vocab_size in features}
        sideinfo_fids = tuple(fid for fid in feature_ids if fid != timestamp_fid)
        sequences[domain] = PCVRSequenceLayout(
            domain=domain,
            prefix=config["prefix"],
            timestamp_fid=timestamp_fid,
            feature_ids=feature_ids,
            sideinfo_fids=sideinfo_fids,
            vocab_sizes=vocab_sizes,
            max_len=int(max_lens.get(domain, 256)),
        )

    return PCVRSchemaLayout(
        schema_path=resolved_path,
        raw_payload=raw,
        user_int_cols=user_int_cols,
        item_int_cols=item_int_cols,
        user_dense_cols=user_dense_cols,
        user_int_schema=user_int_schema,
        item_int_schema=item_int_schema,
        user_dense_schema=user_dense_schema,
        item_dense_schema=FeatureSchema(),
        user_int_vocab_sizes=user_int_vocab_sizes,
        item_int_vocab_sizes=item_int_vocab_sizes,
        sequences=sequences,
    )

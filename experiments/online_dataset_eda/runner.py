from __future__ import annotations

import hashlib
import heapq
import math
import os
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from taac2026.infrastructure.io.json import dumps, loads, read_path
from taac2026.infrastructure.io.streams import write_stdout_line
from taac2026.infrastructure.logging import configure_logging, logger

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit("pyarrow is required for online dataset EDA") from exc


DEFAULT_BATCH_ROWS = 8192
DEFAULT_CARDINALITY_SKETCH_K = 2048
DEFAULT_TOKEN_OVERLAP_SKETCH_K = 256
DEFAULT_TOKEN_SAMPLE_LIMIT_PER_BATCH = 20000
DEFAULT_USER_SAMPLE_LIMIT = 20000
DEFAULT_SEQUENCE_SAMPLE_SIZE = 8192
DEFAULT_REPEAT_SAMPLE_ROWS_PER_DOMAIN = 20000
DEFAULT_PROGRESS_STEP_PERCENT = 10.0
DEFAULT_TOP_K = 20
PROFILE_FILENAME = "online_dataset_eda_profile.json"
DRIFT_FILENAME = "online_dataset_eda_drift.json"
UINT64_MASK = (1 << 64) - 1


@dataclass(slots=True)
class OnlineDatasetEDAConfig:
    dataset_path: Path | None = None
    schema_path: Path | None = None
    output_dir: Path | None = None
    reference_profile_path: Path | None = None
    reference_profile_json: str | None = None
    dataset_role: str = "online"
    batch_rows: int = DEFAULT_BATCH_ROWS
    cardinality_sketch_k: int = DEFAULT_CARDINALITY_SKETCH_K
    token_overlap_sketch_k: int = DEFAULT_TOKEN_OVERLAP_SKETCH_K
    token_sample_limit_per_batch: int = DEFAULT_TOKEN_SAMPLE_LIMIT_PER_BATCH
    user_sample_limit: int = DEFAULT_USER_SAMPLE_LIMIT
    sequence_sample_size: int = DEFAULT_SEQUENCE_SAMPLE_SIZE
    repeat_sample_rows_per_domain: int = DEFAULT_REPEAT_SAMPLE_ROWS_PER_DOMAIN
    max_rows: int | None = None
    sample_percent: float | None = None
    progress_step_percent: float = DEFAULT_PROGRESS_STEP_PERCENT
    top_k: int = DEFAULT_TOP_K
    enable_label_lift: bool = False
    label_feature_top_k: int = DEFAULT_TOP_K
    label_feature_min_support: int = 20
    label_feature_sample_rows: int = 50000


def mix_uint64(value: int) -> int:
    value &= UINT64_MASK
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & UINT64_MASK
    value ^= value >> 31
    return value & UINT64_MASK


def stable_hash(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return mix_uint64(int(value))
    if isinstance(value, int):
        return mix_uint64(value)
    if isinstance(value, float):
        packed = struct.unpack(">Q", struct.pack(">d", value))[0]
        return mix_uint64(packed)
    if isinstance(value, tuple):
        digest = 0x9E3779B97F4A7C15
        for item in value:
            digest = mix_uint64(digest ^ stable_hash(item))
        return digest
    encoded = str(value).encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "big")


def _read_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _read_env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def _read_env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be one of 1/0/true/false/yes/no/on/off")


def apply_env_overrides(config: OnlineDatasetEDAConfig) -> OnlineDatasetEDAConfig:
    updates: dict[str, Any] = {}
    for env_name, field_name, reader in (
        ("ONLINE_EDA_BATCH_ROWS", "batch_rows", _read_env_int),
        ("ONLINE_EDA_CARDINALITY_SKETCH_K", "cardinality_sketch_k", _read_env_int),
        ("ONLINE_EDA_TOKEN_OVERLAP_SKETCH_K", "token_overlap_sketch_k", _read_env_int),
        ("ONLINE_EDA_TOKEN_SAMPLE_LIMIT_PER_BATCH", "token_sample_limit_per_batch", _read_env_int),
        ("ONLINE_EDA_USER_SAMPLE_LIMIT", "user_sample_limit", _read_env_int),
        ("ONLINE_EDA_SEQUENCE_SAMPLE_SIZE", "sequence_sample_size", _read_env_int),
        ("ONLINE_EDA_REPEAT_SAMPLE_ROWS_PER_DOMAIN", "repeat_sample_rows_per_domain", _read_env_int),
        ("ONLINE_EDA_MAX_ROWS", "max_rows", _read_env_int),
        ("ONLINE_EDA_SAMPLE_PERCENT", "sample_percent", _read_env_float),
        ("ONLINE_EDA_PROGRESS_STEP_PERCENT", "progress_step_percent", _read_env_float),
        ("ONLINE_EDA_TOP_K", "top_k", _read_env_int),
        ("ONLINE_EDA_LABEL_FEATURE_TOP_K", "label_feature_top_k", _read_env_int),
        ("ONLINE_EDA_LABEL_FEATURE_MIN_SUPPORT", "label_feature_min_support", _read_env_int),
        ("ONLINE_EDA_LABEL_FEATURE_SAMPLE_ROWS", "label_feature_sample_rows", _read_env_int),
    ):
        raw_value = reader(env_name)
        if raw_value is not None:
            updates[field_name] = raw_value
    enable_label_lift = _read_env_bool("ONLINE_EDA_ENABLE_LABEL_LIFT")
    if enable_label_lift is not None:
        updates["enable_label_lift"] = enable_label_lift
    analysis_level = os.environ.get("ONLINE_EDA_ANALYSIS_LEVEL", "").strip().lower()
    if analysis_level:
        if analysis_level not in {"fast", "full"}:
            raise SystemExit("ONLINE_EDA_ANALYSIS_LEVEL must be 'fast' or 'full'")
        if analysis_level == "full":
            updates.setdefault("enable_label_lift", True)
            updates.setdefault("token_sample_limit_per_batch", max(config.token_sample_limit_per_batch, 100000))
            updates.setdefault("repeat_sample_rows_per_domain", max(config.repeat_sample_rows_per_domain, 100000))
        else:
            updates.setdefault("enable_label_lift", False)
    reference_profile = os.environ.get("ONLINE_EDA_REFERENCE_PROFILE")
    if reference_profile:
        updates["reference_profile_path"] = Path(reference_profile)
    reference_profile_json = os.environ.get("ONLINE_EDA_REFERENCE_PROFILE_JSON")
    if reference_profile_json:
        updates["reference_profile_json"] = reference_profile_json
    return replace(config, **updates) if updates else config


@dataclass(slots=True)
class SequenceDomainLayout:
    name: str
    prefix: str
    ts_column: str | None
    sideinfo_columns: tuple[str, ...]

    @property
    def all_columns(self) -> tuple[str, ...]:
        columns = list(self.sideinfo_columns)
        if self.ts_column is not None:
            columns.append(self.ts_column)
        return tuple(columns)

    @property
    def length_column(self) -> str | None:
        if self.ts_column is not None:
            return self.ts_column
        if self.sideinfo_columns:
            return self.sideinfo_columns[0]
        return None

    @property
    def repeat_column(self) -> str | None:
        if self.sideinfo_columns:
            return self.sideinfo_columns[0]
        return self.ts_column


@dataclass(slots=True)
class SchemaLayout:
    user_int_columns: tuple[str, ...]
    item_int_columns: tuple[str, ...]
    user_dense_columns: tuple[str, ...]
    sequence_domains: tuple[SequenceDomainLayout, ...]

    @classmethod
    def from_path(cls, path: Path) -> SchemaLayout:
        raw = read_path(path)
        user_int_columns = tuple(f"user_int_feats_{fid}" for fid, _vocab_size, _dim in raw.get("user_int", ()))
        item_int_columns = tuple(f"item_int_feats_{fid}" for fid, _vocab_size, _dim in raw.get("item_int", ()))
        user_dense_columns = tuple(f"user_dense_feats_{fid}" for fid, _dim in raw.get("user_dense", ()))
        sequence_domains: list[SequenceDomainLayout] = []
        for name, domain_config in sorted(raw.get("seq", {}).items()):
            prefix = str(domain_config["prefix"])
            ts_fid = domain_config.get("ts_fid")
            ts_column = f"{prefix}_{ts_fid}" if ts_fid is not None else None
            sideinfo_columns = tuple(
                f"{prefix}_{fid}"
                for fid, _vocab_size in domain_config.get("features", ())
                if fid != ts_fid
            )
            sequence_domains.append(
                SequenceDomainLayout(
                    name=str(name),
                    prefix=prefix,
                    ts_column=ts_column,
                    sideinfo_columns=sideinfo_columns,
                )
            )
        return cls(
            user_int_columns=user_int_columns,
            item_int_columns=item_int_columns,
            user_dense_columns=user_dense_columns,
            sequence_domains=tuple(sequence_domains),
        )

    @property
    def sequence_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        for domain in self.sequence_domains:
            columns.extend(domain.all_columns)
        return tuple(columns)

    @property
    def sparse_columns(self) -> tuple[str, ...]:
        return self.user_int_columns + self.item_int_columns

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.user_int_columns + self.user_dense_columns + self.item_int_columns + self.sequence_columns

    @property
    def primary_user_id_column(self) -> str | None:
        if self.user_int_columns:
            return self.user_int_columns[0]
        return None


@dataclass(slots=True)
class DatasetInfo:
    dataset_path: Path
    files: tuple[Path, ...]
    available_columns: tuple[str, ...]
    total_rows: int
    file_count: int


class KMVSketch:
    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self._values: set[int] = set()
        self._heap: list[int] = []

    def add(self, value: Any) -> None:
        hashed = stable_hash(value)
        if hashed in self._values:
            return
        if len(self._values) < self.limit:
            self._values.add(hashed)
            heapq.heappush(self._heap, -hashed)
            return
        largest = -self._heap[0]
        if hashed >= largest:
            return
        removed = -heapq.heapreplace(self._heap, -hashed)
        self._values.remove(removed)
        self._values.add(hashed)

    def add_many(self, values: list[Any]) -> None:
        for value in values:
            self.add(value)

    def estimate(self) -> int:
        if len(self._values) < self.limit:
            return len(self._values)
        threshold = max(self._values) / float(UINT64_MASK)
        if threshold <= 0:
            return len(self._values)
        return round((self.limit - 1) / threshold)

    def fingerprint(self) -> list[str]:
        return [f"{value:016x}" for value in sorted(self._values)]


class ReservoirSampler:
    def __init__(self, limit: int, seed: int) -> None:
        self.limit = max(0, int(limit))
        self.seed = int(seed) & UINT64_MASK
        self.samples: list[float] = []
        self.seen = 0

    def add(self, value: float) -> None:
        if self.limit <= 0:
            return
        self.seen += 1
        if len(self.samples) < self.limit:
            self.samples.append(float(value))
            return
        hashed = stable_hash((self.seed, self.seen))
        index = hashed % self.seen
        if index < self.limit:
            self.samples[index] = float(value)

    def add_many(self, values: list[int | float]) -> None:
        for value in values:
            self.add(float(value))


@dataclass(slots=True)
class DenseStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    zero_count: int = 0

    def add_arrow(self, values: pa.Array) -> None:
        if len(values) == 0:
            return
        values = pc.cast(values, pa.float64(), safe=False)
        total = scalar_as_float(pc.sum(values))
        total_sq = scalar_as_float(pc.sum(pc.multiply(values, values)))
        zero_count = scalar_as_int(pc.sum(pc.cast(pc.equal(values, 0.0), pa.int64())))
        self.count += len(values)
        self.total += total
        self.total_sq += total_sq
        self.zero_count += zero_count


@dataclass(slots=True)
class SequenceStats:
    domain: str
    sampler: ReservoirSampler
    rows: int = 0
    total_length: int = 0
    empty_rows: int = 0
    min_length: int | None = None
    max_length: int = 0
    repeat_rate_sum: float = 0.0
    repeat_rows: int = 0

    def add_lengths(self, lengths: list[int]) -> None:
        if not lengths:
            return
        self.rows += len(lengths)
        self.total_length += sum(lengths)
        self.empty_rows += sum(1 for value in lengths if value == 0)
        batch_min = min(lengths)
        if self.min_length is None or batch_min < self.min_length:
            self.min_length = batch_min
        self.max_length = max(self.max_length, max(lengths))
        self.sampler.add_many(lengths)

    def add_repeat_rates(self, values: list[Any]) -> None:
        for value in values:
            tokens = normalize_list(value)
            if not tokens:
                continue
            self.repeat_rate_sum += 1.0 - len(set(tokens)) / float(len(tokens))
            self.repeat_rows += 1


@dataclass(slots=True)
class ScanResult:
    scanned_rows: int
    null_counts: dict[str, int]
    cardinality_sketches: dict[str, KMVSketch]
    overlap_sketches: dict[str, KMVSketch]
    dense_stats: dict[str, DenseStats]
    sequence_stats: dict[str, SequenceStats]
    label_counters: dict[str, Counter[str]]
    sampled_user_activity: Counter[Any]
    sampled_user_domains: dict[Any, set[str]]
    label_feature_stats: dict[str, Counter[tuple[Any, bool]]]


class ProgressTracker:
    def __init__(self, label: str, total_rows: int, *, step_percent: float) -> None:
        self.label = label
        self.total_rows = max(total_rows, 0)
        self._started = False
        self._last_reported_rows = -1
        self._step_percent = max(step_percent, 0.1)
        self._next_percent = self._step_percent

    def _emit(self, scanned_rows: int) -> None:
        if self.total_rows <= 0:
            percent = 100.0
        else:
            percent = min(100.0, scanned_rows * 100.0 / float(self.total_rows))
        logger.info(
            "[online-eda] progress {}: {}/{} ({:.1f}%)",
            self.label,
            scanned_rows,
            self.total_rows,
            percent,
        )
        self._last_reported_rows = scanned_rows

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._emit(0)

    def update(self, scanned_rows: int) -> None:
        if not self._started:
            self.start()
        if self.total_rows <= 0:
            return
        percent = scanned_rows * 100.0 / float(self.total_rows)
        if scanned_rows >= self.total_rows or percent + 1e-9 >= self._next_percent:
            self._emit(scanned_rows)
            while self._next_percent <= percent + 1e-9:
                self._next_percent += self._step_percent

    def finish(self, scanned_rows: int) -> None:
        if not self._started:
            self.start()
        if self._last_reported_rows != scanned_rows:
            self._emit(scanned_rows)


def scalar_as_int(value: pa.Scalar) -> int:
    raw = value.as_py()
    return 0 if raw is None else int(raw)


def scalar_as_float(value: pa.Scalar) -> float:
    raw = value.as_py()
    return 0.0 if raw is None else float(raw)


def is_list_type(data_type: pa.DataType) -> bool:
    return pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def resolve_schema_path(dataset_path: Path, raw_value: Path | None) -> Path:
    candidates: list[Path] = []
    if raw_value is not None:
        candidates.append(raw_value)
    env_schema = os.environ.get("TAAC_SCHEMA_PATH")
    if env_schema:
        candidates.append(Path(env_schema))
    if dataset_path.is_dir():
        candidates.append(dataset_path / "schema.json")
    else:
        candidates.append(dataset_path.parent / "schema.json")
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    raise SystemExit("schema.json not found; use --schema-path or place it beside the parquet data")


def resolve_reference_profile_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    if resolved.is_dir():
        for name in (PROFILE_FILENAME, "online_dataset_eda_profile_train.json", "dataset_profile_train.json"):
            candidate = resolved / name
            if candidate.exists():
                return candidate
    return None


def resolve_scan_row_limit(total_rows: int, max_rows: int | None, sample_percent: float | None) -> int | None:
    if sample_percent is not None:
        if total_rows <= 0:
            return 0
        return min(total_rows, max(1, math.ceil(total_rows * sample_percent / 100.0)))
    if max_rows is None:
        return None
    return min(total_rows, max_rows)


def validate_config(config: OnlineDatasetEDAConfig) -> None:
    if config.dataset_path is None:
        raise SystemExit("dataset path is required")
    if config.batch_rows <= 0:
        raise SystemExit("ONLINE_EDA_BATCH_ROWS must be positive")
    if config.cardinality_sketch_k <= 0:
        raise SystemExit("ONLINE_EDA_CARDINALITY_SKETCH_K must be positive")
    if config.token_overlap_sketch_k <= 0:
        raise SystemExit("ONLINE_EDA_TOKEN_OVERLAP_SKETCH_K must be positive")
    if config.token_sample_limit_per_batch < 0:
        raise SystemExit("ONLINE_EDA_TOKEN_SAMPLE_LIMIT_PER_BATCH must be non-negative")
    if config.user_sample_limit < 0:
        raise SystemExit("ONLINE_EDA_USER_SAMPLE_LIMIT must be non-negative")
    if config.sequence_sample_size < 0:
        raise SystemExit("ONLINE_EDA_SEQUENCE_SAMPLE_SIZE must be non-negative")
    if config.repeat_sample_rows_per_domain < 0:
        raise SystemExit("ONLINE_EDA_REPEAT_SAMPLE_ROWS_PER_DOMAIN must be non-negative")
    if config.max_rows is not None and config.max_rows <= 0:
        raise SystemExit("ONLINE_EDA_MAX_ROWS must be positive when set")
    if config.sample_percent is not None and not (0.0 < config.sample_percent <= 100.0):
        raise SystemExit("ONLINE_EDA_SAMPLE_PERCENT must be in (0, 100] when set")
    if config.max_rows is not None and config.sample_percent is not None:
        raise SystemExit("ONLINE_EDA_MAX_ROWS and ONLINE_EDA_SAMPLE_PERCENT are mutually exclusive")
    if config.top_k < 0:
        raise SystemExit("ONLINE_EDA_TOP_K must be non-negative")
    if config.label_feature_top_k < 0:
        raise SystemExit("ONLINE_EDA_LABEL_FEATURE_TOP_K must be non-negative")
    if config.label_feature_min_support < 0:
        raise SystemExit("ONLINE_EDA_LABEL_FEATURE_MIN_SUPPORT must be non-negative")
    if config.label_feature_sample_rows < 0:
        raise SystemExit("ONLINE_EDA_LABEL_FEATURE_SAMPLE_ROWS must be non-negative")


def list_parquet_files(dataset_path: Path) -> tuple[Path, ...]:
    if dataset_path.is_dir():
        files = tuple(sorted(dataset_path.rglob("*.parquet")))
    else:
        files = (dataset_path,)
    if not files:
        raise SystemExit(f"no .parquet files found at {dataset_path}")
    return files


def build_dataset_info(dataset_path: Path) -> DatasetInfo:
    files = list_parquet_files(dataset_path)
    total_rows = 0
    common_columns: set[str] | None = None
    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        total_rows += parquet_file.metadata.num_rows
        file_columns = set(parquet_file.schema_arrow.names)
        common_columns = file_columns if common_columns is None else common_columns & file_columns
    return DatasetInfo(
        dataset_path=dataset_path,
        files=files,
        available_columns=tuple(sorted(common_columns or ())),
        total_rows=total_rows,
        file_count=len(files),
    )


def iter_batches(dataset: DatasetInfo, *, columns: list[str], max_rows: int | None, batch_rows: int):
    remaining = max_rows
    iter_columns = [column for column in columns if column in dataset.available_columns]
    for file_path in dataset.files:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=batch_rows, columns=iter_columns, use_threads=True):
            if remaining is not None:
                if remaining <= 0:
                    return
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
                remaining -= batch.num_rows
            yield batch
            if remaining == 0:
                return


def batch_column(batch: pa.RecordBatch, column_name: str) -> pa.Array:
    return batch.column(batch.schema.get_field_index(column_name))


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if normalize_scalar(item) is not None and item != 0]
    scalar = normalize_scalar(value)
    if scalar is None or scalar == 0:
        return []
    return [scalar]


def hashable_value(value: Any) -> Any:
    if isinstance(value, list):
        tokens = tuple(normalize_list(value))
        return tokens or None
    scalar = normalize_scalar(value)
    if isinstance(scalar, (int, float)) and scalar <= 0:
        return None
    return scalar


def missing_count(array: pa.Array) -> int:
    if is_list_type(array.type):
        lengths = pc.list_value_length(array)
        flags = pc.or_(pc.is_null(lengths), pc.equal(pc.fill_null(lengths, 0), 0))
        return scalar_as_int(pc.sum(pc.cast(flags, pa.int64())))
    count = int(array.null_count)
    if pa.types.is_floating(array.type):
        non_null = pc.drop_null(array)
        if len(non_null):
            count += scalar_as_int(pc.sum(pc.cast(pc.is_nan(non_null), pa.int64())))
    return count


def list_lengths(array: pa.Array) -> list[int]:
    if is_list_type(array.type):
        lengths = pc.fill_null(pc.list_value_length(array), 0)
        return [int(value or 0) for value in lengths.to_pylist()]
    return [0 if value is None else 1 for value in array.to_pylist()]


def numeric_values(array: pa.Array) -> pa.Array:
    values = pc.list_flatten(array) if is_list_type(array.type) else array
    values = pc.drop_null(values)
    if len(values) == 0:
        return values
    if pa.types.is_floating(values.type):
        values = pc.filter(values, pc.invert(pc.is_nan(values)))
    return pc.cast(values, pa.float64(), safe=False)


def token_values(array: pa.Array) -> pa.Array:
    values = pc.list_flatten(array) if is_list_type(array.type) else array
    values = pc.drop_null(values)
    if len(values) == 0:
        return values
    if pa.types.is_floating(values.type):
        values = pc.filter(values, pc.invert(pc.is_nan(values)))
    if pa.types.is_integer(values.type) or pa.types.is_floating(values.type):
        values = pc.filter(values, pc.greater(values, 0))
    return values


def sample_array(array: pa.Array, limit: int) -> pa.Array:
    if limit <= 0 or len(array) <= limit:
        return array
    indices = pa.array([(index * len(array)) // limit for index in range(limit)], type=pa.int64())
    return pc.take(array, indices)


def sampled_tokens_for_sketch(array: pa.Array, limit: int) -> list[Any]:
    values = token_values(array)
    if len(values) == 0:
        return []
    values = sample_array(values, limit)
    unique_values = pc.unique(values)
    unique_values = sample_array(unique_values, limit)
    return unique_values.to_pylist()


def sample_indices(batch_size: int, target_count: int) -> list[int]:
    if target_count <= 0 or batch_size <= 0:
        return []
    if target_count >= batch_size:
        return list(range(batch_size))
    return [(index * batch_size) // target_count for index in range(target_count)]


def proportional_sample_count(limit: int, batch_rows: int, total_rows: int) -> int:
    if limit <= 0 or batch_rows <= 0 or total_rows <= 0:
        return 0
    if total_rows <= limit:
        return batch_rows
    return min(batch_rows, max(1, math.ceil(limit * batch_rows / float(total_rows))))


def quantile(sorted_values: list[float], quant: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quant
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def schema_signature(schema_path: Path, available_columns: tuple[str, ...]) -> str:
    digest = hashlib.blake2b(digest_size=12)
    digest.update(schema_path.read_bytes())
    for column in available_columns:
        digest.update(b"\0")
        digest.update(column.encode("utf-8", "surrogatepass"))
    return digest.hexdigest()


def build_label_feature_stats(
    *,
    batch: pa.RecordBatch,
    columns: list[str],
    label_flags: list[bool | None],
    target_rows: int,
    config: OnlineDatasetEDAConfig,
    stats: dict[str, Counter[tuple[Any, bool]]],
) -> None:
    if not config.enable_label_lift or not columns or not label_flags:
        return
    count = proportional_sample_count(config.label_feature_sample_rows, batch.num_rows, target_rows)
    indices = sample_indices(batch.num_rows, count)
    if not indices:
        return
    for column in columns:
        values = pc.take(batch_column(batch, column), pa.array(indices, type=pa.int64())).to_pylist()
        counter = stats[column]
        for value, row_index in zip(values, indices, strict=False):
            label_flag = label_flags[row_index]
            if label_flag is None:
                continue
            for token in normalize_list(value):
                counter[(token, bool(label_flag))] += 1


def scan_dataset(
    dataset: DatasetInfo,
    layout: SchemaLayout,
    *,
    row_limit: int | None,
    config: OnlineDatasetEDAConfig,
) -> ScanResult:
    available = set(dataset.available_columns)
    feature_columns = [column for column in layout.feature_columns if column in available]
    sparse_columns = [column for column in layout.sparse_columns if column in available]
    sequence_sideinfo_columns = [
        column
        for domain in layout.sequence_domains
        for column in domain.sideinfo_columns
        if column in available
    ]
    cardinality_columns = dedupe_preserve_order(sparse_columns + sequence_sideinfo_columns)
    dense_columns = [column for column in layout.user_dense_columns if column in available]
    label_columns = [column for column in ("label_type", "label_action_type") if column in available]
    target_label_columns = [column for column in label_columns if column == "label_type"]
    user_key_column = "user_id" if "user_id" in available else layout.primary_user_id_column
    sequence_length_columns = [
        domain.length_column
        for domain in layout.sequence_domains
        if domain.length_column is not None and domain.length_column in available
    ]
    sequence_repeat_columns = [
        domain.repeat_column
        for domain in layout.sequence_domains
        if domain.repeat_column is not None and domain.repeat_column in available
    ]
    iter_columns = dedupe_preserve_order(
        feature_columns
        + target_label_columns
        + ([user_key_column] if user_key_column else [])
        + [column for column in sequence_length_columns if column]
        + [column for column in sequence_repeat_columns if column]
    )

    null_counts = {column: 0 for column in feature_columns}
    cardinality_sketches = {column: KMVSketch(config.cardinality_sketch_k) for column in cardinality_columns}
    overlap_sketches = {column: KMVSketch(config.token_overlap_sketch_k) for column in cardinality_columns}
    dense_stats = {column: DenseStats() for column in dense_columns}
    sequence_stats = {
        domain.name: SequenceStats(
            domain=domain.name,
            sampler=ReservoirSampler(config.sequence_sample_size, seed=stable_hash(domain.name)),
        )
        for domain in layout.sequence_domains
        if domain.length_column in available
    }
    label_counters: dict[str, Counter[str]] = {column: Counter() for column in target_label_columns}
    sampled_user_activity: Counter[Any] = Counter()
    sampled_user_domains: dict[Any, set[str]] = defaultdict(set)
    label_feature_stats: dict[str, Counter[tuple[Any, bool]]] = {column: Counter() for column in sparse_columns}

    target_rows = row_limit if row_limit is not None else dataset.total_rows
    progress = ProgressTracker("profile-scan", target_rows, step_percent=config.progress_step_percent)
    progress.start()
    scanned_rows = 0

    for batch in iter_batches(dataset, columns=iter_columns, max_rows=row_limit, batch_rows=config.batch_rows):
        scanned_rows += batch.num_rows
        label_flags: list[bool | None] = []

        for column in feature_columns:
            array = batch_column(batch, column)
            null_counts[column] += missing_count(array)

        for column in cardinality_columns:
            sampled_tokens = sampled_tokens_for_sketch(batch_column(batch, column), config.token_sample_limit_per_batch)
            cardinality_sketches[column].add_many(sampled_tokens)
            overlap_sketches[column].add_many(sampled_tokens)

        for column in dense_columns:
            dense_stats[column].add_arrow(numeric_values(batch_column(batch, column)))

        if "label_type" in target_label_columns:
            labels = batch_column(batch, "label_type")
            nulls = int(labels.null_count)
            positive_mask = pc.equal(labels, 2)
            positive = scalar_as_int(pc.sum(pc.cast(pc.fill_null(positive_mask, False), pa.int64())))
            observed = batch.num_rows - nulls
            counter = label_counters["label_type"]
            counter["missing"] += nulls
            counter["positive"] += positive
            counter["negative"] += max(0, observed - positive)
            raw_labels = labels.to_pylist()
            label_flags = [None if value is None else value == 2 for value in raw_labels]

        for domain in layout.sequence_domains:
            stats = sequence_stats.get(domain.name)
            if stats is None or domain.length_column is None or domain.length_column not in batch.schema.names:
                continue
            stats.add_lengths(list_lengths(batch_column(batch, domain.length_column)))
            repeat_column = domain.repeat_column
            if repeat_column is not None and repeat_column in batch.schema.names:
                count = proportional_sample_count(config.repeat_sample_rows_per_domain, batch.num_rows, target_rows)
                indices = sample_indices(batch.num_rows, count)
                if indices:
                    values = pc.take(batch_column(batch, repeat_column), pa.array(indices, type=pa.int64())).to_pylist()
                    stats.add_repeat_rates(values)

        if user_key_column is not None and user_key_column in batch.schema.names:
            count = proportional_sample_count(config.user_sample_limit, batch.num_rows, target_rows)
            indices = sample_indices(batch.num_rows, count)
            if indices:
                index_array = pa.array(indices, type=pa.int64())
                user_values = pc.take(batch_column(batch, user_key_column), index_array).to_pylist()
                domain_presence = {
                    domain.name: list_lengths(pc.take(batch_column(batch, domain.length_column), index_array))
                    for domain in layout.sequence_domains
                    if domain.length_column is not None and domain.length_column in batch.schema.names
                }
                for sample_index, raw_user_value in enumerate(user_values):
                    token = hashable_value(raw_user_value)
                    if token is None:
                        continue
                    sampled_user_activity[token] += 1
                    for domain_name, lengths in domain_presence.items():
                        if lengths[sample_index] > 0:
                            sampled_user_domains[token].add(domain_name)

        build_label_feature_stats(
            batch=batch,
            columns=sparse_columns,
            label_flags=label_flags,
            target_rows=target_rows,
            config=config,
            stats=label_feature_stats,
        )
        progress.update(scanned_rows)

    progress.finish(scanned_rows)
    return ScanResult(
        scanned_rows=scanned_rows,
        null_counts=null_counts,
        cardinality_sketches=cardinality_sketches,
        overlap_sketches=overlap_sketches,
        dense_stats=dense_stats,
        sequence_stats=sequence_stats,
        label_counters=label_counters,
        sampled_user_activity=sampled_user_activity,
        sampled_user_domains=dict(sampled_user_domains),
        label_feature_stats=label_feature_stats,
    )


def build_null_rows(feature_columns: list[str], null_counts: dict[str, int], scanned_rows: int) -> list[dict[str, Any]]:
    rows = [
        {"name": column, "null_rate": round(0.0 if scanned_rows == 0 else null_counts[column] / float(scanned_rows), 6)}
        for column in feature_columns
    ]
    rows.sort(key=lambda item: item["null_rate"], reverse=True)
    return rows


def build_cardinality_rows(columns: list[str], sketches: dict[str, KMVSketch]) -> list[dict[str, Any]]:
    rows = [{"name": column, "cardinality": int(sketches[column].estimate())} for column in columns]
    rows.sort(key=lambda item: item["cardinality"], reverse=True)
    return rows


def build_overlap_sketch_rows(columns: list[str], sketches: dict[str, KMVSketch]) -> list[dict[str, Any]]:
    rows = [
        {
            "name": column,
            "sketch_size": len(sketches[column].fingerprint()),
            "hashes": sketches[column].fingerprint(),
        }
        for column in columns
    ]
    rows.sort(key=lambda item: item["sketch_size"], reverse=True)
    return rows


def build_label_distribution_rows(label_counters: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column, counter in label_counters.items():
        total = int(sum(counter.values()))
        missing = int(counter.get("missing", 0))
        positive = int(counter.get("positive", 0))
        negative = int(counter.get("negative", 0))
        observed = positive + negative
        rows.append(
            {
                "name": column,
                "total": total,
                "observed": observed,
                "positive": positive,
                "negative": negative,
                "missing": missing,
                "positive_rate": round(positive / float(observed), 6) if observed else 0.0,
            }
        )
    return rows


def build_sequence_rows(layout: SchemaLayout, sequence_stats: dict[str, SequenceStats]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in layout.sequence_domains:
        stats = sequence_stats.get(domain.name)
        if stats is None or stats.rows == 0:
            continue
        sorted_lengths = sorted(stats.sampler.samples)
        rows.append(
            {
                "domain": domain.name,
                "min": float(stats.min_length or 0),
                "q1": round(quantile(sorted_lengths, 0.25), 6),
                "median": round(quantile(sorted_lengths, 0.5), 6),
                "q3": round(quantile(sorted_lengths, 0.75), 6),
                "max": float(stats.max_length),
                "mean": round(stats.total_length / float(stats.rows), 6),
                "p95": round(quantile(sorted_lengths, 0.95), 6),
                "empty_rate": round(stats.empty_rows / float(stats.rows), 6),
            }
        )
    return rows


def build_repeat_rows(layout: SchemaLayout, sequence_stats: dict[str, SequenceStats]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in layout.sequence_domains:
        stats = sequence_stats.get(domain.name)
        if stats is None:
            continue
        repeat_rate = 0.0 if stats.repeat_rows == 0 else stats.repeat_rate_sum / float(stats.repeat_rows)
        rows.append({"domain": domain.name, "repeat_rate": round(repeat_rate, 6), "sample_rows": stats.repeat_rows})
    return rows


def build_dense_rows(columns: list[str], dense_stats: dict[str, DenseStats]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in columns:
        stats = dense_stats[column]
        if stats.count == 0:
            rows.append({"name": column, "mean": 0.0, "variance": 0.0, "std": 0.0, "zero_frac": 1.0})
            continue
        mean = stats.total / stats.count
        variance = max(stats.total_sq / stats.count - mean * mean, 0.0)
        rows.append(
            {
                "name": column,
                "mean": round(mean, 6),
                "variance": round(variance, 6),
                "std": round(math.sqrt(variance), 6),
                "zero_frac": round(stats.zero_count / float(stats.count), 6),
            }
        )
    return rows


def build_user_activity_rows(activity_counter: Counter[Any]) -> list[dict[str, Any]]:
    if not activity_counter:
        return []
    bucket_counter: Counter[str] = Counter()
    for count in activity_counter.values():
        bucket_counter[str(count) if count < 20 else "20+"] += 1
    numeric_labels = sorted(int(label) for label in bucket_counter if label != "20+")
    rows = [{"bucket": str(label), "user_count": bucket_counter[str(label)]} for label in numeric_labels]
    if "20+" in bucket_counter:
        rows.append({"bucket": "20+", "user_count": bucket_counter["20+"]})
    return rows


def build_domain_coverage_rows(sampled_user_domains: dict[Any, set[str]], domain_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sampled_user_count = len(sampled_user_domains)
    for domain in domain_names:
        user_count = sum(1 for active_domains in sampled_user_domains.values() if domain in active_domains)
        rows.append(
            {
                "domain": domain,
                "sampled_users": sampled_user_count,
                "covered_users": user_count,
                "coverage": round(user_count / float(sampled_user_count), 6) if sampled_user_count else 0.0,
            }
        )
    return rows


def build_overlap_rows(sampled_user_domains: dict[Any, set[str]], domain_names: list[str]) -> list[dict[str, Any]]:
    users_by_domain: dict[str, set[Any]] = {domain: set() for domain in domain_names}
    for user_token, active_domains in sampled_user_domains.items():
        for domain in active_domains:
            if domain in users_by_domain:
                users_by_domain[domain].add(user_token)
    rows: list[dict[str, Any]] = []
    for left in domain_names:
        for right in domain_names:
            union = users_by_domain[left] | users_by_domain[right]
            overlap = 0.0 if not union else len(users_by_domain[left] & users_by_domain[right]) / float(len(union))
            rows.append({"left": left, "right": right, "overlap": round(overlap, 6)})
    return rows


def cardinality_bins(cardinality_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = {"1-10": 0, "11-100": 0, "101-1000": 0, "1001+": 0}
    for row in cardinality_rows:
        cardinality = int(row["cardinality"])
        if cardinality <= 10:
            bins["1-10"] += 1
        elif cardinality <= 100:
            bins["11-100"] += 1
        elif cardinality <= 1000:
            bins["101-1000"] += 1
        else:
            bins["1001+"] += 1
    return [{"name": name, "count": count} for name, count in bins.items()]


def build_label_lift_rows(
    stats_by_column: dict[str, Counter[tuple[Any, bool]]],
    *,
    baseline_positive_rate: float,
    min_support: int,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    epsilon = 0.5
    for column, counter in stats_by_column.items():
        tokens = {token for token, _label in counter}
        for token in tokens:
            positive = counter[(token, True)]
            negative = counter[(token, False)]
            support = positive + negative
            if support < min_support:
                continue
            positive_rate = positive / float(support) if support else 0.0
            lift = positive_rate / baseline_positive_rate if baseline_positive_rate > 0 else 0.0
            rows.append(
                {
                    "feature": column,
                    "token": token,
                    "support": support,
                    "positive": positive,
                    "negative": negative,
                    "positive_rate": round(positive_rate, 6),
                    "baseline_positive_rate": round(baseline_positive_rate, 6),
                    "lift": round(lift, 6),
                    "log_odds": round(math.log((positive + epsilon) / (negative + epsilon)), 6),
                }
            )
    rows.sort(key=lambda item: (abs(item["lift"] - 1.0), item["support"]), reverse=True)
    return rows[:top_k]


def base_subtitle(row_count: int, total_rows: int, role: str) -> str:
    if row_count == total_rows:
        return f"{role} dataset, {row_count} rows"
    return f"{role} dataset, scanned {row_count}/{total_rows} rows"


def build_report(
    dataset: DatasetInfo,
    schema_path: Path,
    row_limit: int | None,
    config: OnlineDatasetEDAConfig,
) -> dict[str, Any]:
    layout = SchemaLayout.from_path(schema_path)
    available = set(dataset.available_columns)
    feature_columns = [column for column in layout.feature_columns if column in available]
    sparse_columns = [column for column in layout.sparse_columns if column in available]
    sequence_sideinfo_columns = [
        column
        for domain in layout.sequence_domains
        for column in domain.sideinfo_columns
        if column in available
    ]
    cardinality_columns = dedupe_preserve_order(sparse_columns + sequence_sideinfo_columns)
    dense_columns = [column for column in layout.user_dense_columns if column in available]

    scan = scan_dataset(dataset, layout, row_limit=row_limit, config=config)
    row_count = scan.scanned_rows
    null_rows = build_null_rows(feature_columns, scan.null_counts, row_count)
    cardinality_rows = build_cardinality_rows(cardinality_columns, scan.cardinality_sketches)
    overlap_sketch_rows = build_overlap_sketch_rows(cardinality_columns, scan.overlap_sketches)
    scalar_cardinality_rows = [row for row in cardinality_rows if row["name"] in set(sparse_columns)]
    sequence_cardinality_rows = [row for row in cardinality_rows if row["name"] in set(sequence_sideinfo_columns)]
    label_distribution_rows = build_label_distribution_rows(scan.label_counters)
    baseline_positive_rate = (
        label_distribution_rows[0]["positive"] / float(label_distribution_rows[0]["observed"])
        if label_distribution_rows and label_distribution_rows[0]["observed"]
        else 0.0
    )
    user_label_rows = build_label_lift_rows(
        {column: stats for column, stats in scan.label_feature_stats.items() if column in set(layout.user_int_columns)},
        baseline_positive_rate=baseline_positive_rate,
        min_support=config.label_feature_min_support,
        top_k=config.label_feature_top_k,
    ) if config.enable_label_lift else []
    item_label_rows = build_label_lift_rows(
        {column: stats for column, stats in scan.label_feature_stats.items() if column in set(layout.item_int_columns)},
        baseline_positive_rate=baseline_positive_rate,
        min_support=config.label_feature_min_support,
        top_k=config.label_feature_top_k,
    ) if config.enable_label_lift else []

    scalar_columns = sorted(column for column in dataset.available_columns if column not in set(layout.feature_columns))
    layout_counts = {
        "scalar": len(scalar_columns),
        "user_int": len(layout.user_int_columns),
        "user_dense": len(layout.user_dense_columns),
        "item_int": len(layout.item_int_columns),
        "sequence": len(layout.sequence_columns),
    }
    domain_names = [domain.name for domain in layout.sequence_domains if domain.length_column in available]
    domain_counts = {domain.name: len(domain.all_columns) for domain in layout.sequence_domains}

    report = {
        "report": "dataset_profile",
        "profile_version": 2,
        "created_at_unix": int(time.time()),
        "dataset_path": str(dataset.dataset_path),
        "schema_path": str(schema_path),
        "schema_signature": schema_signature(schema_path, dataset.available_columns),
        "dataset_role": config.dataset_role,
        "streaming": True,
        "file_count": dataset.file_count,
        "batch_rows": config.batch_rows,
        "label_columns": list(scan.label_counters),
        "label_dependent_analyses_enabled": bool(label_distribution_rows),
        "label_lift_enabled": bool(config.enable_label_lift),
        "row_count": row_count,
        "total_rows": dataset.total_rows,
        "sampled": row_count != dataset.total_rows,
        "max_rows": row_limit,
        "sample_percent": config.sample_percent,
        "reference_profile_path": None,
        "approximation": {
            "cardinality": {
                "method": "kmv_unique_arrow_samples",
                "k": config.cardinality_sketch_k,
                "token_sample_limit_per_batch": config.token_sample_limit_per_batch,
            },
            "token_overlap": {
                "method": "kmv_hash_fingerprint",
                "k": config.token_overlap_sketch_k,
                "token_sample_limit_per_batch": config.token_sample_limit_per_batch,
            },
            "user_activity": {"method": "even_row_sample", "limit": config.user_sample_limit},
            "cross_domain_overlap": {"method": "even_row_sample", "limit": config.user_sample_limit},
            "sequence_quantiles": {"method": "reservoir_sample", "size": config.sequence_sample_size},
            "sequence_repeat_rate": {"method": "even_row_sample", "limit_per_domain": config.repeat_sample_rows_per_domain},
            "label_lift": {"enabled": config.enable_label_lift, "sample_rows": config.label_feature_sample_rows},
        },
        "stats": {
            "column_layout": {"counts": layout_counts, "domain_counts": domain_counts, "scalar_columns": scalar_columns},
            "available_columns": list(dataset.available_columns),
            "null_rates": null_rows,
            "cardinality": cardinality_rows,
            "token_overlap_sketch": overlap_sketch_rows,
            "scalar_cardinality": scalar_cardinality_rows,
            "sequence_token_cardinality": sequence_cardinality_rows,
            "sequence_lengths": build_sequence_rows(layout, scan.sequence_stats),
            "seq_repeat_rate": build_repeat_rows(layout, scan.sequence_stats),
            "dense_distributions": build_dense_rows(dense_columns, scan.dense_stats),
            "cardinality_bins": cardinality_bins(cardinality_rows),
            "user_activity": build_user_activity_rows(scan.sampled_user_activity),
            "cross_domain_coverage": build_domain_coverage_rows(scan.sampled_user_domains, domain_names),
            "cross_domain_overlap": build_overlap_rows(scan.sampled_user_domains, domain_names),
            "label_distribution": label_distribution_rows,
            "user_feature_label_lift": user_label_rows,
            "item_feature_label_lift": item_label_rows,
            "co_missing": {"columns": []},
            "categorical_pair_associations": [],
            "feature_auc": [],
            "null_rate_by_label": [],
        },
    }
    return report


def _row_map(rows: list[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    return {row[key]: row for row in rows if key in row}


def _relative_delta(current: float, reference: float) -> float:
    denominator = max(abs(reference), 1e-12)
    return (current - reference) / denominator


def _top_deltas(rows: list[dict[str, Any]], value_key: str, top_k: int) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: abs(float(row[value_key])), reverse=True)
    return rows[:top_k]


def compare_profiles(reference: dict[str, Any], current: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    reference_stats = reference.get("stats", {}) if isinstance(reference.get("stats"), dict) else {}
    current_stats = current.get("stats", {}) if isinstance(current.get("stats"), dict) else {}
    risk_flags: list[str] = []

    reference_columns = set(reference_stats.get("available_columns", []))
    current_columns = set(current_stats.get("available_columns", []))
    missing_columns = sorted(reference_columns - current_columns)
    new_columns = sorted(current_columns - reference_columns)
    if missing_columns:
        risk_flags.append(f"missing_columns={len(missing_columns)}")
    if new_columns:
        risk_flags.append(f"new_columns={len(new_columns)}")

    reference_label = _row_map(reference_stats.get("label_distribution", []), "name")
    current_label = _row_map(current_stats.get("label_distribution", []), "name")
    label_rows: list[dict[str, Any]] = []
    for name in sorted(set(reference_label) & set(current_label)):
        ref_rate = float(reference_label[name].get("positive_rate", 0.0))
        cur_rate = float(current_label[name].get("positive_rate", 0.0))
        delta = cur_rate - ref_rate
        label_rows.append({"name": name, "reference_positive_rate": ref_rate, "current_positive_rate": cur_rate, "delta": round(delta, 6)})
        if abs(delta) >= 0.01:
            risk_flags.append(f"label_rate_shift:{name}={delta:+.4f}")

    reference_null = _row_map(reference_stats.get("null_rates", []), "name")
    current_null = _row_map(current_stats.get("null_rates", []), "name")
    null_rows = []
    for name in set(reference_null) & set(current_null):
        ref_value = float(reference_null[name].get("null_rate", 0.0))
        cur_value = float(current_null[name].get("null_rate", 0.0))
        null_rows.append({"name": name, "reference": ref_value, "current": cur_value, "delta": round(cur_value - ref_value, 6)})
    null_rows = _top_deltas(null_rows, "delta", top_k)
    if null_rows and abs(float(null_rows[0]["delta"])) >= 0.05:
        risk_flags.append(f"null_rate_shift:{null_rows[0]['name']}={null_rows[0]['delta']:+.4f}")

    reference_cardinality = _row_map(reference_stats.get("cardinality", []), "name")
    current_cardinality = _row_map(current_stats.get("cardinality", []), "name")
    cardinality_rows = []
    for name in set(reference_cardinality) & set(current_cardinality):
        ref_value = int(reference_cardinality[name].get("cardinality", 0))
        cur_value = int(current_cardinality[name].get("cardinality", 0))
        log2_ratio = math.log2((cur_value + 1.0) / (ref_value + 1.0))
        cardinality_rows.append({"name": name, "reference": ref_value, "current": cur_value, "log2_ratio": round(log2_ratio, 6)})
    cardinality_rows = _top_deltas(cardinality_rows, "log2_ratio", top_k)
    if cardinality_rows and abs(float(cardinality_rows[0]["log2_ratio"])) >= 0.5:
        risk_flags.append(f"cardinality_shift:{cardinality_rows[0]['name']}={cardinality_rows[0]['log2_ratio']:+.3f}log2")

    reference_overlap = _row_map(reference_stats.get("token_overlap_sketch", []), "name")
    current_overlap = _row_map(current_stats.get("token_overlap_sketch", []), "name")
    overlap_rows = []
    for name in set(reference_overlap) & set(current_overlap):
        ref_hashes = set(reference_overlap[name].get("hashes", []))
        cur_hashes = set(current_overlap[name].get("hashes", []))
        if not ref_hashes and not cur_hashes:
            continue
        intersection = len(ref_hashes & cur_hashes)
        union = len(ref_hashes | cur_hashes)
        current_size = len(cur_hashes)
        reference_size = len(ref_hashes)
        overlap_rows.append(
            {
                "name": name,
                "reference_sketch_size": reference_size,
                "current_sketch_size": current_size,
                "intersection": intersection,
                "jaccard": round(intersection / float(union), 6) if union else 1.0,
                "current_novel_sketch_rate": round(1.0 - intersection / float(current_size), 6) if current_size else 0.0,
                "reference_only_sketch_rate": round(1.0 - intersection / float(reference_size), 6) if reference_size else 0.0,
            }
        )
    overlap_rows.sort(key=lambda row: float(row["current_novel_sketch_rate"]), reverse=True)
    overlap_rows = overlap_rows[:top_k]
    if overlap_rows and float(overlap_rows[0]["current_novel_sketch_rate"]) >= 0.30:
        risk_flags.append(f"token_oov_sketch:{overlap_rows[0]['name']}={overlap_rows[0]['current_novel_sketch_rate']:.2%}")

    reference_seq = _row_map(reference_stats.get("sequence_lengths", []), "domain")
    current_seq = _row_map(current_stats.get("sequence_lengths", []), "domain")
    sequence_rows = []
    for domain in sorted(set(reference_seq) & set(current_seq)):
        ref_mean = float(reference_seq[domain].get("mean", 0.0))
        cur_mean = float(current_seq[domain].get("mean", 0.0))
        ref_p95 = float(reference_seq[domain].get("p95", 0.0))
        cur_p95 = float(current_seq[domain].get("p95", 0.0))
        ref_empty = float(reference_seq[domain].get("empty_rate", 0.0))
        cur_empty = float(current_seq[domain].get("empty_rate", 0.0))
        sequence_rows.append(
            {
                "domain": domain,
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "mean_rel_delta": round(_relative_delta(cur_mean, ref_mean), 6),
                "reference_p95": ref_p95,
                "current_p95": cur_p95,
                "p95_rel_delta": round(_relative_delta(cur_p95, ref_p95), 6),
                "empty_rate_delta": round(cur_empty - ref_empty, 6),
            }
        )
    sequence_rows.sort(
        key=lambda row: max(abs(float(row["mean_rel_delta"])), abs(float(row["p95_rel_delta"])), abs(float(row["empty_rate_delta"]))),
        reverse=True,
    )
    sequence_rows = sequence_rows[:top_k]
    if sequence_rows and abs(float(sequence_rows[0]["mean_rel_delta"])) >= 0.10:
        risk_flags.append(f"sequence_length_shift:{sequence_rows[0]['domain']}={sequence_rows[0]['mean_rel_delta']:+.2%}")

    reference_dense = _row_map(reference_stats.get("dense_distributions", []), "name")
    current_dense = _row_map(current_stats.get("dense_distributions", []), "name")
    dense_rows = []
    for name in set(reference_dense) & set(current_dense):
        ref_mean = float(reference_dense[name].get("mean", 0.0))
        cur_mean = float(current_dense[name].get("mean", 0.0))
        ref_std = float(reference_dense[name].get("std", 0.0))
        cur_std = float(current_dense[name].get("std", 0.0))
        dense_rows.append(
            {
                "name": name,
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "mean_rel_delta": round(_relative_delta(cur_mean, ref_mean), 6),
                "reference_std": ref_std,
                "current_std": cur_std,
                "std_rel_delta": round(_relative_delta(cur_std, ref_std), 6),
            }
        )
    dense_rows.sort(key=lambda row: max(abs(float(row["mean_rel_delta"])), abs(float(row["std_rel_delta"]))), reverse=True)
    dense_rows = dense_rows[:top_k]
    if dense_rows and abs(float(dense_rows[0]["mean_rel_delta"])) >= 0.20:
        risk_flags.append(f"dense_mean_shift:{dense_rows[0]['name']}={dense_rows[0]['mean_rel_delta']:+.2%}")

    return {
        "reference_dataset_role": reference.get("dataset_role"),
        "current_dataset_role": current.get("dataset_role"),
        "reference_row_count": reference.get("row_count"),
        "current_row_count": current.get("row_count"),
        "schema_signature_match": reference.get("schema_signature") == current.get("schema_signature"),
        "missing_columns": missing_columns[:top_k],
        "new_columns": new_columns[:top_k],
        "label_rate_drift": label_rows,
        "null_rate_drift": null_rows,
        "cardinality_drift": cardinality_rows,
        "token_overlap_drift": overlap_rows,
        "sequence_length_drift": sequence_rows,
        "dense_distribution_drift": dense_rows,
        "risk_flags": risk_flags,
    }


def print_section(title: str) -> None:
    write_stdout_line(f"\n== {title} ==")


def print_key_values(rows: list[tuple[str, Any]]) -> None:
    for key, value in rows:
        write_stdout_line(f"{key}: {value}")


def print_ranked_rows(rows: list[dict[str, Any]], *, name_key: str, value_keys: list[str], limit: int) -> None:
    if not rows:
        write_stdout_line("(empty)")
        return
    for index, row in enumerate(rows[:limit], start=1):
        fragments = [f"{key}={row[key]}" for key in value_keys if key in row]
        write_stdout_line(f"{index}. {row[name_key]} | " + " | ".join(fragments))


def print_label_distribution_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_stdout_line("(empty)")
        return
    for row in rows:
        write_stdout_line(
            f"{row['name']}: positive={row['positive']} negative={row['negative']} "
            f"observed={row['observed']} missing={row['missing']} positive_rate={row['positive_rate']}"
        )


def print_domain_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_stdout_line("(empty)")
        return
    for row in rows:
        write_stdout_line(
            f"{row['domain']}: mean={row['mean']} p95={row['p95']} empty_rate={row['empty_rate']} "
            f"min={row['min']} median={row['median']} max={row['max']}"
        )


def print_coverage_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_stdout_line("(empty)")
        return
    for row in rows:
        write_stdout_line(
            f"{row['domain']}: covered_users={row['covered_users']} sampled_users={row['sampled_users']} coverage={row['coverage']}"
        )


def print_overlap_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_stdout_line("(empty)")
        return
    domain_names = [row["left"] for row in rows if row["left"] == row["right"]]
    for left in domain_names:
        pieces = [
            f"{right}={next(row for row in rows if row['left'] == left and row['right'] == right)['overlap']}"
            for right in domain_names
        ]
        write_stdout_line(f"{left}: " + ", ".join(pieces))


def print_label_lift_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_stdout_line("(empty)")
        return
    for index, row in enumerate(rows, start=1):
        write_stdout_line(
            f"{index}. {row['feature']}={row['token']} | support={row['support']} "
            f"positive_rate={row['positive_rate']} lift={row['lift']} log_odds={row['log_odds']}"
        )


def print_drift_rows(rows: list[dict[str, Any]], *, name_key: str, value_keys: list[str]) -> None:
    if not rows:
        write_stdout_line("(empty)")
        return
    for index, row in enumerate(rows, start=1):
        fragments = [f"{key}={row[key]}" for key in value_keys if key in row]
        write_stdout_line(f"{index}. {row[name_key]} | " + " | ".join(fragments))


def print_report(report: dict[str, Any]) -> None:
    write_stdout_line("ONLINE_DATASET_EDA_RESULT=" + dumps(report))


def run_online_dataset_eda(config: OnlineDatasetEDAConfig) -> dict[str, Any]:
    configure_logging()
    config = apply_env_overrides(config)
    validate_config(config)
    dataset_path = config.dataset_path.expanduser().resolve() if config.dataset_path is not None else None
    if dataset_path is None:
        raise SystemExit("dataset path is required")
    schema_path = resolve_schema_path(dataset_path, config.schema_path)
    dataset = build_dataset_info(dataset_path)
    effective_max_rows = resolve_scan_row_limit(dataset.total_rows, config.max_rows, config.sample_percent)
    logger.info("[online-eda] role={}", config.dataset_role)
    logger.info("[online-eda] dataset={}", dataset_path)
    logger.info("[online-eda] schema={}", schema_path)
    logger.info("[online-eda] files={} columns={}", dataset.file_count, len(dataset.available_columns))
    if config.sample_percent is not None:
        logger.info(
            "[online-eda] scan=arrow-profile sample_percent={:.1f} max_rows={} batch_rows={}",
            config.sample_percent,
            effective_max_rows,
            config.batch_rows,
        )
    elif effective_max_rows is None:
        logger.info("[online-eda] scan=arrow-profile full batch_rows={}", config.batch_rows)
    else:
        logger.info("[online-eda] scan=arrow-profile max_rows={} batch_rows={}", effective_max_rows, config.batch_rows)

    report = build_report(dataset, schema_path, effective_max_rows, config)
    report["top_k"] = config.top_k
    reference_path = resolve_reference_profile_path(config.reference_profile_path)
    if reference_path is not None:
        reference_profile = read_path(reference_path)
        report["reference_profile_path"] = str(reference_path)
        report["reference_profile_source"] = "path"
        report["comparison"] = compare_profiles(reference_profile, report, top_k=config.top_k)
    elif config.reference_profile_json:
        reference_profile = loads(config.reference_profile_json.strip().removeprefix("ONLINE_DATASET_EDA_RESULT="))
        report["reference_profile_source"] = "ONLINE_EDA_REFERENCE_PROFILE_JSON"
        report["comparison"] = compare_profiles(reference_profile, report, top_k=config.top_k)
    print_report(report)
    return report


__all__ = [
    "DRIFT_FILENAME",
    "PROFILE_FILENAME",
    "OnlineDatasetEDAConfig",
    "apply_env_overrides",
    "compare_profiles",
    "resolve_reference_profile_path",
    "resolve_schema_path",
    "run_online_dataset_eda",
]

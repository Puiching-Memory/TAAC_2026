from __future__ import annotations

import hashlib
import heapq
import math
import os
import random
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taac2026.infrastructure.io.json import read_path
from taac2026.infrastructure.io.streams import write_stdout_line
from taac2026.infrastructure.logging import configure_logging, logger

try:
  import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - runtime guard
  raise SystemExit("pyarrow is required for online dataset EDA") from exc


DEFAULT_BATCH_ROWS = 128
DEFAULT_CARDINALITY_SKETCH_K = 4096
DEFAULT_USER_SAMPLE_LIMIT = 50000
DEFAULT_SEQUENCE_SAMPLE_SIZE = 16384
DEFAULT_PROGRESS_STEP_PERCENT = 10.0
DEFAULT_LABEL_FEATURE_CANDIDATE_K = 128
DEFAULT_LABEL_FEATURE_TOP_K = 20
DEFAULT_LABEL_FEATURE_MIN_SUPPORT = 20
DEFAULT_CATEGORICAL_PAIR_MAX_COLUMNS = 16
DEFAULT_CATEGORICAL_PAIR_MAX_CARDINALITY = 128
DEFAULT_CATEGORICAL_PAIR_SAMPLE_ROWS = 50000
DEFAULT_CATEGORICAL_PAIR_TOP_K = 20
SKIPPED_CHARTS = ["feature_auc"]
UINT64_MASK = (1 << 64) - 1


@dataclass(slots=True)
class OnlineDatasetEDAConfig:
  dataset_path: Path | None = None
  schema_path: Path | None = None
  batch_rows: int = DEFAULT_BATCH_ROWS
  cardinality_sketch_k: int = DEFAULT_CARDINALITY_SKETCH_K
  user_sample_limit: int = DEFAULT_USER_SAMPLE_LIMIT
  sequence_sample_size: int = DEFAULT_SEQUENCE_SAMPLE_SIZE
  max_rows: int | None = None
  sample_percent: float | None = None
  progress_step_percent: float = DEFAULT_PROGRESS_STEP_PERCENT
  label_feature_candidate_k: int = DEFAULT_LABEL_FEATURE_CANDIDATE_K
  label_feature_top_k: int = DEFAULT_LABEL_FEATURE_TOP_K
  label_feature_min_support: int = DEFAULT_LABEL_FEATURE_MIN_SUPPORT
  categorical_pair_max_columns: int = DEFAULT_CATEGORICAL_PAIR_MAX_COLUMNS
  categorical_pair_max_cardinality: int = DEFAULT_CATEGORICAL_PAIR_MAX_CARDINALITY
  categorical_pair_sample_rows: int = DEFAULT_CATEGORICAL_PAIR_SAMPLE_ROWS
  categorical_pair_top_k: int = DEFAULT_CATEGORICAL_PAIR_TOP_K


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
    user_int_columns = tuple(f"user_int_feats_{fid}" for fid, _vocab_size, _dim in raw["user_int"])
    item_int_columns = tuple(f"item_int_feats_{fid}" for fid, _vocab_size, _dim in raw["item_int"])
    user_dense_columns = tuple(f"user_dense_feats_{fid}" for fid, _dim in raw["user_dense"])
    sequence_domains: list[SequenceDomainLayout] = []
    for name, config in sorted(raw["seq"].items()):
      prefix = str(config["prefix"])
      ts_fid = config.get("ts_fid")
      ts_column = f"{prefix}_{ts_fid}" if ts_fid is not None else None
      sideinfo_columns = tuple(
        f"{prefix}_{fid}"
        for fid, _vocab_size in config["features"]
        if fid != ts_fid
      )
      sequence_domains.append(
        SequenceDomainLayout(
          name=name,
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

  @property
  def group_by_column(self) -> dict[str, str]:
    result: dict[str, str] = {}
    result.update({column: "user_int" for column in self.user_int_columns})
    result.update({column: "user_dense" for column in self.user_dense_columns})
    result.update({column: "item_int" for column in self.item_int_columns})
    for domain in self.sequence_domains:
      for column in domain.all_columns:
        result[column] = domain.name
    return result


@dataclass(slots=True)
class DatasetInfo:
  dataset_path: Path
  files: tuple[Path, ...]
  available_columns: tuple[str, ...]
  total_rows: int


class KMVSketch:
  def __init__(self, limit: int) -> None:
    self.limit = limit
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

  def estimate(self) -> int:
    if len(self._values) < self.limit:
      return len(self._values)
    threshold = max(self._values) / float(UINT64_MASK)
    if threshold <= 0:
      return len(self._values)
    estimate = (self.limit - 1) / threshold
    return round(estimate)


class BottomKUserSampler:
  def __init__(self, limit: int) -> None:
    self.limit = limit
    self._hash_by_token: dict[Any, int] = {}
    self._heap: list[tuple[int, Any]] = []

  def consider(self, token: Any) -> None:
    if token is None or token in self._hash_by_token:
      return
    hashed = stable_hash(token)
    if len(self._hash_by_token) < self.limit:
      self._hash_by_token[token] = hashed
      heapq.heappush(self._heap, (-hashed, token))
      return
    largest_hash = -self._heap[0][0]
    if hashed >= largest_hash:
      return
    _, removed_token = heapq.heapreplace(self._heap, (-hashed, token))
    self._hash_by_token.pop(removed_token, None)
    self._hash_by_token[token] = hashed

  def tokens(self) -> set[Any]:
    return set(self._hash_by_token)


class ReservoirSampler:
  def __init__(self, limit: int, seed: int) -> None:
    self.limit = limit
    self._rng = random.Random(seed)
    self.samples: list[float] = []
    self.seen = 0

  def add(self, value: float) -> None:
    self.seen += 1
    if len(self.samples) < self.limit:
      self.samples.append(value)
      return
    index = self._rng.randrange(self.seen)
    if index < self.limit:
      self.samples[index] = value


class TopKTokenSketch:
  def __init__(self, limit: int) -> None:
    self.limit = limit
    self._counts: Counter[Any] = Counter()

  def add(self, token: Any) -> None:
    if self.limit <= 0:
      return
    if token in self._counts or len(self._counts) < self.limit:
      self._counts[token] += 1
      return
    for existing in list(self._counts):
      self._counts[existing] -= 1
      if self._counts[existing] <= 0:
        del self._counts[existing]

  def tokens(self) -> set[Any]:
    return set(self._counts)


class RowReservoirSampler:
  def __init__(self, limit: int, seed: int) -> None:
    self.limit = limit
    self._rng = random.Random(seed)
    self.samples: list[dict[str, Any]] = []
    self.seen = 0

  def add(self, row: dict[str, Any]) -> None:
    if self.limit <= 0:
      return
    self.seen += 1
    if len(self.samples) < self.limit:
      self.samples.append(row)
      return
    index = self._rng.randrange(self.seen)
    if index < self.limit:
      self.samples[index] = row


@dataclass(slots=True)
class DenseStats:
  count: int = 0
  total: float = 0.0
  total_sq: float = 0.0
  zero_count: int = 0

  def add(self, value: float) -> None:
    self.count += 1
    self.total += value
    self.total_sq += value * value
    if value == 0.0:
      self.zero_count += 1


@dataclass(slots=True)
class LabelNullStats:
  positive_rows: int = 0
  negative_rows: int = 0
  positive_nulls: int = 0
  negative_nulls: int = 0

  def add(self, *, is_positive: bool, missing: bool) -> None:
    if is_positive:
      self.positive_rows += 1
      if missing:
        self.positive_nulls += 1
      return
    self.negative_rows += 1
    if missing:
      self.negative_nulls += 1


@dataclass(slots=True)
class LabelTokenStats:
  positive: int = 0
  negative: int = 0

  def add(self, *, is_positive: bool) -> None:
    if is_positive:
      self.positive += 1
    else:
      self.negative += 1

  @property
  def support(self) -> int:
    return self.positive + self.negative


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

  def add_length(self, length: int) -> None:
    self.rows += 1
    self.total_length += length
    if length == 0:
      self.empty_rows += 1
    if self.min_length is None or length < self.min_length:
      self.min_length = length
    if length > self.max_length:
      self.max_length = length
    self.sampler.add(float(length))

  def add_repeat_rate(self, repeat_rate: float) -> None:
    self.repeat_rate_sum += repeat_rate
    self.repeat_rows += 1


@dataclass(slots=True)
class FirstPassResult:
  scanned_rows: int
  null_counts: dict[str, int]
  cardinality_sketches: dict[str, KMVSketch]
  dense_stats: dict[str, DenseStats]
  sequence_stats: dict[str, SequenceStats]
  user_sampler: BottomKUserSampler
  label_counters: dict[str, Counter[Any]]
  label_null_stats: dict[str, LabelNullStats]
  label_feature_candidates: dict[str, set[Any]]


@dataclass(slots=True)
class SecondPassResult:
  co_missing_counts: list[list[int]]
  sampled_user_activity: Counter[Any]
  sampled_user_domains: dict[Any, set[str]]
  label_feature_stats: dict[str, dict[Any, LabelTokenStats]]
  categorical_samples: list[dict[str, Any]]


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
  if raw_value:
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
  if config.max_rows is not None and config.max_rows <= 0:
    raise SystemExit("ONLINE_EDA_MAX_ROWS must be positive when set")
  if config.sample_percent is not None and not (0.0 < config.sample_percent <= 100.0):
    raise SystemExit("ONLINE_EDA_SAMPLE_PERCENT must be in (0, 100] when set")
  if config.max_rows is not None and config.sample_percent is not None:
    raise SystemExit("ONLINE_EDA_MAX_ROWS and ONLINE_EDA_SAMPLE_PERCENT are mutually exclusive")
  if config.label_feature_candidate_k < 0:
    raise SystemExit("label_feature_candidate_k must be non-negative")
  if config.label_feature_top_k < 0:
    raise SystemExit("label_feature_top_k must be non-negative")
  if config.label_feature_min_support < 0:
    raise SystemExit("label_feature_min_support must be non-negative")
  if config.categorical_pair_max_columns < 0:
    raise SystemExit("categorical_pair_max_columns must be non-negative")
  if config.categorical_pair_max_cardinality < 0:
    raise SystemExit("categorical_pair_max_cardinality must be non-negative")
  if config.categorical_pair_sample_rows < 0:
    raise SystemExit("categorical_pair_sample_rows must be non-negative")
  if config.categorical_pair_top_k < 0:
    raise SystemExit("categorical_pair_top_k must be non-negative")


def list_parquet_files(dataset_path: Path) -> tuple[Path, ...]:
  if dataset_path.is_dir():
    files = tuple(sorted(dataset_path.glob("*.parquet")))
  else:
    files = (dataset_path,)
  if not files:
    raise SystemExit(f"no .parquet files found at {dataset_path}")
  return files


def build_dataset_info(dataset_path: Path) -> DatasetInfo:
  files = list_parquet_files(dataset_path)
  available_columns = tuple(pq.ParquetFile(files[0]).schema_arrow.names)
  total_rows = 0
  for file_path in files:
    total_rows += pq.ParquetFile(file_path).metadata.num_rows
  return DatasetInfo(
    dataset_path=dataset_path,
    files=files,
    available_columns=available_columns,
    total_rows=total_rows,
  )


def iter_batches(dataset: DatasetInfo, *, columns: list[str], max_rows: int | None, batch_rows: int):
  remaining = max_rows
  for file_path in dataset.files:
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=batch_rows, columns=columns):
      if remaining is not None:
        if remaining <= 0:
          return
        if batch.num_rows > remaining:
          batch = batch.slice(0, remaining)
        remaining -= batch.num_rows
      yield batch
      if remaining == 0:
        return


def get_batch_values(batch, cache: dict[str, list[Any]], column_name: str) -> list[Any]:
  values = cache.get(column_name)
  if values is None:
    values = batch.column(batch.schema.get_field_index(column_name)).to_pylist()
    cache[column_name] = values
  return values


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
    return [item for item in value if normalize_scalar(item) is not None]
  scalar = normalize_scalar(value)
  return [] if scalar is None else [scalar]


def is_missing(value: Any) -> bool:
  if isinstance(value, list):
    return len(normalize_list(value)) == 0
  return normalize_scalar(value) is None


def sparse_tokens(value: Any) -> tuple[Any, ...]:
  tokens: list[Any] = []
  for item in normalize_list(value):
    if isinstance(item, (int, float)) and item <= 0:
      continue
    tokens.append(item)
  return tuple(tokens)


def hashable_value(value: Any) -> Any:
  if isinstance(value, list):
    tokens = sparse_tokens(value)
    return tokens or None
  scalar = normalize_scalar(value)
  if isinstance(scalar, (int, float)) and scalar <= 0:
    return None
  return scalar


def binary_label(value: Any) -> bool | None:
  normalized = normalize_scalar(value)
  if normalized is None:
    return None
  return normalized == 2


def row_sparse_tokens(value: Any) -> tuple[Any, ...]:
  return tuple(dict.fromkeys(sparse_tokens(value)))


def row_categorical_token(value: Any) -> Any:
  if isinstance(value, list):
    tokens = row_sparse_tokens(value)
    if not tokens:
      return None
    if len(tokens) == 1:
      return tokens[0]
    return tokens
  return hashable_value(value)


def column_null_rate(values: list[Any]) -> float:
  if not values:
    return 0.0
  return sum(1 for value in values if is_missing(value)) / float(len(values))


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


def chart_title(text: str, subtitle: str) -> dict[str, str]:
  return {"text": text, "subtext": subtitle}


def series_bar(names: list[str], values: list[float], *, title: str, subtitle: str, series_name: str, horizontal: bool = False) -> dict[str, Any]:
  x_axis = {"type": "category", "data": names, "axisLabel": {"rotate": 35}} if not horizontal else {"type": "value"}
  y_axis = {"type": "value"} if not horizontal else {"type": "category", "data": names}
  return {
    "title": chart_title(title, subtitle),
    "tooltip": {"trigger": "axis"},
    "grid": {"left": 100 if horizontal else 60, "right": 30, "top": 70, "bottom": 90 if not horizontal else 40},
    "xAxis": x_axis,
    "yAxis": y_axis,
    "series": [{"name": series_name, "type": "bar", "data": values, "itemStyle": {"borderRadius": 6}}],
  }


def heatmap_chart(*, title: str, subtitle: str, x_labels: list[str], y_labels: list[str], data: list[list[Any]], value_name: str) -> dict[str, Any]:
  return {
    "title": chart_title(title, subtitle),
    "tooltip": {"trigger": "item"},
    "grid": {"left": 80, "right": 30, "top": 70, "bottom": 110},
    "xAxis": {"type": "category", "data": x_labels, "axisLabel": {"rotate": 45}},
    "yAxis": {"type": "category", "data": y_labels},
    "visualMap": {"min": 0, "max": 1, "orient": "horizontal", "left": "center", "bottom": 25, "text": [value_name, "0"]},
    "series": [{"type": "heatmap", "data": data, "itemStyle": {"borderRadius": 2}}],
  }


def boxplot_chart(domains: list[str], stats: list[list[float]], subtitle: str) -> dict[str, Any]:
  return {
    "title": chart_title("序列长度分布", subtitle),
    "tooltip": {"trigger": "item"},
    "grid": {"left": 60, "right": 30, "top": 70, "bottom": 60},
    "xAxis": {"type": "category", "data": domains},
    "yAxis": {"type": "value", "name": "length"},
    "series": [{"type": "boxplot", "data": stats}],
  }


def sequence_summary_chart(domains: list[str], means: list[float], p95s: list[float], empty_rates: list[float], subtitle: str) -> dict[str, Any]:
  return {
    "title": chart_title("序列长度摘要", subtitle),
    "tooltip": {"trigger": "axis"},
    "legend": {"bottom": 0},
    "grid": {"left": 60, "right": 60, "top": 70, "bottom": 90},
    "xAxis": {"type": "category", "data": domains},
    "yAxis": [{"type": "value", "name": "length"}, {"type": "value", "name": "empty_rate", "min": 0, "max": 1}],
    "series": [
      {"name": "mean", "type": "bar", "data": means, "itemStyle": {"borderRadius": 6}},
      {"name": "p95", "type": "bar", "data": p95s, "itemStyle": {"borderRadius": 6}},
      {"name": "empty_rate", "type": "line", "yAxisIndex": 1, "data": empty_rates, "smooth": True},
    ],
  }


def scatter_chart(*, title: str, subtitle: str, points: list[dict[str, Any]]) -> dict[str, Any]:
  return {
    "title": chart_title(title, subtitle),
    "tooltip": {"trigger": "item"},
    "xAxis": {"type": "value", "name": "mean"},
    "yAxis": {"type": "value", "name": "std"},
    "series": [{"type": "scatter", "label": {"show": True, "formatter": "{b}", "position": "top"}, "data": points}],
  }


def column_layout_chart(layout_counts: dict[str, int], domain_counts: dict[str, int], subtitle: str) -> dict[str, Any]:
  return {
    "title": chart_title("列布局概览", subtitle),
    "tooltip": {"trigger": "axis"},
    "legend": {"bottom": 0},
    "grid": {"left": 50, "right": 30, "top": 70, "bottom": 100},
    "xAxis": {"type": "category", "data": ["scalar", "user_int", "user_dense", "item_int", "sequence"]},
    "yAxis": {"type": "value", "name": "columns"},
    "series": [
      {
        "name": "column_count",
        "type": "bar",
        "data": [
          layout_counts["scalar"],
          layout_counts["user_int"],
          layout_counts["user_dense"],
          layout_counts["item_int"],
          layout_counts["sequence"],
        ],
        "itemStyle": {"borderRadius": 6},
      },
      {
        "name": "sequence_domains",
        "type": "pie",
        "radius": ["35%", "55%"],
        "center": ["82%", "35%"],
        "label": {"formatter": "{b}: {c}"},
        "data": [{"name": key, "value": value} for key, value in domain_counts.items()],
      },
    ],
  }


def base_subtitle(row_count: int, total_rows: int) -> str:
  if row_count == total_rows:
    return f"online dataset, {row_count} rows"
  return f"online dataset, scanned {row_count}/{total_rows} rows"


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
    fragments = [f"{key}={row[key]}" for key in value_keys]
    write_stdout_line(f"{index}. {row[name_key]} | " + " | ".join(fragments))


def print_domain_rows(rows: list[dict[str, Any]]) -> None:
  if not rows:
    write_stdout_line("(empty)")
    return
  for row in rows:
    write_stdout_line(
      f"{row['domain']}: mean={row['mean']} p95={row['p95']} empty_rate={row['empty_rate']} "
      f"min={row['min']} median={row['median']} max={row['max']}"
    )


def print_overlap_rows(rows: list[dict[str, Any]], domain_names: list[str]) -> None:
  if not rows or not domain_names:
    write_stdout_line("(empty)")
    return
  for left in domain_names:
    pieces = []
    for right in domain_names:
      match = next(row for row in rows if row["left"] == left and row["right"] == right)
      pieces.append(f"{right}={match['overlap']}")
    write_stdout_line(f"{left}: " + ", ".join(pieces))


def print_label_distribution_rows(rows: list[dict[str, Any]]) -> None:
  if not rows:
    write_stdout_line("(empty)")
    return
  for row in rows:
    write_stdout_line(
      f"{row['name']}: positive={row['positive']} negative={row['negative']} "
      f"observed={row['observed']} missing={row['missing']} positive_rate={row['positive_rate']}"
    )


def print_coverage_rows(rows: list[dict[str, Any]]) -> None:
  if not rows:
    write_stdout_line("(empty)")
    return
  for row in rows:
    write_stdout_line(
      f"{row['domain']}: covered_users={row['covered_users']} sampled_users={row['sampled_users']} coverage={row['coverage']}"
    )


def print_label_lift_rows(rows: list[dict[str, Any]]) -> None:
  if not rows:
    write_stdout_line("(empty)")
    return
  for index, row in enumerate(rows, start=1):
    write_stdout_line(
      f"{index}. {row['feature']}={row['token']} | group={row['group']} support={row['support']} "
      f"positive_rate={row['positive_rate']} lift={row['lift']} log_odds={row['log_odds']}"
    )


def print_categorical_pair_rows(rows: list[dict[str, Any]]) -> None:
  if not rows:
    write_stdout_line("(empty)")
    return
  for index, row in enumerate(rows, start=1):
    write_stdout_line(
      f"{index}. {row['left']} ~ {row['right']} | cramers_v={row['cramers_v']} "
      f"sample_rows={row['sample_rows']} left_cardinality={row['left_cardinality']} right_cardinality={row['right_cardinality']}"
    )


def build_null_rows(feature_columns: list[str], null_counts: dict[str, int], scanned_rows: int) -> list[dict[str, Any]]:
  rows = []
  for column in feature_columns:
    null_rate = 0.0 if scanned_rows == 0 else null_counts[column] / float(scanned_rows)
    rows.append({"name": column, "null_rate": round(null_rate, 6)})
  rows.sort(key=lambda item: item["null_rate"], reverse=True)
  return rows


def build_null_by_label_rows(label_null_stats: dict[str, LabelNullStats]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for column, stats in label_null_stats.items():
    positive_rate = stats.positive_nulls / float(stats.positive_rows) if stats.positive_rows else 0.0
    negative_rate = stats.negative_nulls / float(stats.negative_rows) if stats.negative_rows else 0.0
    rows.append(
      {
        "name": column,
        "positive_null_rate": round(positive_rate, 6),
        "negative_null_rate": round(negative_rate, 6),
        "delta": round(positive_rate - negative_rate, 6),
        "positive_rows": stats.positive_rows,
        "negative_rows": stats.negative_rows,
      }
    )
  rows.sort(key=lambda item: abs(item["delta"]), reverse=True)
  return rows


def build_cardinality_rows(sparse_columns: list[str], sketches: dict[str, KMVSketch]) -> list[dict[str, Any]]:
  rows = []
  for column in sparse_columns:
    rows.append({"name": column, "cardinality": int(sketches[column].estimate())})
  rows.sort(key=lambda item: item["cardinality"], reverse=True)
  return rows


def build_label_distribution_rows(label_columns: list[str], label_counters: dict[str, Counter[Any]]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for column in label_columns:
    counter = label_counters.get(column, Counter())
    total = int(sum(counter.values()))
    missing_count = int(counter.get("missing", 0))
    positive_count = int(counter.get("positive", 0))
    negative_count = int(counter.get("negative", 0))
    observed_count = positive_count + negative_count
    rows.append(
      {
        "name": column,
        "total": total,
        "observed": observed_count,
        "positive": positive_count,
        "negative": negative_count,
        "missing": missing_count,
        "positive_rate": round(positive_count / float(observed_count), 6) if observed_count else 0.0,
      }
    )
  return rows


def build_label_lift_rows(
  *,
  stats_by_column: dict[str, dict[Any, LabelTokenStats]],
  column_groups: dict[str, str],
  baseline_positive_rate: float,
  min_support: int,
  top_k: int,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  epsilon = 0.5
  for column, token_stats in stats_by_column.items():
    group = column_groups.get(column, "feature")
    for token, stats in token_stats.items():
      if stats.support < min_support:
        continue
      positive_rate = stats.positive / float(stats.support) if stats.support else 0.0
      lift = positive_rate / baseline_positive_rate if baseline_positive_rate > 0 else 0.0
      log_odds = math.log((stats.positive + epsilon) / (stats.negative + epsilon))
      rows.append(
        {
          "group": group,
          "feature": column,
          "token": token,
          "support": stats.support,
          "positive": stats.positive,
          "negative": stats.negative,
          "positive_rate": round(positive_rate, 6),
          "baseline_positive_rate": round(baseline_positive_rate, 6),
          "lift": round(lift, 6),
          "log_odds": round(log_odds, 6),
        }
      )
  rows.sort(key=lambda item: (abs(item["lift"] - 1.0), item["support"]), reverse=True)
  return rows[:top_k]


def categorical_pair_candidates(
  cardinality_rows: list[dict[str, Any]],
  *,
  max_columns: int,
  max_cardinality: int,
) -> list[str]:
  candidates = [row for row in cardinality_rows if 1 < int(row["cardinality"]) <= max_cardinality]
  candidates.sort(key=lambda item: (int(item["cardinality"]), item["name"]))
  return [str(row["name"]) for row in candidates[:max_columns]]


def cramers_v_from_counts(counts: Counter[tuple[Any, Any]]) -> tuple[float, int, int, int]:
  if not counts:
    return 0.0, 0, 0, 0
  row_totals: Counter[Any] = Counter()
  column_totals: Counter[Any] = Counter()
  total = 0
  for (left, right), count in counts.items():
    row_totals[left] += count
    column_totals[right] += count
    total += count
  row_count = len(row_totals)
  column_count = len(column_totals)
  if total <= 0 or row_count <= 1 or column_count <= 1:
    return 0.0, total, row_count, column_count
  chi_square = 0.0
  for (left, right), observed in counts.items():
    expected = row_totals[left] * column_totals[right] / float(total)
    if expected > 0:
      chi_square += (observed - expected) * (observed - expected) / expected
  denominator = total * float(min(row_count - 1, column_count - 1))
  if denominator <= 0:
    return 0.0, total, row_count, column_count
  return math.sqrt(chi_square / denominator), total, row_count, column_count


def build_categorical_pair_rows(samples: list[dict[str, Any]], columns: list[str], *, top_k: int) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for left_index, left in enumerate(columns):
    for right in columns[left_index + 1 :]:
      counts: Counter[tuple[Any, Any]] = Counter()
      for sample in samples:
        left_value = sample.get(left)
        right_value = sample.get(right)
        if left_value is None or right_value is None:
          continue
        counts[(left_value, right_value)] += 1
      value, sample_rows, left_cardinality, right_cardinality = cramers_v_from_counts(counts)
      if sample_rows == 0:
        continue
      rows.append(
        {
          "left": left,
          "right": right,
          "cramers_v": round(value, 6),
          "sample_rows": sample_rows,
          "left_cardinality": left_cardinality,
          "right_cardinality": right_cardinality,
        }
      )
  rows.sort(key=lambda item: item["cramers_v"], reverse=True)
  return rows[:top_k]


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
        "q1": quantile(sorted_lengths, 0.25),
        "median": quantile(sorted_lengths, 0.5),
        "q3": quantile(sorted_lengths, 0.75),
        "max": float(stats.max_length),
        "mean": round(stats.total_length / float(stats.rows), 6),
        "p95": quantile(sorted_lengths, 0.95),
        "empty_rate": round(stats.empty_rows / float(stats.rows), 6),
      }
    )
  return rows


def build_dense_rows(dense_columns: list[str], dense_stats: dict[str, DenseStats]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for column in dense_columns:
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


def build_repeat_rows(layout: SchemaLayout, sequence_stats: dict[str, SequenceStats]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for domain in layout.sequence_domains:
    stats = sequence_stats.get(domain.name)
    if stats is None:
      continue
    value = 0.0 if stats.repeat_rows == 0 else stats.repeat_rate_sum / float(stats.repeat_rows)
    rows.append({"domain": domain.name, "repeat_rate": round(value, 6)})
  return rows


def build_user_activity_rows(activity_counter: Counter[Any]) -> list[dict[str, Any]]:
  if not activity_counter:
    return []
  bucket_counter: Counter[str] = Counter()
  for count in activity_counter.values():
    label = str(count) if count < 20 else "20+"
    bucket_counter[label] += 1
  numeric_labels = sorted(int(label) for label in bucket_counter if label != "20+")
  rows = [{"bucket": str(label), "user_count": bucket_counter[str(label)]} for label in numeric_labels]
  if "20+" in bucket_counter:
    rows.append({"bucket": "20+", "user_count": bucket_counter["20+"]})
  return rows


def build_overlap_rows(sampled_user_domains: dict[Any, set[str]], domain_names: list[str]) -> list[dict[str, Any]]:
  if not domain_names:
    return []
  users_by_domain: dict[str, set[Any]] = {domain: set() for domain in domain_names}
  for user_token, active_domains in sampled_user_domains.items():
    for domain in active_domains:
      users_by_domain[domain].add(user_token)
  rows: list[dict[str, Any]] = []
  for left in domain_names:
    for right in domain_names:
      union = users_by_domain[left] | users_by_domain[right]
      overlap = 0.0 if not union else len(users_by_domain[left] & users_by_domain[right]) / float(len(union))
      rows.append({"left": left, "right": right, "overlap": round(overlap, 6)})
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


def run_first_pass(
  dataset: DatasetInfo,
  layout: SchemaLayout,
  *,
  feature_columns: list[str],
  cardinality_columns: list[str],
  dense_columns: list[str],
  label_columns: list[str],
  label_feature_columns: list[str],
  user_key_column: str | None,
  row_limit: int | None,
  config: OnlineDatasetEDAConfig,
) -> FirstPassResult:
  null_counts = {column: 0 for column in feature_columns}
  cardinality_sketches = {column: KMVSketch(config.cardinality_sketch_k) for column in cardinality_columns}
  dense_stats = {column: DenseStats() for column in dense_columns}
  target_label_columns = [column for column in label_columns if column == "label_type"]
  label_counters = {column: Counter() for column in target_label_columns}
  label_null_stats = {column: LabelNullStats() for column in feature_columns}
  label_feature_sketches = {column: TopKTokenSketch(config.label_feature_candidate_k) for column in label_feature_columns}
  sequence_stats = {
    domain.name: SequenceStats(
      domain=domain.name,
      sampler=ReservoirSampler(config.sequence_sample_size, seed=stable_hash(domain.name) & 0xFFFFFFFF),
    )
    for domain in layout.sequence_domains
    if domain.length_column in dataset.available_columns
  }
  user_sampler = BottomKUserSampler(config.user_sample_limit)
  scanned_rows = 0
  progress = ProgressTracker(
    "first-pass",
    row_limit if row_limit is not None else dataset.total_rows,
    step_percent=config.progress_step_percent,
  )

  iter_columns = dedupe_preserve_order(feature_columns + target_label_columns + ([user_key_column] if user_key_column else []))
  progress.start()
  for batch in iter_batches(dataset, columns=iter_columns, max_rows=row_limit, batch_rows=config.batch_rows):
    scanned_rows += batch.num_rows
    cache: dict[str, list[Any]] = {}
    label_values = get_batch_values(batch, cache, "label_type") if "label_type" in target_label_columns else []
    label_flags = [binary_label(value) for value in label_values]

    for column in feature_columns:
      values = get_batch_values(batch, cache, column)
      null_counts[column] += sum(1 for value in values if is_missing(value))
      if column in cardinality_sketches:
        sketch = cardinality_sketches[column]
        for value in values:
          for token in sparse_tokens(value):
            sketch.add(token)
      elif column in dense_stats:
        stats = dense_stats[column]
        for raw_value in values:
          for item in normalize_list(raw_value):
            stats.add(float(item))

      if label_flags:
        null_stats = label_null_stats[column]
        for value, label_flag in zip(values, label_flags, strict=False):
          if label_flag is None:
            continue
          null_stats.add(is_positive=label_flag, missing=is_missing(value))

      label_feature_sketch = label_feature_sketches.get(column)
      if label_feature_sketch is not None and label_flags:
        for value, label_flag in zip(values, label_flags, strict=False):
          if label_flag is None:
            continue
          for token in row_sparse_tokens(value):
            label_feature_sketch.add(token)

    for column in target_label_columns:
      counter = label_counters[column]
      for value in get_batch_values(batch, cache, column):
        normalized = normalize_scalar(value)
        if normalized is None:
          counter["missing"] += 1
        elif normalized == 2:
          counter["positive"] += 1
        else:
          counter["negative"] += 1

    for domain in layout.sequence_domains:
      stats = sequence_stats.get(domain.name)
      if stats is None or domain.length_column is None:
        continue
      length_values = get_batch_values(batch, cache, domain.length_column)
      repeat_values = None
      if domain.repeat_column is not None and domain.repeat_column in batch.schema.names:
        repeat_values = get_batch_values(batch, cache, domain.repeat_column)
      for index, raw_length_value in enumerate(length_values):
        length = len(normalize_list(raw_length_value))
        stats.add_length(length)
        if repeat_values is None:
          continue
        tokens = [token for token in normalize_list(repeat_values[index]) if token not in (0, None)]
        if not tokens:
          continue
        stats.add_repeat_rate(1.0 - len(set(tokens)) / float(len(tokens)))

    if user_key_column is not None:
      for value in get_batch_values(batch, cache, user_key_column):
        token = hashable_value(value)
        if token is not None:
          user_sampler.consider(token)
    progress.update(scanned_rows)

  progress.finish(scanned_rows)

  return FirstPassResult(
    scanned_rows=scanned_rows,
    null_counts=null_counts,
    cardinality_sketches=cardinality_sketches,
    dense_stats=dense_stats,
    sequence_stats=sequence_stats,
    user_sampler=user_sampler,
    label_counters=label_counters,
    label_null_stats=label_null_stats,
    label_feature_candidates={column: sketch.tokens() for column, sketch in label_feature_sketches.items()},
  )


def run_second_pass(
  dataset: DatasetInfo,
  layout: SchemaLayout,
  *,
  user_key_column: str | None,
  co_missing_columns: list[str],
  sampled_users: set[Any],
  label_feature_candidates: dict[str, set[Any]],
  categorical_pair_columns: list[str],
  row_limit: int | None,
  config: OnlineDatasetEDAConfig,
) -> SecondPassResult:
  matrix_size = len(co_missing_columns)
  co_missing_counts = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
  sampled_user_activity: Counter[Any] = Counter()
  sampled_user_domains: dict[Any, set[str]] = defaultdict(set)
  label_feature_stats = {column: defaultdict(LabelTokenStats) for column in label_feature_candidates}
  categorical_sampler = RowReservoirSampler(config.categorical_pair_sample_rows, seed=0xC47E60)
  scanned_rows = 0

  domain_length_columns = [
    domain.length_column
    for domain in layout.sequence_domains
    if domain.length_column is not None and domain.length_column in dataset.available_columns
  ]
  iter_columns = dedupe_preserve_order(
    co_missing_columns
    + list(label_feature_candidates)
    + categorical_pair_columns
    + (["label_type"] if label_feature_candidates and "label_type" in dataset.available_columns else [])
    + ([user_key_column] if user_key_column else [])
    + domain_length_columns
  )
  progress = ProgressTracker(
    "second-pass",
    row_limit if row_limit is not None else dataset.total_rows,
    step_percent=config.progress_step_percent,
  )

  progress.start()
  for batch in iter_batches(dataset, columns=iter_columns, max_rows=row_limit, batch_rows=config.batch_rows):
    scanned_rows += batch.num_rows
    cache: dict[str, list[Any]] = {}
    missing_flags = {column: [is_missing(value) for value in get_batch_values(batch, cache, column)] for column in co_missing_columns}
    for left_index, left_column in enumerate(co_missing_columns):
      left_values = missing_flags[left_column]
      for right_index, right_column in enumerate(co_missing_columns):
        right_values = missing_flags[right_column]
        co_missing_counts[left_index][right_index] += sum(
          1
          for left_flag, right_flag in zip(left_values, right_values, strict=False)
          if left_flag and right_flag
        )

    if label_feature_candidates:
      label_values = get_batch_values(batch, cache, "label_type")
      label_flags = [binary_label(value) for value in label_values]
      for column, candidate_tokens in label_feature_candidates.items():
        if not candidate_tokens:
          continue
        values = get_batch_values(batch, cache, column)
        stats_by_token = label_feature_stats[column]
        for value, label_flag in zip(values, label_flags, strict=False):
          if label_flag is None:
            continue
          for token in row_sparse_tokens(value):
            if token in candidate_tokens:
              stats_by_token[token].add(is_positive=label_flag)

    if categorical_pair_columns:
      categorical_values = {column: get_batch_values(batch, cache, column) for column in categorical_pair_columns}
      for row_index in range(batch.num_rows):
        row = {
          column: row_categorical_token(values[row_index])
          for column, values in categorical_values.items()
        }
        categorical_sampler.add(row)

    if user_key_column is None or not sampled_users:
      progress.update(scanned_rows)
      continue

    user_values = get_batch_values(batch, cache, user_key_column)
    domain_presence = {
      domain.name: [len(normalize_list(value)) > 0 for value in get_batch_values(batch, cache, domain.length_column)]
      for domain in layout.sequence_domains
      if domain.length_column is not None and domain.length_column in batch.schema.names
    }
    for row_index, raw_user_value in enumerate(user_values):
      token = hashable_value(raw_user_value)
      if token not in sampled_users:
        continue
      sampled_user_activity[token] += 1
      for domain_name, flags in domain_presence.items():
        if flags[row_index]:
          sampled_user_domains[token].add(domain_name)

    progress.update(scanned_rows)

  progress.finish(scanned_rows)

  return SecondPassResult(
    co_missing_counts=co_missing_counts,
    sampled_user_activity=sampled_user_activity,
    sampled_user_domains=sampled_user_domains,
    label_feature_stats={column: dict(stats) for column, stats in label_feature_stats.items()},
    categorical_samples=categorical_sampler.samples,
  )


def build_report(
  dataset: DatasetInfo,
  schema_path: Path,
  row_limit: int | None,
  config: OnlineDatasetEDAConfig,
) -> dict[str, Any]:
  layout = SchemaLayout.from_path(schema_path)
  user_key_column = "user_id" if "user_id" in dataset.available_columns else layout.primary_user_id_column
  feature_columns = [column for column in layout.feature_columns if column in dataset.available_columns]
  sparse_columns = [column for column in layout.sparse_columns if column in dataset.available_columns]
  sequence_sideinfo_columns = [column for domain in layout.sequence_domains for column in domain.sideinfo_columns if column in dataset.available_columns]
  cardinality_columns = dedupe_preserve_order(sparse_columns + sequence_sideinfo_columns)
  sparse_column_groups = {column: "user_int" for column in layout.user_int_columns}
  sparse_column_groups.update({column: "item_int" for column in layout.item_int_columns})
  dense_columns = [column for column in layout.user_dense_columns if column in dataset.available_columns]
  label_columns = [name for name in ("label_type", "label_action_type") if name in dataset.available_columns]
  label_feature_columns = sparse_columns if "label_type" in label_columns else []

  first_pass = run_first_pass(
    dataset,
    layout,
    feature_columns=feature_columns,
    cardinality_columns=cardinality_columns,
    dense_columns=dense_columns,
    label_columns=label_columns,
    label_feature_columns=label_feature_columns,
    user_key_column=user_key_column,
    row_limit=row_limit,
    config=config,
  )

  row_count = first_pass.scanned_rows
  null_rows = build_null_rows(feature_columns, first_pass.null_counts, row_count)
  cardinality_rows = build_cardinality_rows(cardinality_columns, first_pass.cardinality_sketches)
  scalar_cardinality_rows = [row for row in cardinality_rows if row["name"] in set(sparse_columns)]
  sequence_cardinality_rows = [row for row in cardinality_rows if row["name"] in set(sequence_sideinfo_columns)]
  label_distribution_rows = build_label_distribution_rows([column for column in label_columns if column == "label_type"], first_pass.label_counters)
  label_null_rows = build_null_by_label_rows(first_pass.label_null_stats) if label_distribution_rows else []
  sequence_rows = build_sequence_rows(layout, first_pass.sequence_stats)
  dense_rows = build_dense_rows(dense_columns, first_pass.dense_stats)
  repeat_rows = build_repeat_rows(layout, first_pass.sequence_stats)
  cardinality_bin_rows = cardinality_bins(cardinality_rows)
  co_missing_names = [row["name"] for row in null_rows[:12] if row["null_rate"] > 0.0]
  categorical_pair_columns = categorical_pair_candidates(
    scalar_cardinality_rows,
    max_columns=config.categorical_pair_max_columns,
    max_cardinality=config.categorical_pair_max_cardinality,
  )

  second_pass = run_second_pass(
    dataset,
    layout,
    user_key_column=user_key_column,
    co_missing_columns=co_missing_names,
    sampled_users=first_pass.user_sampler.tokens(),
    label_feature_candidates=first_pass.label_feature_candidates if label_distribution_rows else {},
    categorical_pair_columns=categorical_pair_columns,
    row_limit=row_limit,
    config=config,
  )

  activity_rows = build_user_activity_rows(second_pass.sampled_user_activity)
  overlap_domain_names = [domain.name for domain in layout.sequence_domains if domain.name in {key for values in second_pass.sampled_user_domains.values() for key in values}]
  if not overlap_domain_names:
    overlap_domain_names = [domain.name for domain in layout.sequence_domains if domain.length_column in dataset.available_columns]
  overlap_rows = build_overlap_rows(second_pass.sampled_user_domains, overlap_domain_names)
  coverage_rows = build_domain_coverage_rows(second_pass.sampled_user_domains, overlap_domain_names)
  if label_distribution_rows and label_distribution_rows[0]["observed"]:
    baseline_positive_rate = label_distribution_rows[0]["positive"] / float(label_distribution_rows[0]["observed"])
  else:
    baseline_positive_rate = 0.0
  user_label_feature_rows = build_label_lift_rows(
    stats_by_column={column: stats for column, stats in second_pass.label_feature_stats.items() if column in set(layout.user_int_columns)},
    column_groups=sparse_column_groups,
    baseline_positive_rate=float(baseline_positive_rate),
    min_support=config.label_feature_min_support,
    top_k=config.label_feature_top_k,
  )
  item_label_feature_rows = build_label_lift_rows(
    stats_by_column={column: stats for column, stats in second_pass.label_feature_stats.items() if column in set(layout.item_int_columns)},
    column_groups=sparse_column_groups,
    baseline_positive_rate=float(baseline_positive_rate),
    min_support=config.label_feature_min_support,
    top_k=config.label_feature_top_k,
  )
  label_feature_rows = user_label_feature_rows + item_label_feature_rows
  categorical_pair_rows = build_categorical_pair_rows(
    second_pass.categorical_samples,
    categorical_pair_columns,
    top_k=config.categorical_pair_top_k,
  )
  scalar_columns = sorted(column for column in dataset.available_columns if column not in set(layout.feature_columns))
  layout_counts = {
    "scalar": len(scalar_columns),
    "user_int": len(layout.user_int_columns),
    "user_dense": len(layout.user_dense_columns),
    "item_int": len(layout.item_int_columns),
    "sequence": len(layout.sequence_columns),
  }
  domain_counts = {domain.name: len(domain.all_columns) for domain in layout.sequence_domains}
  chart_manifest: dict[str, str] = {}

  report = {
    "report": "dataset_eda",
    "dataset_path": str(dataset.dataset_path),
    "schema_path": str(schema_path),
    "dataset_role": "online",
    "streaming": True,
    "batch_rows": config.batch_rows,
    "label_columns": label_columns,
    "label_dependent_analyses_enabled": bool(label_distribution_rows),
    "row_count": row_count,
    "total_rows": dataset.total_rows,
    "sampled": row_count != dataset.total_rows,
    "max_rows": row_limit,
    "sample_percent": config.sample_percent,
    "chart_dir": None,
    "generated_charts": chart_manifest,
    "skipped_charts": SKIPPED_CHARTS,
    "approximation": {
      "cardinality": {"method": "kmv", "k": config.cardinality_sketch_k},
      "user_activity": {"method": "bottom_k_users", "limit": config.user_sample_limit},
      "cross_domain_overlap": {"method": "bottom_k_users", "limit": config.user_sample_limit},
      "sequence_quantiles": {"method": "reservoir_sample", "size": config.sequence_sample_size},
    },
    "stats": {
      "column_layout": {"counts": layout_counts, "domain_counts": domain_counts, "scalar_columns": scalar_columns},
      "null_rates": null_rows,
      "cardinality": cardinality_rows,
      "scalar_cardinality": scalar_cardinality_rows,
      "sequence_token_cardinality": sequence_cardinality_rows,
      "sequence_lengths": sequence_rows,
      "user_activity": activity_rows,
      "cross_domain_overlap": overlap_rows,
      "cross_domain_coverage": coverage_rows,
      "dense_distributions": dense_rows,
      "cardinality_bins": cardinality_bin_rows,
      "seq_repeat_rate": repeat_rows,
      "co_missing": {"columns": co_missing_names},
      "feature_auc": [],
      "null_rate_by_label": label_null_rows,
      "label_feature_lift": label_feature_rows,
      "user_feature_label_lift": user_label_feature_rows,
      "item_feature_label_lift": item_label_feature_rows,
      "categorical_pair_associations": categorical_pair_rows,
      "label_distribution": label_distribution_rows,
    },
  }
  return report


def print_report(report: dict[str, Any]) -> None:
  subtitle = base_subtitle(report["row_count"], report["total_rows"])
  stats = report["stats"]
  layout_counts = stats["column_layout"]["counts"]
  domain_counts = stats["column_layout"]["domain_counts"]

  print_section("Dataset")
  rows = [
    ("dataset_path", report["dataset_path"]),
    ("schema_path", report["schema_path"]),
    ("summary", subtitle),
    ("batch_rows", report["batch_rows"]),
    ("sampled", report["sampled"]),
  ]
  if report.get("sample_percent") is not None:
    rows.append(("sample_percent", report["sample_percent"]))
  if report.get("max_rows") is not None:
    rows.append(("max_rows", report["max_rows"]))
  print_key_values(rows)

  print_section("Column Layout")
  print_key_values(
    [
      ("scalar", layout_counts["scalar"]),
      ("user_int", layout_counts["user_int"]),
      ("user_dense", layout_counts["user_dense"]),
      ("item_int", layout_counts["item_int"]),
      ("sequence", layout_counts["sequence"]),
    ]
  )
  write_stdout_line("sequence_domains: " + ", ".join(f"{key}={value}" for key, value in domain_counts.items()))

  print_section("Label Distribution")
  print_label_distribution_rows(stats["label_distribution"])

  print_section("Null Rate By Label")
  print_ranked_rows(
    stats["null_rate_by_label"],
    name_key="name",
    value_keys=["positive_null_rate", "negative_null_rate", "delta", "positive_rows", "negative_rows"],
    limit=15,
  )

  print_section("Top Null Rates")
  print_ranked_rows(stats["null_rates"], name_key="name", value_keys=["null_rate"], limit=15)

  print_section("Top Cardinalities")
  print_ranked_rows(stats["cardinality"], name_key="name", value_keys=["cardinality"], limit=15)

  print_section("Top Sequence Token Cardinalities")
  print_ranked_rows(stats["sequence_token_cardinality"], name_key="name", value_keys=["cardinality"], limit=15)

  print_section("Top User Feature Label Lift")
  print_label_lift_rows(stats["user_feature_label_lift"])

  print_section("Top Item Feature Label Lift")
  print_label_lift_rows(stats["item_feature_label_lift"])

  print_section("Top Categorical Pair Associations")
  print_categorical_pair_rows(stats["categorical_pair_associations"])

  print_section("Sequence Length Summary")
  print_domain_rows(stats["sequence_lengths"])

  print_section("Sequence Repeat Rate")
  print_ranked_rows(stats["seq_repeat_rate"], name_key="domain", value_keys=["repeat_rate"], limit=10)

  print_section("Dense Feature Summary")
  print_ranked_rows(stats["dense_distributions"], name_key="name", value_keys=["mean", "variance", "std", "zero_frac"], limit=15)

  print_section("Cardinality Bins")
  print_ranked_rows(stats["cardinality_bins"], name_key="name", value_keys=["count"], limit=10)

  print_section("Sampled User Activity")
  print_ranked_rows(stats["user_activity"], name_key="bucket", value_keys=["user_count"], limit=20)

  print_section("Sampled Cross-Domain Coverage")
  print_coverage_rows(stats["cross_domain_coverage"])

  print_section("Sampled Cross-Domain Overlap")
  print_overlap_rows(stats["cross_domain_overlap"], [row["left"] for row in stats["cross_domain_overlap"] if row["left"] == row["right"]])

  print_section("Approximation")
  for key, value in report["approximation"].items():
    write_stdout_line(f"{key}: " + ", ".join(f"{inner_key}={inner_value}" for inner_key, inner_value in value.items()))


def run_online_dataset_eda(config: OnlineDatasetEDAConfig) -> dict[str, Any]:
  configure_logging()
  validate_config(config)
  dataset_path = config.dataset_path.expanduser().resolve()
  schema_path = resolve_schema_path(dataset_path, config.schema_path)
  dataset = build_dataset_info(dataset_path)
  effective_max_rows = resolve_scan_row_limit(dataset.total_rows, config.max_rows, config.sample_percent)
  logger.info("[online-eda] dataset={}", dataset_path)
  logger.info("[online-eda] schema={}", schema_path)
  logger.info("[online-eda] sink=stdout")
  if config.sample_percent is not None:
    logger.info(
      "[online-eda] scan=streaming sample_percent={:.1f} max_rows={} batch_rows={}",
      config.sample_percent,
      effective_max_rows,
      config.batch_rows,
    )
  elif effective_max_rows is None:
    logger.info("[online-eda] scan=streaming full batch_rows={}", config.batch_rows)
  else:
    logger.info("[online-eda] scan=streaming max_rows={} batch_rows={}", effective_max_rows, config.batch_rows)
  report = build_report(dataset, schema_path, effective_max_rows, config)
  print_report(report)
  write_stdout_line("skipped: " + ", ".join(report["skipped_charts"]))
  return report

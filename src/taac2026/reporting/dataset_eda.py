from __future__ import annotations

"""Reusable dataset analysis primitives for TAAC 2026.

This module provides composable functions that inspect a loaded dataset and
return plain-data summary dicts.  ECharts option generators accept those
summaries and produce JSON-serialisable dicts for interactive visualisation.
Both layers are designed for use by CLI scripts *and* notebook cells.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from taac2026.domain.config import DEFAULT_SEQUENCE_NAMES

# ---------------------------------------------------------------------------
# Constants – column naming conventions mirrored from tests/support.py
# ---------------------------------------------------------------------------

_USER_INT_PREFIX = "user_int_feats_"
_USER_DENSE_PREFIX = "user_dense_feats_"
_ITEM_INT_PREFIX = "item_int_feats_"
_DOMAIN_SEQ_PREFIXES: dict[str, str] = {
    d: f"{d}_seq_" for d in DEFAULT_SEQUENCE_NAMES
}

# ---------------------------------------------------------------------------
# 1. Schema summary
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ColumnGroups:
    """Column names classified into semantic groups."""

    scalar: list[str] = field(default_factory=list)
    user_int: list[str] = field(default_factory=list)
    user_dense: list[str] = field(default_factory=list)
    item_int: list[str] = field(default_factory=list)
    domain_seq: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        domain_count = sum(len(v) for v in self.domain_seq.values())
        return len(self.scalar) + len(self.user_int) + len(self.user_dense) + len(self.item_int) + domain_count


def classify_columns(column_names: Sequence[str]) -> ColumnGroups:
    """Split dataset column names into semantic groups."""
    groups = ColumnGroups()
    for col in column_names:
        if col.startswith(_USER_INT_PREFIX):
            groups.user_int.append(col)
        elif col.startswith(_USER_DENSE_PREFIX):
            groups.user_dense.append(col)
        elif col.startswith(_ITEM_INT_PREFIX):
            groups.item_int.append(col)
        else:
            matched = False
            for domain, prefix in _DOMAIN_SEQ_PREFIXES.items():
                if col.startswith(prefix):
                    groups.domain_seq.setdefault(domain, []).append(col)
                    matched = True
                    break
            if not matched:
                groups.scalar.append(col)
    for key in groups.domain_seq:
        groups.domain_seq[key].sort()
    groups.user_int.sort()
    groups.user_dense.sort()
    groups.item_int.sort()
    groups.scalar.sort()
    return groups


# ---------------------------------------------------------------------------
# 2. Per-column statistics (streaming, single-pass)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ColumnStats:
    name: str
    count: int = 0
    non_null: int = 0
    n_unique: int = 0
    is_list: bool = False
    min_val: float | None = None
    max_val: float | None = None
    sum_val: float = 0.0
    sum_len: int = 0  # for list columns

    @property
    def null_rate(self) -> float:
        return 1.0 - self.non_null / self.count if self.count else 0.0

    @property
    def mean_val(self) -> float | None:
        return self.sum_val / self.non_null if self.non_null else None

    @property
    def mean_list_len(self) -> float | None:
        return self.sum_len / self.non_null if self.non_null and self.is_list else None


def compute_column_stats(
    rows: Iterable[dict[str, Any]],
    columns: Sequence[str] | None = None,
    *,
    max_unique_track: int = 200_000,
) -> dict[str, ColumnStats]:
    """Compute per-column statistics in a single streaming pass."""
    accumulators: dict[str, ColumnStats] = {}
    unique_sets: dict[str, set[Any]] = {}

    for row in rows:
        if columns is None:
            row_items = row.items()
        else:
            row_items = ((col, row.get(col)) for col in columns)

        for col, value in row_items:
            if col not in accumulators:
                accumulators[col] = ColumnStats(name=col)
                unique_sets[col] = set()
            stats = accumulators[col]
            stats.count += 1
            if value is None:
                continue
            stats.non_null += 1

            if isinstance(value, (list, tuple)):
                stats.is_list = True
                stats.sum_len += len(value)
            else:
                if len(unique_sets[col]) < max_unique_track:
                    unique_sets[col].add(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if math.isfinite(value):
                        stats.sum_val += value
                        if stats.min_val is None or value < stats.min_val:
                            stats.min_val = value
                        if stats.max_val is None or value > stats.max_val:
                            stats.max_val = value

    for col, uniques in unique_sets.items():
        accumulators[col].n_unique = len(uniques)

    return accumulators


# ---------------------------------------------------------------------------
# 3. Label / action-type distribution
# ---------------------------------------------------------------------------

_LABEL_NAMES: dict[int, str] = {0: "曝光", 1: "点击", 2: "转化"}


@dataclass(slots=True)
class LabelDistribution:
    counts: Counter[int] = field(default_factory=Counter)
    total: int = 0

    def add(self, label: int | None) -> None:
        self.total += 1
        if label is not None:
            self.counts[label] = self.counts.get(label, 0) + 1

    def as_table(self) -> list[dict[str, Any]]:
        rows = []
        for label in sorted(self.counts):
            count = self.counts[label]
            rows.append({
                "label_type": label,
                "name": _LABEL_NAMES.get(label, f"unknown_{label}"),
                "count": count,
                "ratio": count / self.total if self.total else 0.0,
            })
        return rows


def compute_label_distribution(rows: Iterable[dict[str, Any]], label_col: str = "label_type") -> LabelDistribution:
    dist = LabelDistribution()
    for row in rows:
        dist.add(row.get(label_col))
    return dist


# ---------------------------------------------------------------------------
# 4. Sequence-length analysis
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SequenceLengthStats:
    domain: str
    lengths: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.lengths)

    def summary(self) -> dict[str, Any]:
        if not self.lengths:
            return {"domain": self.domain, "count": 0}
        arr = np.array(self.lengths)
        return {
            "domain": self.domain,
            "count": len(arr),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "zero_rate": float((arr == 0).mean()),
        }


def compute_sequence_lengths(
    rows: Iterable[dict[str, Any]],
    domain_prefixes: dict[str, str] | None = None,
) -> dict[str, SequenceLengthStats]:
    prefixes = domain_prefixes or _DOMAIN_SEQ_PREFIXES
    result: dict[str, SequenceLengthStats] = {d: SequenceLengthStats(domain=d) for d in prefixes}
    _probe_cols: dict[str, str | None] = {d: None for d in prefixes}

    for row in rows:
        for domain, prefix in prefixes.items():
            probe = _probe_cols[domain]
            if probe is None:
                for col in row:
                    if col.startswith(prefix):
                        _probe_cols[domain] = col
                        probe = col
                        break
            if probe is None:
                result[domain].lengths.append(0)
                continue
            seq = row.get(probe)
            if seq is None or not isinstance(seq, (list, tuple)):
                result[domain].lengths.append(0)
            else:
                result[domain].lengths.append(len(seq))

    return result


# ---------------------------------------------------------------------------
# 5. Feature cardinality ranking
# ---------------------------------------------------------------------------

def compute_cardinality_ranking(
    stats: dict[str, ColumnStats],
    groups: ColumnGroups,
) -> list[dict[str, Any]]:
    """Rank sparse features by unique-value count (descending)."""
    sparse_cols = groups.user_int + groups.item_int
    ranking = []
    for col in sparse_cols:
        st = stats.get(col)
        if st is None:
            continue
        ranking.append({
            "column": col,
            "n_unique": st.n_unique,
            "coverage": 1.0 - st.null_rate,
            "group": "user" if col.startswith(_USER_INT_PREFIX) else "item",
        })
    ranking.sort(key=lambda r: r["n_unique"], reverse=True)
    return ranking


# ---------------------------------------------------------------------------
# 6. ECharts interactive option generators
# ---------------------------------------------------------------------------

_EC_COLORS = ["#89b4fa", "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7", "#f9e2af"]


def echarts_label_distribution(dist: LabelDistribution) -> dict[str, Any]:
    """ECharts option for label-type pie chart."""
    table = dist.as_table()
    return {
        "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
        "legend": {"bottom": 0},
        "series": [{
            "type": "pie",
            "radius": ["35%", "65%"],
            "avoidLabelOverlap": True,
            "itemStyle": {"borderRadius": 6, "borderWidth": 2},
            "label": {"formatter": "{b}\n{d}%"},
            "data": [
                {"name": r["name"], "value": r["count"]}
                for r in table
            ],
            "color": _EC_COLORS,
        }],
    }


def echarts_cardinality(ranking: list[dict[str, Any]], *, top_n: int = 25) -> dict[str, Any]:
    """ECharts option for horizontal bar chart of sparse-feature cardinalities."""
    data = list(reversed(ranking[:top_n]))
    names = [
        r["column"].replace("user_int_feats_", "u_").replace("item_int_feats_", "i_")
        for r in data
    ]
    values = [r["n_unique"] for r in data]
    colors = [_EC_COLORS[0] if r["group"] == "user" else _EC_COLORS[2] for r in data]
    return {
        "_height": "600px",
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 90, "right": 30, "top": 40, "bottom": 30},
        "xAxis": {"type": "log", "name": "基数 (unique values)"},
        "yAxis": {"type": "category", "data": names, "axisLabel": {"fontSize": 10}},
        "series": [{
            "type": "bar",
            "data": [{"value": v, "itemStyle": {"color": c}} for v, c in zip(values, colors)],
            "label": {"show": True, "position": "right", "formatter": "{c}"},
        }],
    }


def echarts_sequence_lengths(seq_stats: dict[str, SequenceLengthStats]) -> dict[str, Any]:
    """ECharts option for per-domain sequence-length box-style chart."""
    domains = [d for d in seq_stats if seq_stats[d].lengths]
    summaries = [seq_stats[d].summary() for d in domains]
    return {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": domains},
        "grid": {"left": 60, "right": 30, "top": 60, "bottom": 40},
        "xAxis": {"type": "category", "data": ["min", "P25", "median", "mean", "P75", "P95", "max"]},
        "yAxis": {"type": "value", "name": "序列长度"},
        "series": [
            {
                "name": d,
                "type": "line",
                "smooth": True,
                "symbol": "circle",
                "symbolSize": 8,
                "data": [
                    s["min"], int(s["p25"]), int(s["median"]),
                    int(s["mean"]), int(s["p75"]), int(s["p95"]), s["max"],
                ],
                "itemStyle": {"color": _EC_COLORS[i % len(_EC_COLORS)]},
                "areaStyle": {"opacity": 0.08},
            }
            for i, (d, s) in enumerate(zip(domains, summaries))
            if s["count"] > 0
        ],
    }


def echarts_coverage_heatmap(
    stats: dict[str, ColumnStats], groups: ColumnGroups,
) -> dict[str, Any]:
    """ECharts option for sparse-feature coverage heatmap."""
    cols = groups.user_int + groups.item_int
    names = [c.replace("user_int_feats_", "u_").replace("item_int_feats_", "i_") for c in cols]
    coverages = [round(1.0 - stats[c].null_rate, 3) if c in stats else 0.0 for c in cols]
    data = [[i, 0, coverages[i]] for i in range(len(cols))]
    return {
        "_height": "180px",
        "tooltip": {},
        "grid": {"left": 60, "right": 30, "top": 10, "bottom": 60},
        "xAxis": {"type": "category", "data": names, "axisLabel": {"rotate": 60, "fontSize": 7}},
        "yAxis": {"type": "category", "data": ["coverage"], "show": False},
        "visualMap": {
            "min": 0, "max": 1, "show": True, "orient": "horizontal",
            "left": "center", "bottom": 0,
            "inRange": {"color": ["#f38ba8", "#f9e2af", "#a6e3a1"]},
            "text": ["100%", "0%"],
        },
        "series": [{
            "type": "heatmap",
            "data": data,
            "label": {"show": len(cols) <= 30},
        }],
    }


def echarts_ndcg_decay(k_max: int = 10) -> dict[str, Any]:
    """ECharts option for NDCG@K position-discount curve."""
    ks = list(range(1, k_max + 1))
    gains = [round(1.0 / math.log2(k + 1), 4) for k in ks]
    return {
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 60, "right": 30, "top": 50, "bottom": 40},
        "xAxis": {"type": "category", "data": [str(k) for k in ks], "name": "排名位置 k"},
        "yAxis": {"type": "value", "name": "1/log₂(k+1)", "max": 1.15},
        "series": [{
            "type": "line",
            "data": gains,
            "smooth": False,
            "symbol": "circle",
            "symbolSize": 10,
            "label": {"show": True, "position": "top", "fontSize": 9},
            "areaStyle": {"opacity": 0.1},
            "itemStyle": {"color": _EC_COLORS[0]},
        }],
    }


def echarts_cross_edition() -> dict[str, Any]:
    """ECharts option for cross-edition label distribution comparison."""
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["曝光", "点击", "转化"]},
        "grid": {"left": 60, "right": 30, "top": 50, "bottom": 30},
        "xAxis": {"type": "category", "data": ["上届 1M", "上届 10M", "本届 sample"]},
        "yAxis": {"type": "value", "name": "占比 (%)", "max": 100},
        "series": [
            {"name": "曝光", "type": "bar", "data": [90.19, 94.63, 0], "color": _EC_COLORS[0]},
            {"name": "点击", "type": "bar", "data": [9.81, 2.85, 87.6], "color": _EC_COLORS[1]},
            {"name": "转化", "type": "bar", "data": [0, 2.52, 12.4], "color": _EC_COLORS[2]},
        ],
    }


def echarts_column_layout(groups: ColumnGroups) -> dict[str, Any]:
    """ECharts option for column-group donut chart."""
    pieces = [
        ("标量列", len(groups.scalar)),
        ("user_int", len(groups.user_int)),
        ("user_dense", len(groups.user_dense)),
        ("item_int", len(groups.item_int)),
    ]
    for domain, cols in sorted(groups.domain_seq.items()):
        pieces.append((f"{domain}_seq", len(cols)))
    return {
        "tooltip": {"trigger": "item", "formatter": "{b}: {c} 列 ({d}%)"},
        "legend": {"bottom": 0, "type": "scroll"},
        "series": [{
            "type": "pie",
            "radius": ["35%", "65%"],
            "avoidLabelOverlap": True,
            "itemStyle": {"borderRadius": 6, "borderWidth": 2},
            "label": {"formatter": "{b}\n{c} ({d}%)"},
            "data": [{"name": n, "value": v} for n, v in pieces],
            "color": _EC_COLORS,
        }],
    }


def echarts_null_rates(stats: dict[str, ColumnStats], *, top_n: int = 30) -> dict[str, Any]:
    """ECharts option for horizontal bar chart of columns with highest null rates."""
    items = [(s.name, round(s.null_rate, 4)) for s in stats.values() if s.null_rate > 0]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:top_n]
    names = [n for n, _ in reversed(items)]
    rates = [r for _, r in reversed(items)]
    return {
        "_height": f"{max(300, len(items) * 22)}px",
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 120, "right": 40, "top": 30, "bottom": 30},
        "xAxis": {"type": "value", "name": "缺失率", "max": 1.0},
        "yAxis": {"type": "category", "data": names, "axisLabel": {"fontSize": 8}},
        "series": [{
            "type": "bar",
            "data": rates,
            "itemStyle": {"color": _EC_COLORS[1]},
            "label": {"show": True, "position": "right", "formatter": "{c}"},
            "markLine": {
                "silent": True,
                "data": [{"xAxis": 0.5, "lineStyle": {"color": _EC_COLORS[4], "type": "dashed"}}],
            },
        }],
    }


def echarts_edition_comparison() -> dict[str, Any]:
    """ECharts option for cross-edition dataset dimension comparison."""
    metrics = ["用户特征数", "物品特征数", "行为域数", "总列数", "序列最大长度", "序列均值(主域)"]
    taac2025 = [8, 12, 1, 20, 100, 94]
    taac2026 = [56, 14, 4, 120, 3951, 1099]
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["TAAC 2025", "TAAC 2026"]},
        "grid": {"left": 100, "right": 40, "top": 50, "bottom": 30},
        "xAxis": {"type": "log", "name": "数值 (log)"},
        "yAxis": {"type": "category", "data": metrics, "axisLabel": {"fontSize": 9}},
        "series": [
            {
                "name": "TAAC 2025",
                "type": "bar",
                "data": taac2025,
                "itemStyle": {"color": _EC_COLORS[4]},
                "label": {"show": True, "position": "right", "formatter": "{c}"},
            },
            {
                "name": "TAAC 2026",
                "type": "bar",
                "data": taac2026,
                "itemStyle": {"color": _EC_COLORS[0]},
                "label": {"show": True, "position": "right", "formatter": "{c}"},
            },
        ],
    }


def echarts_seq_length_summary(seq_stats: dict[str, SequenceLengthStats]) -> dict[str, Any]:
    """ECharts option for per-domain sequence-length box-style summary."""
    domains = [d for d in seq_stats if seq_stats[d].lengths]
    summaries = [seq_stats[d].summary() for d in domains]
    if not domains or not summaries:
        return {
            "tooltip": {},
            "legend": {"data": []},
            "radar": {"indicator": []},
            "series": [{"type": "radar", "data": []}],
        }
    # Radar chart for multi-domain comparison
    radar_max = max(s["max"] for s in summaries) * 1.1
    indicators = [
        {"name": "min", "max": radar_max},
        {"name": "P25", "max": radar_max},
        {"name": "median", "max": radar_max},
        {"name": "mean", "max": radar_max},
        {"name": "P75", "max": radar_max},
        {"name": "P95", "max": radar_max},
        {"name": "max", "max": radar_max},
    ]
    radar_data = []
    for i, (d, s) in enumerate(zip(domains, summaries)):
        if s["count"] == 0:
            continue
        radar_data.append({
            "name": d,
            "value": [s["min"], int(s["p25"]), int(s["median"]),
                       int(s["mean"]), int(s["p75"]), int(s["p95"]), s["max"]],
            "lineStyle": {"color": _EC_COLORS[i % len(_EC_COLORS)]},
            "itemStyle": {"color": _EC_COLORS[i % len(_EC_COLORS)]},
            "areaStyle": {"opacity": 0.1},
        })
    return {
        "tooltip": {},
        "legend": {"data": [d for d, s in zip(domains, summaries) if s["count"] > 0]},
        "radar": {"indicator": indicators},
        "series": [{
            "type": "radar",
            "data": radar_data,
        }],
    }


def serialize_echarts(option: dict[str, Any]) -> str:
    """Serialize ECharts option dict to JSON string."""
    import json as _json

    return _json.dumps(option, indent=2, ensure_ascii=False)

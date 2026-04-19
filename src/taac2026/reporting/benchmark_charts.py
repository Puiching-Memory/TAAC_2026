from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable

from taac2026.infrastructure.io.files import write_json


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "assets" / "figures" / "benchmarks"
DEFAULT_PERFORMANCE_DIR = REPO_ROOT / "outputs" / "performance"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "benchmark_acceptance.json"

COMPONENT_ORDER = ["collate", "embedding", "attention", "ffn", "rmsnorm"]
COMPONENT_LABELS = {
    "collate": "collate",
    "embedding": "embedding",
    "attention": "attention",
    "ffn": "ffn",
    "rmsnorm": "rmsnorm",
}
ACCEPTANCE_THRESHOLDS = {
    "embedding_throughput_min_gain": 2.0,
    "attention_latency_max_ratio": 0.7,
}
PHASE_NAME_PATTERN = re.compile(r"^phase-(\d+)$", re.IGNORECASE)


def serialize_echarts(option: dict[str, Any]) -> str:
    return json.dumps(option, indent=2, ensure_ascii=False)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_component(name: str) -> str:
    lowered = name.lower()
    if "e2e" in lowered or "train_step" in lowered:
        return "e2e_train_step"
    if "infer" in lowered:
        return "inference"
    if "quant" in lowered:
        return "quantization"
    if "embed" in lowered:
        return "embedding"
    if "attn" in lowered or "attention" in lowered:
        return "attention"
    if "ffn" in lowered or "feedforward" in lowered:
        return "ffn"
    if "norm" in lowered:
        return "rmsnorm"
    if "collate" in lowered or "loader" in lowered:
        return "collate"
    return lowered.replace(" ", "_")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _normalize_record(record: dict[str, Any], *, default_label: str = "") -> dict[str, Any]:
    name = str(record.get("name") or record.get("fullname") or default_label or "benchmark")
    component = str(record.get("component") or _infer_component(name))
    label = str(record.get("label") or record.get("commit") or default_label or component)
    phase = str(record.get("phase") or label)
    times_ms = [
        float(value)
        for value in record.get("times_ms", [])
        if isinstance(value, (int, float))
    ]
    median_ms = _safe_float(record.get("median_ms"))
    if median_ms is None and times_ms:
        median_ms = _percentile(times_ms, 50)
    if median_ms is None:
        median_ms = 0.0
    mean_ms = _safe_float(record.get("mean_ms"))
    if mean_ms is None and times_ms:
        mean_ms = sum(times_ms) / len(times_ms)
    if mean_ms is None:
        mean_ms = median_ms
    iqr_ms = _safe_float(record.get("iqr_ms"))
    if iqr_ms is None and times_ms:
        iqr_ms = _percentile(times_ms, 75) - _percentile(times_ms, 25)
    if iqr_ms is None:
        iqr_ms = 0.0

    return {
        "name": name,
        "component": component,
        "phase": phase,
        "label": label,
        "model": str(record.get("model") or label),
        "metric": str(record.get("metric") or "latency"),
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "iqr_ms": iqr_ms,
        "p50_ms": _safe_float(record.get("p50_ms")) or median_ms,
        "p95_ms": _safe_float(record.get("p95_ms")),
        "p99_ms": _safe_float(record.get("p99_ms")),
        "throughput": _safe_float(record.get("throughput") or record.get("ops_per_second")),
        "memory_mb": _safe_float(record.get("memory_mb")),
        "times_ms": times_ms,
    }


def load_pytest_benchmark_file(path: str | Path, *, default_label: str = "") -> list[dict[str, Any]]:
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        return []

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    commit_info = payload.get("commit_info") or {}
    context = payload.get("context") or {}
    fallback_label = default_label or str(commit_info.get("id") or context.get("benchmark_phase") or benchmark_path.stem)

    records: list[dict[str, Any]] = []
    for benchmark in payload.get("benchmarks", []):
        stats = benchmark.get("stats") or {}
        extra_info = benchmark.get("extra_info") or {}
        record = _normalize_record(
            {
                "name": extra_info.get("name") or benchmark.get("name") or benchmark.get("fullname") or benchmark_path.stem,
                "component": extra_info.get("component"),
                "phase": extra_info.get("phase") or fallback_label,
                "label": extra_info.get("label") or fallback_label,
                "model": extra_info.get("model") or fallback_label,
                "metric": extra_info.get("metric") or "latency",
                "median_ms": _safe_float(stats.get("median")) * 1e3 if _safe_float(stats.get("median")) is not None else None,
                "mean_ms": _safe_float(stats.get("mean")) * 1e3 if _safe_float(stats.get("mean")) is not None else None,
                "iqr_ms": _safe_float(stats.get("iqr")) * 1e3 if _safe_float(stats.get("iqr")) is not None else None,
                "throughput": extra_info.get("throughput") or stats.get("ops"),
                "memory_mb": extra_info.get("memory_mb"),
                "times_ms": [
                    float(value) * 1e3
                    for value in stats.get("data", [])
                    if isinstance(value, (int, float))
                ],
            },
            default_label=fallback_label,
        )
        records.append(record)
    return records


def load_pytest_benchmark_files(paths: Iterable[str | Path], *, default_label: str = "") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_pytest_benchmark_file(path, default_label=default_label))
    return records


def load_performance_records(performance_dir: str | Path = DEFAULT_PERFORMANCE_DIR) -> list[dict[str, Any]]:
    root = Path(performance_dir)
    if not root.exists():
        return []

    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            source_records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            source_records = payload["records"]
        else:
            source_records = [payload]
        for record in source_records:
            if not isinstance(record, dict):
                continue
            records.append(_normalize_record(record, default_label=path.stem))
    return records


def load_benchmark_records(
    pytest_json_paths: Iterable[str | Path] = (),
    *,
    performance_dir: str | Path = DEFAULT_PERFORMANCE_DIR,
    default_label: str = "",
) -> list[dict[str, Any]]:
    records = load_pytest_benchmark_files(pytest_json_paths, default_label=default_label)
    records.extend(load_performance_records(performance_dir))
    return records


def _select_candidate_phase(phases: list[str], baseline_phase: str, candidate_phase: str | None) -> str | None:
    if candidate_phase is not None:
        return candidate_phase
    return _select_latest_phase([phase for phase in phases if phase != baseline_phase], baseline_phase)


def _phase_sort_key(phase: str, baseline_phase: str) -> tuple[int, int, str]:
    if phase == baseline_phase:
        return (0, -1, phase)
    match = PHASE_NAME_PATTERN.match(phase)
    if match is not None:
        return (2, int(match.group(1)), phase)
    return (1, 0, phase)


def _select_latest_phase(phases: Iterable[str], baseline_phase: str) -> str | None:
    ordered = _dedupe(phase for phase in phases if phase != baseline_phase)
    if not ordered:
        return None
    return max(ordered, key=lambda phase: _phase_sort_key(phase, baseline_phase))


def _select_component_candidate_phase(
    phases: list[str],
    values_by_phase: dict[str, dict[str, float]],
    component: str,
    *,
    baseline_phase: str,
    candidate_phase: str | None,
) -> str | None:
    if candidate_phase is not None:
        if component in values_by_phase.get(candidate_phase, {}):
            return candidate_phase
        return None
    return _select_latest_phase(
        (phase for phase in phases if component in values_by_phase.get(phase, {})),
        baseline_phase,
    )


def _acceptance_check(
    *,
    baseline_value: float | None,
    candidate_value: float | None,
    candidate_phase: str | None,
    threshold: float,
    higher_is_better: bool,
) -> dict[str, Any]:
    if baseline_value is None or candidate_value is None:
        return {
            "status": "not_enough_data",
            "baseline_value": baseline_value,
            "candidate_value": candidate_value,
            "candidate_phase": candidate_phase,
            "threshold": threshold,
            "ratio": None,
        }

    ratio = candidate_value / max(baseline_value, 1.0e-12)
    passed = ratio >= threshold if higher_is_better else ratio <= threshold
    return {
        "status": "pass" if passed else "fail",
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "candidate_phase": candidate_phase,
        "threshold": threshold,
        "ratio": ratio,
    }


def build_benchmark_acceptance_summary(
    records: list[dict[str, Any]],
    *,
    baseline_phase: str = "baseline",
    candidate_phase: str | None = None,
) -> dict[str, Any]:
    normalized = [_normalize_record(record) for record in records]
    phases = _dedupe(record["phase"] for record in normalized)

    latency_by_phase: dict[str, dict[str, float]] = {}
    throughput_by_phase: dict[str, dict[str, float]] = {}
    for record in normalized:
        if record["component"] in COMPONENT_ORDER and record["median_ms"] > 0:
            latency_by_phase.setdefault(record["phase"], {})[record["component"]] = record["median_ms"]
        if record["throughput"] is not None and record["throughput"] > 0:
            throughput_by_phase.setdefault(record["phase"], {})[record["component"]] = float(record["throughput"])

    quantization_records = [record for record in normalized if record["component"] == "quantization"]
    inference_records = [record for record in normalized if record["component"] == "inference"]

    embedding_candidate_phase = _select_component_candidate_phase(
        phases,
        throughput_by_phase,
        "embedding",
        baseline_phase=baseline_phase,
        candidate_phase=candidate_phase,
    )
    attention_candidate_phase = _select_component_candidate_phase(
        phases,
        latency_by_phase,
        "attention",
        baseline_phase=baseline_phase,
        candidate_phase=candidate_phase,
    )
    candidate_phases = {
        "embedding": embedding_candidate_phase,
        "attention": attention_candidate_phase,
    }
    resolved_candidate_phase = candidate_phase
    if resolved_candidate_phase is None:
        unique_candidate_phases = _dedupe(phase for phase in candidate_phases.values() if phase is not None)
        resolved_candidate_phase = unique_candidate_phases[0] if len(unique_candidate_phases) == 1 else None

    embedding_check = _acceptance_check(
        baseline_value=throughput_by_phase.get(baseline_phase, {}).get("embedding"),
        candidate_value=None if embedding_candidate_phase is None else throughput_by_phase.get(embedding_candidate_phase, {}).get("embedding"),
        candidate_phase=embedding_candidate_phase,
        threshold=ACCEPTANCE_THRESHOLDS["embedding_throughput_min_gain"],
        higher_is_better=True,
    )
    attention_check = _acceptance_check(
        baseline_value=latency_by_phase.get(baseline_phase, {}).get("attention"),
        candidate_value=None if attention_candidate_phase is None else latency_by_phase.get(attention_candidate_phase, {}).get("attention"),
        candidate_phase=attention_candidate_phase,
        threshold=ACCEPTANCE_THRESHOLDS["attention_latency_max_ratio"],
        higher_is_better=False,
    )

    return {
        "baseline_phase": baseline_phase,
        "candidate_phase": resolved_candidate_phase,
        "candidate_phases": candidate_phases,
        "phases": phases,
        "component_latency_ms": latency_by_phase,
        "component_throughput": throughput_by_phase,
        "inference_models": [
            {
                "model": record["model"],
                "phase": record["phase"],
                "median_ms": record["median_ms"],
                "p95_ms": record["p95_ms"],
                "throughput": record["throughput"],
            }
            for record in inference_records
        ],
        "quantization_models": [
            {
                "model": record["model"],
                "phase": record["phase"],
                "median_ms": record["median_ms"],
                "memory_mb": record["memory_mb"],
            }
            for record in quantization_records
        ],
        "acceptance": {
            "embedding_throughput_vs_baseline": embedding_check,
            "attention_latency_vs_baseline": attention_check,
            "int8_quantization_record_present": {
                "status": "pass" if bool(quantization_records) else "missing",
                "model_count": len(quantization_records),
            },
        },
    }


def write_benchmark_summary(
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    *,
    records: list[dict[str, Any]] | None = None,
    baseline_phase: str = "baseline",
    candidate_phase: str | None = None,
) -> Path:
    destination = Path(summary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        destination,
        build_benchmark_acceptance_summary(
            records or [],
            baseline_phase=baseline_phase,
            candidate_phase=candidate_phase,
        ),
    )
    return destination


def _placeholder_chart(title: str, y_axis_name: str) -> dict[str, Any]:
    return {
        "title": {"text": title, "left": "center"},
        "grid": {"left": 60, "right": 30, "top": 80, "bottom": 40, "outerBoundsMode": "none"},
        "graphic": [{
            "type": "text",
            "left": "center",
            "top": "middle",
            "style": {"text": "No benchmark data available", "fontSize": 16},
        }],
        "xAxis": {"type": "category", "data": [], "nameMoveOverlap": False},
        "yAxis": {"type": "value", "name": y_axis_name, "nameMoveOverlap": False},
        "series": [],
    }


def echarts_component_latency(records: list[dict[str, Any]]) -> dict[str, Any]:
    component_records = [
        record
        for record in records
        if record["component"] in COMPONENT_ORDER and record["median_ms"] > 0
    ]
    if not component_records:
        return _placeholder_chart("各组件延迟对比", "延迟 (ms)")

    phases = _dedupe(record["phase"] for record in component_records)
    components = [component for component in COMPONENT_ORDER if any(record["component"] == component for record in component_records)]
    return {
        "title": {"text": "各组件延迟对比", "left": "center"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"top": 32, "data": phases},
        "grid": {"left": 70, "right": 30, "top": 90, "bottom": 40, "outerBoundsMode": "none"},
        "xAxis": {
            "type": "category",
            "data": [COMPONENT_LABELS[component] for component in components],
            "nameMoveOverlap": False,
        },
        "yAxis": {"type": "value", "name": "延迟 (ms)", "nameMoveOverlap": False},
        "series": [
            {
                "name": phase,
                "type": "bar",
                "data": [
                    next(
                        (
                            record["median_ms"]
                            for record in component_records
                            if record["component"] == component and record["phase"] == phase
                        ),
                        None,
                    )
                    for component in components
                ],
            }
            for phase in phases
        ],
    }


def echarts_throughput_trend(records: list[dict[str, Any]]) -> dict[str, Any]:
    throughput_records = [record for record in records if record["throughput"] is not None and record["throughput"] > 0]
    if not throughput_records:
        return _placeholder_chart("Embedding 吞吐趋势", "lookups/sec")

    labels = _dedupe(record["label"] for record in throughput_records)
    components = _dedupe(record["component"] for record in throughput_records)
    if "embedding" in components:
        components = ["embedding"] + [component for component in components if component != "embedding"]
    return {
        "title": {"text": "Embedding 吞吐趋势", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 32, "data": components},
        "grid": {"left": 80, "right": 30, "top": 90, "bottom": 40, "outerBoundsMode": "none"},
        "xAxis": {"type": "category", "data": labels, "nameMoveOverlap": False},
        "yAxis": {"type": "value", "name": "lookups/sec", "nameMoveOverlap": False},
        "series": [
            {
                "name": component,
                "type": "line",
                "smooth": True,
                "data": [
                    next(
                        (
                            record["throughput"]
                            for record in throughput_records
                            if record["component"] == component and record["label"] == label
                        ),
                        None,
                    )
                    for label in labels
                ],
            }
            for component in components
        ],
    }


def echarts_e2e_train_step(records: list[dict[str, Any]]) -> dict[str, Any]:
    e2e_records = [record for record in records if record["component"] == "e2e_train_step" and record["median_ms"] > 0]
    if not e2e_records:
        return _placeholder_chart("端到端训练步延迟", "延迟 (ms)")

    labels = _dedupe(record["label"] for record in e2e_records)
    return {
        "title": {"text": "端到端训练步延迟", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 80, "right": 30, "top": 70, "bottom": 40, "outerBoundsMode": "none"},
        "xAxis": {"type": "category", "data": labels, "nameMoveOverlap": False},
        "yAxis": {"type": "value", "name": "ms/step", "nameMoveOverlap": False},
        "series": [{
            "name": "e2e_train_step",
            "type": "line",
            "smooth": True,
            "data": [next(record["median_ms"] for record in e2e_records if record["label"] == label) for label in labels],
        }],
    }


def _boxplot_summary(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        min(values),
        _percentile(values, 25),
        _percentile(values, 50),
        _percentile(values, 75),
        max(values),
    ]


def echarts_inference_boxplot(records: list[dict[str, Any]]) -> dict[str, Any]:
    inference_records = [record for record in records if record["component"] == "inference"]
    if not inference_records:
        return _placeholder_chart("推理延迟分布", "延迟 (ms)")

    labels = _dedupe(record["model"] for record in inference_records)
    series_data = []
    for label in labels:
        values: list[float] = []
        for record in inference_records:
            if record["model"] != label:
                continue
            if record["times_ms"]:
                values.extend(record["times_ms"])
            elif record["median_ms"] > 0:
                values.append(record["median_ms"])
        series_data.append(_boxplot_summary(values))

    return {
        "title": {"text": "推理延迟分布", "left": "center"},
        "tooltip": {"trigger": "item"},
        "grid": {"left": 80, "right": 30, "top": 70, "bottom": 40, "outerBoundsMode": "none"},
        "xAxis": {"type": "category", "data": labels, "nameMoveOverlap": False},
        "yAxis": {"type": "value", "name": "延迟 (ms)", "nameMoveOverlap": False},
        "series": [{
            "name": "inference",
            "type": "boxplot",
            "data": series_data,
        }],
    }


def echarts_quantization_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    quant_records = [record for record in records if record["component"] == "quantization"]
    if not quant_records:
        return _placeholder_chart("量化前后对比", "延迟 (ms)")

    models = _dedupe(record["model"] for record in quant_records)
    return {
        "title": {"text": "量化前后对比", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 32, "data": ["Latency (ms)", "Memory (MB)"]},
        "grid": {"left": 80, "right": 80, "top": 90, "bottom": 40, "outerBoundsMode": "none"},
        "xAxis": {"type": "category", "data": models, "nameMoveOverlap": False},
        "yAxis": [
            {"type": "value", "name": "延迟 (ms)", "nameMoveOverlap": False},
            {"type": "value", "name": "显存 (MB)", "nameMoveOverlap": False},
        ],
        "series": [
            {
                "name": "Latency (ms)",
                "type": "bar",
                "data": [
                    next(
                        (
                            record["median_ms"]
                            for record in quant_records
                            if record["model"] == model
                        ),
                        None,
                    )
                    for model in models
                ],
            },
            {
                "name": "Memory (MB)",
                "type": "line",
                "yAxisIndex": 1,
                "smooth": True,
                "data": [
                    next(
                        (
                            record["memory_mb"]
                            for record in quant_records
                            if record["model"] == model
                        ),
                        None,
                    )
                    for model in models
                ],
            },
        ],
    }


def build_benchmark_charts(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized = [_normalize_record(record) for record in records]
    return {
        "component_latency.echarts.json": echarts_component_latency(normalized),
        "throughput_trend.echarts.json": echarts_throughput_trend(normalized),
        "e2e_train_step.echarts.json": echarts_e2e_train_step(normalized),
        "inference_boxplot.echarts.json": echarts_inference_boxplot(normalized),
        "quantization_comparison.echarts.json": echarts_quantization_comparison(normalized),
    }


def write_benchmark_charts(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    records: list[dict[str, Any]] | None = None,
) -> list[Path]:
    chart_dir = Path(output_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    rendered = build_benchmark_charts(records or [])
    written: list[Path] = []
    for filename, option in rendered.items():
        destination = chart_dir / filename
        destination.write_text(serialize_echarts(option), encoding="utf-8")
        written.append(destination)
    return written


__all__ = [
    "ACCEPTANCE_THRESHOLDS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PERFORMANCE_DIR",
    "DEFAULT_SUMMARY_PATH",
    "build_benchmark_acceptance_summary",
    "build_benchmark_charts",
    "echarts_component_latency",
    "echarts_e2e_train_step",
    "echarts_inference_boxplot",
    "echarts_quantization_comparison",
    "echarts_throughput_trend",
    "load_benchmark_records",
    "load_performance_records",
    "load_pytest_benchmark_file",
    "load_pytest_benchmark_files",
    "serialize_echarts",
    "write_benchmark_charts",
    "write_benchmark_summary",
]
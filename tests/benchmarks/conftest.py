from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from torch.utils.benchmark import Timer

from tests.support import TestWorkspace, create_test_workspace


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERFORMANCE_DIR = REPO_ROOT / "outputs" / "performance"


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


@pytest.fixture
def benchmark_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def benchmark_workspace(tmp_path_factory: pytest.TempPathFactory) -> TestWorkspace:
    return create_test_workspace(tmp_path_factory.mktemp("bench_workspace"))


@pytest.fixture
def cuda_timer(benchmark_device: torch.device):
    def _time(target: Callable[[], Any], *, min_run_time: float = 0.2) -> dict[str, Any]:
        def wrapped() -> Any:
            value = target()
            if benchmark_device.type == "cuda":
                torch.cuda.synchronize(benchmark_device)
            return value

        measurement = Timer(stmt="wrapped()", globals={"wrapped": wrapped}).blocked_autorange(min_run_time=min_run_time)
        times_ms = [float(sample) * 1e3 for sample in measurement.times]
        return {
            "median_ms": float(measurement.median * 1e3),
            "mean_ms": float(measurement.mean * 1e3),
            "iqr_ms": float(measurement.iqr * 1e3),
            "p50_ms": _percentile(times_ms, 50),
            "p95_ms": _percentile(times_ms, 95),
            "p99_ms": _percentile(times_ms, 99),
            "times_ms": times_ms,
        }

    return _time


@pytest.fixture
def performance_recorder():
    DEFAULT_PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)

    def _record(name: str, payload: dict[str, Any]) -> Path:
        destination = DEFAULT_PERFORMANCE_DIR / f"{name}.json"
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return destination

    return _record

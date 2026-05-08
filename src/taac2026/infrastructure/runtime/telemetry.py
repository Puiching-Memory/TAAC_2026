"""Lightweight runtime telemetry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import resource
import sys
import time
from typing import Any

import torch


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _cuda_device(device: str | torch.device | None) -> torch.device | None:
    if device is None or not torch.cuda.is_available():
        return None
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    return resolved


@dataclass(slots=True)
class RuntimeTelemetry:
    label: str
    device: str | torch.device | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _started_at_unix: float = 0.0
    _started_at_perf: float = 0.0
    _cuda_device: torch.device | None = None

    def start(self) -> RuntimeTelemetry:
        self._started_at_unix = time.time()
        self._started_at_perf = time.perf_counter()
        self._cuda_device = _cuda_device(self.device)
        if self._cuda_device is not None:
            try:
                torch.cuda.reset_peak_memory_stats(self._cuda_device)
            except RuntimeError:
                self._cuda_device = None
        return self

    def finish(self, **extra: Any) -> dict[str, Any]:
        ended_at_unix = time.time()
        elapsed_sec = time.perf_counter() - self._started_at_perf if self._started_at_perf else 0.0
        payload: dict[str, Any] = {
            "label": self.label,
            "started_at_unix": self._started_at_unix,
            "ended_at_unix": ended_at_unix,
            "elapsed_sec": elapsed_sec,
            "cpu_peak_rss_mb": _peak_rss_mb(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self._cuda_device is not None:
            payload.update(
                {
                    "cuda_device": str(self._cuda_device),
                    "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(self._cuda_device) / (1024.0 * 1024.0),
                    "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved(self._cuda_device) / (1024.0 * 1024.0),
                }
            )
        else:
            payload.update(
                {
                    "cuda_device": None,
                    "cuda_peak_allocated_mb": 0.0,
                    "cuda_peak_reserved_mb": 0.0,
                }
            )
        payload.update(extra)
        _add_rate(payload, "rows", "rows_per_sec")
        _add_rate(payload, "batches", "batches_per_sec")
        _add_rate(payload, "steps", "steps_per_sec")
        return payload


def _add_rate(payload: dict[str, Any], count_key: str, rate_key: str) -> None:
    count = payload.get(count_key)
    elapsed = float(payload.get("elapsed_sec") or 0.0)
    if count is None or elapsed <= 0.0:
        return
    try:
        payload[rate_key] = float(count) / elapsed
    except (TypeError, ValueError):
        return


def file_size_mb(path: str | Path | None) -> float | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        return None
    return resolved.stat().st_size / (1024.0 * 1024.0)


__all__ = ["RuntimeTelemetry", "file_size_mb"]

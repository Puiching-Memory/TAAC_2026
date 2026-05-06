from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from taac2026.infrastructure.logging import logger


def _normalize_log_level(level: int | str) -> int | str:
    if isinstance(level, str):
        return level.upper()
    resolved = logging.getLevelName(level)
    return resolved if isinstance(resolved, str) else level


@dataclass(frozen=True, slots=True)
class CapturedLogRecord:
    message: str
    level: str

    def getMessage(self) -> str:
        return self.message


class LoguruCapture:
    def __init__(self) -> None:
        self.records: list[CapturedLogRecord] = []

    @property
    def text(self) -> str:
        return "\n".join(record.message for record in self.records)

    def _sink(self, message) -> None:
        record = message.record
        self.records.append(
            CapturedLogRecord(
                message=record["message"],
                level=record["level"].name,
            )
        )

    @contextmanager
    def at_level(self, level: int | str):
        self.records.clear()
        sink_id = logger.add(self._sink, level=_normalize_log_level(level), format="{message}")
        try:
            yield self
        finally:
            logger.remove(sink_id)


@pytest.fixture
def log_capture() -> LoguruCapture:
    return LoguruCapture()


def pytest_collection_modifyitems(config, items):
    del config
    for item in items:
        path = Path(str(item.fspath))
        parts = path.parts
        if "unit" in parts:
            item.add_marker("unit")
        elif "integration" in parts:
            item.add_marker("integration")

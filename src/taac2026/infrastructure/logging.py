"""Shared loguru configuration with tqdm-safe console output."""

from __future__ import annotations

from datetime import timedelta
import logging
from pathlib import Path
import sys
import time
from typing import Any

from loguru import logger
from tqdm import tqdm


_configured_sink_ids: list[int] = []
_elapsed_origin = time.time()


def _reset_elapsed_origin() -> None:
    global _elapsed_origin
    _elapsed_origin = time.time()


def _record_prefix(record: dict[str, Any]) -> str:
    elapsed_seconds = max(0, round(record["time"].timestamp() - _elapsed_origin))
    wall_clock = record["time"].astimezone().strftime("%x %X")
    return f"{wall_clock} - {timedelta(seconds=elapsed_seconds)}"


def _format_record(record: dict[str, Any]) -> str:
    prefix = _record_prefix(record)
    formatted_message = record["message"].replace("\n", "\n" + " " * (len(prefix) + 3))
    record["extra"]["taac_prefix"] = prefix
    record["extra"]["taac_message"] = formatted_message
    return "{extra[taac_prefix]} - {extra[taac_message]}\n{exception}"


def _console_sink(message: Any) -> None:
    tqdm.write(str(message).rstrip("\n"), file=sys.stderr)


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _configure_stdlib_logging() -> None:
    logging.captureWarnings(True)
    root_logger = logging.getLogger()
    root_logger.handlers = [_InterceptHandler()]
    root_logger.setLevel(logging.DEBUG)
    root_logger.propagate = False


def configure_logging(
    log_path: str | Path | None = None,
    *,
    console_level: str | int = "INFO",
    file_level: str | int = "DEBUG",
) -> None:
    """Configure loguru sinks and route stdlib logging through them."""

    global _configured_sink_ids

    for sink_id in _configured_sink_ids:
        logger.remove(sink_id)
    _configured_sink_ids = []

    _reset_elapsed_origin()
    _configure_stdlib_logging()

    _configured_sink_ids.append(
        logger.add(
            _console_sink,
            colorize=False,
            diagnose=False,
            backtrace=False,
            format=_format_record,
            level=console_level,
        )
    )

    if log_path is not None:
        resolved_log_path = Path(log_path).expanduser().resolve()
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        _configured_sink_ids.append(
            logger.add(
                resolved_log_path,
                mode="w",
                colorize=False,
                diagnose=False,
                backtrace=False,
                format=_format_record,
                level=file_level,
            )
        )


def reset_logging_timer() -> None:
    _reset_elapsed_origin()


__all__ = ["configure_logging", "logger", "reset_logging_timer"]
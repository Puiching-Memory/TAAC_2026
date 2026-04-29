"""Project JSON helpers backed by orjson."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson


def _dump_option(*, indent: int | None = None) -> int:
    if indent is None:
        return 0
    if indent == 2:
        return orjson.OPT_INDENT_2
    raise ValueError(f"unsupported JSON indent: {indent}")


def dump_bytes(
    payload: Any,
    *,
    indent: int | None = None,
    trailing_newline: bool = False,
) -> bytes:
    encoded = orjson.dumps(payload, option=_dump_option(indent=indent))
    if trailing_newline:
        encoded += b"\n"
    return encoded


def dumps(
    payload: Any,
    *,
    indent: int | None = None,
    trailing_newline: bool = False,
) -> str:
    return dump_bytes(payload, indent=indent, trailing_newline=trailing_newline).decode("utf-8")


def loads(payload: str | bytes | bytearray | memoryview) -> Any:
    return orjson.loads(payload)


def load(handle: Any) -> Any:
    return loads(handle.read())


def write_path(
    path: Path,
    payload: Any,
    *,
    indent: int | None = None,
    trailing_newline: bool = False,
) -> Path:
    path.write_bytes(dump_bytes(payload, indent=indent, trailing_newline=trailing_newline))
    return path


def read_path(path: Path) -> Any:
    return loads(path.read_bytes())
"""Filesystem helpers used by CLIs and package builders."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from taac2026.infrastructure.io.json_utils import read_path, write_path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def stable_hash64(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) or 1


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> Path:
    ensure_parent(path)
    return write_path(path, payload, indent=2, trailing_newline=True)


def read_json(path: Path) -> Any:
    return read_path(path)

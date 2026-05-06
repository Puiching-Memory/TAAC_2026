"""Shared stdout/stderr helpers to avoid ad hoc print usage."""

from __future__ import annotations

import sys
from typing import TextIO


def _write(stream: TextIO, text: str, *, end: str = "", flush: bool = False) -> None:
    stream.write(f"{text}{end}")
    if flush:
        stream.flush()


def write_stdout(text: str, *, end: str = "", flush: bool = False) -> None:
    _write(sys.stdout, text, end=end, flush=flush)


def write_stdout_line(text: str = "", *, flush: bool = False) -> None:
    _write(sys.stdout, text, end="\n", flush=flush)


def write_stderr(text: str, *, end: str = "", flush: bool = False) -> None:
    _write(sys.stderr, text, end=end, flush=flush)


def write_stderr_line(text: str = "", *, flush: bool = False) -> None:
    _write(sys.stderr, text, end="\n", flush=flush)


__all__ = ["write_stderr", "write_stderr_line", "write_stdout", "write_stdout_line"]
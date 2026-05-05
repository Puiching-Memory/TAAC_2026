"""Shared placeholder reporting commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from taac2026.infrastructure.io.json_utils import write_path


def _write_placeholder_report(
    argv: Sequence[str] | None,
    *,
    description: str,
    default_output: str,
    report_name: str,
    include_inputs: bool = False,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    if include_inputs:
        parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output", default=default_output)
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"report": report_name, "status": "placeholder"}
    if include_inputs:
        payload["input"] = list(args.input)
    write_path(output, payload, indent=2, trailing_newline=True)
    print(output)
    return 0


def benchmark_report_main(argv: Sequence[str] | None = None) -> int:
    return _write_placeholder_report(
        argv,
        description="Write a minimal benchmark report placeholder",
        default_output="outputs/reports/benchmark_report.json",
        report_name="benchmark",
        include_inputs=True,
    )


def tech_timeline_main(argv: Sequence[str] | None = None) -> int:
    return _write_placeholder_report(
        argv,
        description="Write a minimal technology timeline placeholder",
        default_output="outputs/reports/tech_timeline.json",
        report_name="tech_timeline",
    )


__all__ = ["benchmark_report_main", "tech_timeline_main"]
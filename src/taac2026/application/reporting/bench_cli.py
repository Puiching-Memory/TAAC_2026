"""Benchmark reporting command placeholder."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from taac2026.infrastructure.io.json_utils import write_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a minimal benchmark report placeholder")
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output", default="outputs/reports/benchmark_report.json")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"report": "benchmark", "input": args.input, "status": "placeholder"}
    write_path(output, payload, indent=2, trailing_newline=True)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

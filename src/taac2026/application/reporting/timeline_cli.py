"""CLI entry point for the technology-timeline generator.

Usage::

    uv run taac-tech-timeline
    uv run taac-tech-timeline --expand 1 --min-citations 200
    uv run taac-tech-timeline --api-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from taac2026.reporting.tech_timeline import (
    build_graph,
    to_echarts,
    write_echarts_json,
)

DEFAULT_OUTPUT = Path("docs/assets/figures/papers/tech-timeline.echarts.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fetch recommendation-system papers from Semantic Scholar "
            "and generate an ECharts timeline graph."
        ),
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output path for ECharts JSON (default: %(default)s)",
    )
    p.add_argument(
        "--expand",
        type=int,
        default=0,
        choices=[0, 1],
        help="Citation expansion depth: 0 = seeds only, 1 = one hop (default: 0)",
    )
    p.add_argument(
        "--min-citations",
        type=int,
        default=100,
        help="Minimum citation count for expanded papers (default: %(default)s)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Semantic Scholar API key (optional, increases rate limit)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)

    def _log(msg: str) -> None:
        print(msg, file=sys.stderr)

    graph = build_graph(
        expand_depth=args.expand,
        min_citations=args.min_citations,
        api_key=args.api_key,
        progress_callback=_log,
    )

    option = to_echarts(graph)
    write_echarts_json(option, output)
    _log(f"Written → {output}")
    return 0

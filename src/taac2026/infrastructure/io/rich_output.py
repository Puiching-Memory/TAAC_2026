"""Rich-formatted terminal output helpers for CLI commands."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def print_rich_summary(
    title: str,
    fields: Sequence[tuple[str, str]],
    *,
    sections: Sequence[tuple[str, Sequence[tuple[str, str]]]] = (),
    subtitle: str | None = None,
    border_style: str = "green",
) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1), expand=False)
    table.add_column(style="bold", min_width=14)
    table.add_column()

    for key, value in fields:
        table.add_row(key, value)
    for section_title, section_fields in sections:
        table.add_row(f"[bold]{section_title}[/]", "")
        for key, value in section_fields:
            table.add_row(f"  {key}", value)

    panel = Panel.fit(
        table,
        title=f"[bold]{title}[/]",
        subtitle=subtitle,
        border_style=border_style,
    )
    Console().print(panel)


__all__ = ["print_rich_summary"]

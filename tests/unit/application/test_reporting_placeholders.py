from __future__ import annotations

from pathlib import Path

from taac2026.application.reporting.placeholders import benchmark_report_main, tech_timeline_main
from taac2026.infrastructure.io.json_utils import loads


def test_benchmark_report_main_writes_placeholder_payload(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "reports" / "benchmark.json"

    exit_code = benchmark_report_main(
        [
            "--input",
            "inputs/one.json",
            "--input",
            "inputs/two.json",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == str(output)
    assert loads(output.read_bytes()) == {
        "report": "benchmark",
        "input": ["inputs/one.json", "inputs/two.json"],
        "status": "placeholder",
    }


def test_tech_timeline_main_writes_placeholder_payload(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "reports" / "timeline.json"

    exit_code = tech_timeline_main(["--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == str(output)
    assert loads(output.read_bytes()) == {
        "report": "tech_timeline",
        "status": "placeholder",
    }
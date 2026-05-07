from __future__ import annotations

from pathlib import Path

from taac2026.infrastructure import logging as taac_logging


def test_configure_logging_removes_loguru_default_console_sink(monkeypatch, capsys) -> None:
    console_lines: list[str] = []
    monkeypatch.setattr(taac_logging, "_console_sink", lambda message: console_lines.append(str(message)))

    taac_logging.configure_logging()
    taac_logging.logger.info("single console event")

    captured = capsys.readouterr()
    assert "single console event" not in captured.err
    assert len(console_lines) == 1
    assert "single console event" in console_lines[0]


def test_bundle_mode_console_logs_message_without_taac_prefix(monkeypatch, tmp_path: Path) -> None:
    console_lines: list[str] = []
    monkeypatch.setenv("TAAC_BUNDLE_MODE", "1")
    monkeypatch.setattr(taac_logging, "_console_sink", lambda message: console_lines.append(str(message)))

    log_path = tmp_path / "train.log"
    taac_logging.configure_logging(log_path)
    taac_logging.logger.info("Validation progress {}", "184/461")

    assert console_lines == ["Validation progress 184/461\n"]
    file_log = log_path.read_text(encoding="utf-8")
    assert "Validation progress 184/461" in file_log
    assert " - Validation progress 184/461" in file_log


def test_local_console_logs_keep_taac_prefix(monkeypatch) -> None:
    console_lines: list[str] = []
    monkeypatch.delenv("TAAC_BUNDLE_MODE", raising=False)
    monkeypatch.setattr(taac_logging, "_console_sink", lambda message: console_lines.append(str(message)))

    taac_logging.configure_logging()
    taac_logging.logger.info("local progress")

    assert len(console_lines) == 1
    assert " - local progress" in console_lines[0]

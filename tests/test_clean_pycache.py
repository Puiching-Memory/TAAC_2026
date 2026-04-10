from __future__ import annotations

from pathlib import Path

import pytest

from taac2026.application.maintenance.clean_pycache import clean_pycache, find_pycache_dirs, main

pytestmark = pytest.mark.unit


def _write_cache_file(path: Path, *, size: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_find_pycache_dirs_skips_environment_directories_by_default(tmp_path: Path) -> None:
    cache_dir = tmp_path / "src" / "taac2026" / "__pycache__"
    venv_cache = tmp_path / ".venv" / "lib" / "python3.12" / "__pycache__"
    _write_cache_file(cache_dir / "clean_pycache.cpython-312.pyc", size=11)
    _write_cache_file(venv_cache / "site.cpython-312.pyc")

    matches = find_pycache_dirs(tmp_path)

    assert matches == [cache_dir]
    assert cache_dir.exists()
    assert venv_cache.exists()


def test_clean_pycache_removes_detected_directories(tmp_path: Path) -> None:
    cache_dir = tmp_path / "src" / "taac2026" / "__pycache__"
    nested_cache = tmp_path / "tests" / "unit" / "__pycache__"
    _write_cache_file(cache_dir / "a.pyc")
    _write_cache_file(nested_cache / "b.pyc")

    result = clean_pycache(tmp_path)

    assert result.root == tmp_path.resolve()
    assert cache_dir in result.matched_dirs
    assert nested_cache in result.matched_dirs
    assert result.matched_files == 2
    assert result.total_bytes == 6
    assert result.failures == []
    assert not cache_dir.exists()
    assert not nested_cache.exists()


def test_main_can_include_environment_directories(tmp_path: Path, capsys) -> None:
    venv_cache = tmp_path / ".venv" / "lib" / "python3.12" / "__pycache__"
    _write_cache_file(venv_cache / "site.cpython-312.pyc")

    exit_code = main(["--root", str(tmp_path), "--dry-run", "--include-env-dirs"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert str(venv_cache) in captured.out
    assert "include_env_dirs=True" in captured.out

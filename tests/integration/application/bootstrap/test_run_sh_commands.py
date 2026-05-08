from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

from tests.support.paths import locate_repo_root


REPO_ROOT = locate_repo_root(Path(__file__))
RUN_SH_PATH = REPO_ROOT / "run.sh"


def _prepare_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    run_script_path = project_dir / "run.sh"
    shutil.copy2(RUN_SH_PATH, run_script_path)
    run_script_path.chmod(run_script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    (project_dir / "pyproject.toml").write_text(
        "[project]\nname = \"stub\"\nversion = \"0.0.0\"\n",
        encoding="utf-8",
    )
    return project_dir


def test_run_sh_rejects_package_infer_subcommand(tmp_path: Path) -> None:
    project_dir = _prepare_project(tmp_path)

    result = subprocess.run(
        ["bash", "run.sh", "package-infer", "--help"],
        cwd=project_dir,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "unknown command: package-infer\n"


def test_run_sh_rejects_package_subcommand(tmp_path: Path) -> None:
    project_dir = _prepare_project(tmp_path)

    result = subprocess.run(
        ["bash", "run.sh", "package", "--help"],
        cwd=project_dir,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "unknown command: package\n"

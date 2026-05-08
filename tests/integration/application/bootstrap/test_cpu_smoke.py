from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

from taac2026.infrastructure.io.json import loads
from tests.support.env import clean_subprocess_env
from tests.support.paths import locate_repo_root


REPO_ROOT = locate_repo_root(Path(__file__))
RUN_SH_PATH = REPO_ROOT / "run.sh"


def _write_smoke_experiment(package_dir: Path) -> None:
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "from taac2026.domain.experiment import ExperimentSpec\n"
        "\n"
        "\n"
        "def _train(request):\n"
        "    request.run_dir.mkdir(parents=True, exist_ok=True)\n"
        "    marker = request.run_dir / 'smoke_train.txt'\n"
        "    marker.write_text('ok', encoding='utf-8')\n"
        "    return {'experiment': request.experiment, 'run_dir': str(request.run_dir), 'marker': str(marker)}\n"
        "\n"
        "\n"
        "def _evaluate(request):\n"
        "    return {'experiment': request.experiment, 'run_dir': str(request.run_dir)}\n"
        "\n"
        "\n"
        "def _infer(request):\n"
        "    return {'experiment': request.experiment, 'result_dir': str(request.result_dir)}\n"
        "\n"
        "\n"
        "EXPERIMENT = ExperimentSpec(\n"
        "    name='smoke_experiment',\n"
        "    package_dir=Path(__file__).resolve().parent,\n"
        "    train_fn=_train,\n"
        "    evaluate_fn=_evaluate,\n"
        "    infer_fn=_infer,\n"
        "    metadata={'requires_dataset': False},\n"
        ")\n",
        encoding="utf-8",
    )


def test_run_sh_train_cpu_smoke_uses_repository_runtime(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    run_script = project_dir / "run.sh"
    run_script.write_bytes(RUN_SH_PATH.read_bytes())
    run_script.chmod(run_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (project_dir / "pyproject.toml").write_text(
        "[project]\nname = 'taac-smoke-project'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )
    experiment_dir = project_dir / "experiments" / "smoke"
    _write_smoke_experiment(experiment_dir)
    run_dir = tmp_path / "outputs" / "smoke"

    env = clean_subprocess_env(
        {
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}{os.environ.get('PYTHONPATH', '') and ':' + os.environ['PYTHONPATH']}",
            "TAAC_RUNNER": "python",
            "TAAC_SKIP_PIP_INSTALL": "1",
        },
        include_platform_paths=True,
    )
    completed = subprocess.run(
        [
            "bash",
            "run.sh",
            "train",
            "--experiment",
            str(experiment_dir),
            "--run-dir",
            str(run_dir),
        ],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = loads(completed.stdout)

    assert payload["experiment"] == str(experiment_dir)
    assert payload["run_dir"] == str(run_dir)
    assert (run_dir / "smoke_train.txt").read_text(encoding="utf-8") == "ok"

from __future__ import annotations

import zipfile
from pathlib import Path

from tests.support.paths import locate_repo_root
from taac2026.infrastructure.io.json import dump_bytes, loads


REPO_ROOT = locate_repo_root(Path(__file__))

_MINIMAL_PYPROJECT = "[project]\nname = \"minimal\"\nversion = \"0.0.0\"\n"

_MINIMAL_TRAINING_CLI = (
    "from __future__ import annotations\n"
    "\n"
    "import orjson\n"
    "import os\n"
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def main() -> None:\n"
    "    print(orjson.dumps({\"cwd\": str(Path.cwd()), \"argv\": sys.argv[1:], "
    "\"experiment\": os.environ.get(\"TAAC_EXPERIMENT\")}).decode())\n"
    "\n"
    "\n"
    "if __name__ == \"__main__\":\n"
    "    main()\n"
)

_MINIMAL_EVALUATION_CLI = (
    "from __future__ import annotations\n"
    "\n"
    "import json\n"
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def main() -> None:\n"
    "    print(json.dumps({\"cwd\": str(Path.cwd()), \"argv\": sys.argv[1:]}))\n"
    "\n"
    "\n"
    "if __name__ == \"__main__\":\n"
    "    main()\n"
)

_MINIMAL_INFERENCE_ENTRYPOINT = (
    "from __future__ import annotations\n"
    "\n"
    "import orjson\n"
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def main() -> None:\n"
    "    print(orjson.dumps({\"cwd\": str(Path.cwd()), \"experiment\": os.environ.get(\"TAAC_EXPERIMENT\")}).decode())\n"
)


def code_package_names(code_package_path: Path) -> set[str]:
    with zipfile.ZipFile(code_package_path) as code_archive:
        return set(code_archive.namelist())


def code_package_manifest(code_package_path: Path, manifest_name: str) -> dict[str, object]:
    with zipfile.ZipFile(code_package_path) as code_archive:
        payload = code_archive.read(f"project/{manifest_name}")
    return loads(payload)


def write_platform_runtime(code_archive: zipfile.ZipFile) -> None:
    runtime_files = [
        *sorted(Path("src/taac2026/application/bootstrap").glob("*.py")),
        Path("src/taac2026/infrastructure/__init__.py"),
        *sorted(Path("src/taac2026/infrastructure/io").glob("*.py")),
        *sorted(Path("src/taac2026/infrastructure/platform").glob("*.py")),
    ]
    for relative_path in runtime_files:
        code_archive.write(REPO_ROOT / relative_path, f"project/{relative_path.as_posix()}")


def _write_minimal_runtime_package(
    code_package_path: Path,
    *,
    manifest_name: str,
    bundled_experiment_path: str | None,
    package_inits: tuple[str, ...],
    entrypoint_path: str,
    entrypoint_source: str,
) -> None:
    with zipfile.ZipFile(code_package_path, mode="w", compression=zipfile.ZIP_DEFLATED) as code_archive:
        manifest: dict[str, object] = {}
        if bundled_experiment_path is not None:
            manifest["bundled_experiment_path"] = bundled_experiment_path
        code_archive.writestr(
            f"project/{manifest_name}",
            dump_bytes(manifest, indent=2, trailing_newline=True),
        )
        code_archive.writestr("project/pyproject.toml", _MINIMAL_PYPROJECT)
        for package_init in (
            "project/src/taac2026/__init__.py",
            "project/src/taac2026/application/__init__.py",
            *package_inits,
        ):
            code_archive.writestr(package_init, "")
        write_platform_runtime(code_archive)
        code_archive.writestr(entrypoint_path, entrypoint_source)


def write_minimal_training_runtime_package(
    code_package_path: Path,
    *,
    bundled_experiment_path: str | None = "experiments/minimal",
) -> None:
    _write_minimal_runtime_package(
        code_package_path,
        manifest_name=".taac_training_manifest.json",
        bundled_experiment_path=bundled_experiment_path,
        package_inits=("project/src/taac2026/application/training/__init__.py",),
        entrypoint_path="project/src/taac2026/application/training/cli.py",
        entrypoint_source=_MINIMAL_TRAINING_CLI,
    )


def write_minimal_eval_runtime_package(
    code_package_path: Path,
    *,
    bundled_experiment_path: str | None = "experiments/minimal",
) -> None:
    _write_minimal_runtime_package(
        code_package_path,
        manifest_name=".taac_training_manifest.json",
        bundled_experiment_path=bundled_experiment_path,
        package_inits=("project/src/taac2026/application/evaluation/__init__.py",),
        entrypoint_path="project/src/taac2026/application/evaluation/cli.py",
        entrypoint_source=_MINIMAL_EVALUATION_CLI,
    )


def write_minimal_inference_runtime_package(
    code_package_path: Path,
    *,
    bundled_experiment_path: str | None = "experiments/minimal",
) -> None:
    _write_minimal_runtime_package(
        code_package_path,
        manifest_name=".taac_inference_manifest.json",
        bundled_experiment_path=bundled_experiment_path,
        package_inits=("project/src/taac2026/application/evaluation/__init__.py",),
        entrypoint_path="project/src/taac2026/application/evaluation/infer.py",
        entrypoint_source=_MINIMAL_INFERENCE_ENTRYPOINT,
    )


def write_fake_pip_package(root: Path, log_path: Path) -> Path:
    fake_pip = root / "fake_pip"
    pip_package = fake_pip / "pip"
    pip_package.mkdir(parents=True)
    (pip_package / "__init__.py").write_text("", encoding="utf-8")
    (pip_package / "__main__.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "import orjson\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        f"Path({str(log_path)!r}).write_bytes(orjson.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    return fake_pip


def assert_pip_install_args(pip_args: list[str], *, expected_target: str) -> None:
    assert pip_args[:2] == ["install", "--disable-pip-version-check"]
    assert "-q" in pip_args
    assert expected_target in pip_args

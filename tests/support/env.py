from __future__ import annotations

import os
import sys
from collections.abc import Mapping


TAAC_BUNDLE_ENV_VARS = (
    "TAAC_BUNDLE_WORKDIR",
    "TAAC_CODE_PACKAGE",
    "TAAC_EXPERIMENT",
    "TAAC_INSTALL_PROJECT_DEPS",
    "TAAC_BUNDLE_PIP_EXTRAS",
    "TAAC_PIP_EXTRA_ARGS",
    "TAAC_PIP_EXTRAS",
    "TAAC_PIP_INDEX_URL",
    "TAAC_PYTHON",
    "TAAC_RUNNER",
    "TAAC_SKIP_PIP_INSTALL",
)

TAAC_PLATFORM_PATH_ENV_VARS = (
    "TRAIN_DATA_PATH",
    "TRAIN_CKPT_PATH",
    "TRAIN_LOG_PATH",
    "TRAIN_TF_EVENTS_PATH",
    "EVAL_DATA_PATH",
    "EVAL_RESULT_PATH",
    "MODEL_OUTPUT_PATH",
    "USER_CACHE_PATH",
    "TAAC_SCHEMA_PATH",
)


def clean_subprocess_env(
    updates: Mapping[str, str] | None = None,
    *,
    remove: tuple[str, ...] = (),
    include_platform_paths: bool = False,
    python_runner: bool = False,
) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        *TAAC_BUNDLE_ENV_VARS,
        *(TAAC_PLATFORM_PATH_ENV_VARS if include_platform_paths else ()),
        *remove,
    ):
        env.pop(name, None)
    if python_runner:
        env.update({"TAAC_PYTHON": sys.executable, "TAAC_RUNNER": "python"})
    if updates:
        env.update(updates)
    return env

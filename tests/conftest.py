from __future__ import annotations

from pathlib import Path


def pytest_collection_modifyitems(config, items):
    del config
    for item in items:
        path = Path(str(item.fspath))
        parts = path.parts
        if "unit" in parts:
            item.add_marker("unit")
        elif "integration" in parts:
            item.add_marker("integration")

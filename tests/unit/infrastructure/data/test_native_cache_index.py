from __future__ import annotations

from pathlib import Path

import taac2026.infrastructure.data.native.cache_index as cache_index


def test_native_cache_index_loader_uses_generic_source_and_extension_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_load(**kwargs):
        captured.update(kwargs)
        return object()

    cache_index.load_native_cache_index.cache_clear()
    monkeypatch.setattr(cache_index, "load", fake_load)
    monkeypatch.setenv("TAAC_TORCH_EXTENSIONS_DIR", str(tmp_path))

    try:
        module = cache_index.load_native_cache_index()
    finally:
        cache_index.load_native_cache_index.cache_clear()

    assert module is not None
    assert captured["name"] == "taac_cache_index"
    assert captured["sources"] == [
        str(Path(cache_index.__file__).with_name("cache_index.cpp"))
    ]
    assert captured["build_directory"] == str((tmp_path / "taac_cache_index").resolve())
    assert captured["with_cuda"] is False

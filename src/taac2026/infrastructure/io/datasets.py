from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


PREFERRED_PARQUET_NAME = "demo_1000.parquet"


def _read_ref_revision(ref_path: Path) -> str | None:
    try:
        revision = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return revision or None


def _sorted_parquet_candidates(root: Path) -> list[Path]:
    candidates = [path for path in root.rglob("*.parquet") if path.is_file()]
    candidates.sort(key=lambda path: (path.name != PREFERRED_PARQUET_NAME, len(path.parts), str(path)))
    return candidates


def _resolve_from_snapshot_dir(snapshot_dir: Path) -> Path | None:
    if not snapshot_dir.is_dir():
        return None
    candidates = _sorted_parquet_candidates(snapshot_dir)
    if candidates:
        return candidates[0]
    return None


def _resolve_from_snapshot_container(snapshots_dir: Path) -> Path | None:
    if not snapshots_dir.is_dir():
        return None
    snapshot_dirs = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    snapshot_dirs.sort(key=lambda path: (-path.stat().st_mtime_ns, str(path)))
    for snapshot_dir in snapshot_dirs:
        resolved = _resolve_from_snapshot_dir(snapshot_dir)
        if resolved is not None:
            return resolved
    return None


def _resolve_from_huggingface_cache_root(cache_root: Path) -> Path | None:
    snapshots_dir = cache_root / "snapshots"
    refs_dir = cache_root / "refs"
    if not snapshots_dir.is_dir():
        return None

    ref_paths: list[Path] = []
    preferred_main_ref = refs_dir / "main"
    if preferred_main_ref.is_file():
        ref_paths.append(preferred_main_ref)
    if refs_dir.is_dir():
        ref_paths.extend(
            path
            for path in sorted(refs_dir.rglob("*"))
            if path.is_file() and path != preferred_main_ref
        )

    for ref_path in ref_paths:
        revision = _read_ref_revision(ref_path)
        if revision is None:
            continue
        resolved = _resolve_from_snapshot_dir(snapshots_dir / revision)
        if resolved is not None:
            return resolved

    return _resolve_from_snapshot_container(snapshots_dir)


def _infer_hf_dataset_name_from_cache_root(dataset_path: Path) -> tuple[str, str] | None:
    name = dataset_path.name
    if not name.startswith("datasets--"):
        return None
    payload = name.removeprefix("datasets--")
    if "--" not in payload:
        return None
    owner, repo_name = payload.split("--", 1)
    if not owner or not repo_name:
        return None
    return f"{owner}/{repo_name}", str(dataset_path.parent)


def resolve_parquet_dataset_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        if path.name == "snapshots":
            resolved = _resolve_from_snapshot_container(path)
            if resolved is not None:
                return resolved
        resolved = _resolve_from_huggingface_cache_root(path)
        if resolved is not None:
            return resolved
        candidates = _sorted_parquet_candidates(path)
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Cannot resolve parquet dataset from {dataset_path}")


def iter_dataset_rows(dataset_path: str | Path) -> Iterable[dict[str, Any]]:
    """Load rows from a HF Hub identifier or local parquet path via ``datasets``.

    Accepts:
      - A HF Hub dataset name, e.g. ``"TAAC2026/data_sample_1000"``
      - A local ``.parquet`` file path
      - A local directory (HF cache root or folder containing parquet files)
    """
    from datasets import load_dataset

    path = Path(dataset_path).expanduser()

    if path.is_file():
        return load_dataset("parquet", data_files=str(path), split="train")

    if path.is_dir():
        resolved = resolve_parquet_dataset_path(dataset_path)
        return load_dataset("parquet", data_files=str(resolved), split="train")

    inferred_dataset = _infer_hf_dataset_name_from_cache_root(path)
    if inferred_dataset is not None:
        dataset_name, cache_dir = inferred_dataset
        return load_dataset(dataset_name, split="train", cache_dir=cache_dir)

    return load_dataset(str(dataset_path), split="train")


__all__ = ["iter_dataset_rows", "resolve_parquet_dataset_path"]
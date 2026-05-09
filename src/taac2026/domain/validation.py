"""Shared validation primitives for repository boundary contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TAACBoundaryModel(BaseModel):
    """Base model for JSON, manifest, sidecar, and platform boundary payloads."""

    model_config = ConfigDict(extra="forbid")


__all__ = ["TAACBoundaryModel"]
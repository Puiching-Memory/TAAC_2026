"""Schema and constant helpers for the shared PCVR data pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch.multiprocessing


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature."""

    def __init__(self) -> None:
        self.entries: list[tuple[int, int, int]] = []
        self.total_dim: int = 0
        self._fid_to_entry: dict[int, tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> tuple[int, int]:
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> list[int]:
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": self.entries,
            "total_dim": self.total_dim,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureSchema":
        schema = cls()
        for fid, offset, length in payload["entries"]:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = payload["total_dim"]
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])"
        )
        return "\n".join(lines)


torch.multiprocessing.set_sharing_strategy("file_system")


BUCKET_BOUNDARIES = np.array(
    [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        120,
        180,
        240,
        300,
        360,
        420,
        480,
        540,
        600,
        900,
        1200,
        1500,
        1800,
        2100,
        2400,
        2700,
        3000,
        3300,
        3600,
        5400,
        7200,
        9000,
        10800,
        12600,
        14400,
        16200,
        18000,
        19800,
        21600,
        32400,
        43200,
        54000,
        64800,
        75600,
        86400,
        172800,
        259200,
        345600,
        432000,
        518400,
        604800,
        1123200,
        1641600,
        2160000,
        2592000,
        4320000,
        6048000,
        7776000,
        11664000,
        15552000,
        31536000,
    ],
    dtype=np.int64,
)


NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1


__all__ = ["BUCKET_BOUNDARIES", "FeatureSchema", "NUM_TIME_BUCKETS"]
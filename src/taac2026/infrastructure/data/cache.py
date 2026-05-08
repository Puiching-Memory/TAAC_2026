"""PCVR data cache components."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Hashable, Iterable
from typing import Any

import torch

from taac2026.domain.config import PCVRDataCacheConfig
from taac2026.infrastructure.data.batches import (
    PCVRBatch,
    PCVRSharedTensorSpec,
    clone_pcvr_batch,
    pcvr_batch_row_count,
)
from taac2026.infrastructure.data.native.cache_index import load_native_cache_index


_CACHE_POLICY_CODES = {
    "lru": 0,
    "fifo": 1,
    "lfu": 2,
    "rr": 3,
    "opt": 4,
}
_CACHE_POLICIES = tuple(_CACHE_POLICY_CODES)


def _normalize_cache_policy(policy: str) -> str:
    if policy not in _CACHE_POLICIES:
        raise ValueError(f"unsupported cache policy: {policy}")
    return policy


def _policy_code(policy: str) -> int:
    return _CACHE_POLICY_CODES[_normalize_cache_policy(policy)]


def _key_ids_from_universe(keys: Iterable[Hashable]) -> dict[Hashable, int]:
    key_ids: dict[Hashable, int] = {}
    for key in keys:
        key_ids.setdefault(key, len(key_ids))
    return key_ids


def _trace_index_tensors(
    trace_positions_by_key: dict[int, tuple[int, ...]], key_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    flat_positions: list[int] = []
    for key_id in range(key_count):
        key_positions = trace_positions_by_key.get(key_id, ())
        flat_positions.extend(key_positions)
        offsets.append(len(flat_positions))
    return (
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(flat_positions, dtype=torch.int64),
    )


class PCVRMemoryBatchCache:
    """Per-process cache for converted base PCVR batches.

    Batch payloads stay in Python objects; all policy decisions and slot metadata
    live in the native C++ index so single-worker and multi-worker paths share the
    same eviction behavior.
    """

    uses_global_access_trace = False

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_batches: int = 0,
        policy: str = "lru",
    ) -> None:
        self.enabled = enabled
        self.max_batches = max(0, int(max_batches))
        self.policy = _normalize_cache_policy(policy)
        self._slot_items: dict[int, PCVRBatch] = {}
        self._key_ids: dict[Hashable, int] = {}
        self._key_to_slot = torch.empty(0, dtype=torch.int64)
        self._slot_to_key = torch.full((self.max_batches,), -1, dtype=torch.int64)
        self._access_count = torch.zeros(1, dtype=torch.int64)
        self._slot_last_access = torch.zeros(self.max_batches, dtype=torch.int64)
        self._slot_frequency = torch.zeros(self.max_batches, dtype=torch.int64)
        self._slot_insert_order = torch.zeros(self.max_batches, dtype=torch.int64)
        self._slot_versions = torch.zeros(self.max_batches, dtype=torch.int64)
        self._rr_state = torch.ones(1, dtype=torch.int64)
        self._trace_offsets_tensor = torch.empty(0, dtype=torch.int64)
        self._trace_positions_tensor = torch.empty(0, dtype=torch.int64)
        self._trace_positions_by_key: dict[int, tuple[int, ...]] = {}
        self._trace_length = 0
        self._trace_cyclic = True
        self._native_cache_available = False
        self._hits = 0
        self._misses = 0

    @classmethod
    def from_config(cls, config: PCVRDataCacheConfig | None) -> PCVRMemoryBatchCache:
        if config is None:
            return cls()
        return cls(
            enabled=config.enabled,
            max_batches=config.max_batches,
            policy="lru" if config.mode == "none" else config.mode,
        )

    def configure_access_trace(
        self,
        trace: Iterable[Hashable],
        *,
        cyclic: bool = True,
        key_universe: Iterable[Hashable] = (),
    ) -> None:
        if not self.enabled or self.max_batches <= 0:
            return

        if not self._opt_enabled:
            return

        key_ids = _key_ids_from_universe(key_universe)
        positions: dict[int, list[int]] = {}
        trace_length = 0
        for key in trace:
            key_id = key_ids.setdefault(key, len(key_ids))
            positions.setdefault(key_id, []).append(trace_length)
            trace_length += 1

        trace_positions_by_key = {
            key_id: tuple(values) for key_id, values in positions.items()
        }
        if not key_ids:
            self._reset_index({}, clear_items=True)
            self._trace_positions_by_key = {}
            self._trace_offsets_tensor = torch.empty(0, dtype=torch.int64)
            self._trace_positions_tensor = torch.empty(0, dtype=torch.int64)
            self._trace_length = 0
            self._trace_cyclic = bool(cyclic)
            return

        same_index = key_ids == self._key_ids
        same_trace = (
            trace_length == self._trace_length
            and cyclic == self._trace_cyclic
            and trace_positions_by_key == self._trace_positions_by_key
        )
        if same_index and same_trace:
            self._access_count.zero_()
            return

        self._reset_index(key_ids, clear_items=True)
        self._trace_positions_by_key = trace_positions_by_key
        self._trace_offsets_tensor, self._trace_positions_tensor = _trace_index_tensors(
            trace_positions_by_key, len(key_ids)
        )
        self._trace_length = trace_length
        self._trace_cyclic = bool(cyclic)

    def configure_key_universe(self, keys: Iterable[Hashable]) -> None:
        if not self.enabled or self.max_batches <= 0:
            return
        key_ids = _key_ids_from_universe(keys)
        if key_ids == self._key_ids and self._key_to_slot.numel() == len(key_ids):
            return
        self._reset_index(key_ids, clear_items=True)
        self._trace_positions_by_key = {}
        self._trace_offsets_tensor = torch.zeros(len(key_ids) + 1, dtype=torch.int64)
        self._trace_positions_tensor = torch.empty(0, dtype=torch.int64)
        self._trace_length = 0
        self._trace_cyclic = True

    def get(self, key: Hashable) -> PCVRBatch | None:
        if not self.enabled or self.max_batches <= 0:
            return None
        key_id = self._key_id_for(key, create=True)
        slot_index = self._native().get_slot(
            self._effective_policy_code(),
            key_id,
            self._effective_trace_length(),
            self._trace_cyclic,
            self._key_to_slot,
            self._slot_to_key,
            self._access_count,
            self._slot_last_access,
            self._slot_frequency,
            self._slot_versions,
        )
        if int(slot_index) < 0:
            self._misses += 1
            return None
        batch = self._slot_items.get(int(slot_index))
        if batch is None:
            self._misses += 1
            return None
        self._hits += 1
        return clone_pcvr_batch(batch)

    def put(self, key: Hashable, batch: PCVRBatch) -> None:
        if not self.enabled or self.max_batches <= 0:
            return
        key_id = self._key_id_for(key, create=True)
        slot_index = int(
            self._native().allocate_slot(
                self._effective_policy_code(),
                key_id,
                self._effective_trace_length(),
                self._trace_cyclic,
                self._key_to_slot,
                self._slot_to_key,
                self._access_count,
                self._slot_last_access,
                self._slot_frequency,
                self._slot_insert_order,
                self._slot_versions,
                self._rr_state,
                self._trace_offsets_tensor,
                self._trace_positions_tensor,
            )
        )
        if slot_index < 0:
            return
        self._slot_items[slot_index] = clone_pcvr_batch(batch)

    def __len__(self) -> int:
        if not self.enabled or self.max_batches <= 0:
            return 0
        return int(self._native().size(self._slot_to_key))

    def stats(self) -> dict[str, float | int | str | bool]:
        total = self._hits + self._misses
        return {
            "enabled": self.enabled,
            "policy": self.policy,
            "effective_policy": self._effective_policy(),
            "items": len(self),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "opt_active": self._opt_active,
            "native_cache_active": self._native_cache_available,
            "native_opt_active": self._opt_active and self._native_cache_available,
            "trace_length": self._trace_length,
        }

    @property
    def _opt_enabled(self) -> bool:
        return self.enabled and self.policy == "opt"

    @property
    def _opt_active(self) -> bool:
        return self._opt_enabled and self._trace_length > 0

    def _effective_policy(self) -> str:
        return "opt" if self._opt_active else ("lru" if self.policy == "opt" else self.policy)

    def _effective_policy_code(self) -> int:
        return _policy_code(self._effective_policy())

    def _effective_trace_length(self) -> int:
        return self._trace_length if self._opt_active else 0

    def _native(self) -> Any:
        module = load_native_cache_index()
        self._native_cache_available = True
        return module

    def _key_id_for(self, key: Hashable, *, create: bool) -> int:
        key_id = self._key_ids.get(key)
        if key_id is not None:
            return key_id
        if not create:
            raise KeyError(f"cache key missing from configured key universe: {key!r}")
        key_id = len(self._key_ids)
        self._key_ids[key] = key_id
        if self._key_to_slot.numel() <= key_id:
            extension = torch.full((1,), -1, dtype=torch.int64)
            self._key_to_slot = torch.cat((self._key_to_slot, extension))
            if self._trace_offsets_tensor.numel() == 0:
                self._trace_offsets_tensor = torch.zeros(
                    self._key_to_slot.numel() + 1, dtype=torch.int64
                )
            elif self._trace_offsets_tensor.numel() < self._key_to_slot.numel() + 1:
                last = self._trace_offsets_tensor[-1:].clone()
                self._trace_offsets_tensor = torch.cat((self._trace_offsets_tensor, last))
        return key_id

    def _reset_index(self, key_ids: dict[Hashable, int], *, clear_items: bool) -> None:
        self._key_ids = dict(key_ids)
        self._key_to_slot = torch.full((len(key_ids),), -1, dtype=torch.int64)
        self._slot_to_key = torch.full((self.max_batches,), -1, dtype=torch.int64)
        self._access_count.zero_()
        self._slot_last_access.zero_()
        self._slot_frequency.zero_()
        self._slot_insert_order.zero_()
        self._slot_versions.zero_()
        self._rr_state.fill_(1)
        self._hits = 0
        self._misses = 0
        if clear_items:
            self._slot_items.clear()


class PCVRSharedBatchCache:
    """Shared-memory CPU batch cache for multi-worker single-card training."""

    uses_global_access_trace = True
    uses_native_opt_index = False

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_batches: int = 0,
        policy: str = "lru",
        tensor_specs: dict[str, PCVRSharedTensorSpec] | None = None,
        static_values: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_batches = max(0, int(max_batches))
        self.policy = _normalize_cache_policy(policy)
        self.uses_global_access_trace = self.policy == "opt"
        self.uses_native_opt_index = self.policy == "opt"
        self.static_values = dict(static_values or {})
        self.tensor_specs = dict(tensor_specs or {})
        self._lock = mp.RLock()
        self._key_ids: dict[Hashable, int] = {}
        self._key_to_slot = torch.empty(0, dtype=torch.int64).share_memory_()
        self._slot_to_key = torch.full((self.max_batches,), -1, dtype=torch.int64).share_memory_()
        self._row_counts = torch.zeros(self.max_batches, dtype=torch.int64).share_memory_()
        self._access_count = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._slot_last_access = torch.zeros(self.max_batches, dtype=torch.int64).share_memory_()
        self._slot_frequency = torch.zeros(self.max_batches, dtype=torch.int64).share_memory_()
        self._slot_insert_order = torch.zeros(self.max_batches, dtype=torch.int64).share_memory_()
        self._slot_versions = torch.zeros(self.max_batches, dtype=torch.int64).share_memory_()
        self._rr_state = torch.ones(1, dtype=torch.int64).share_memory_()
        self._hits = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._misses = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._trace_positions_by_key: dict[int, tuple[int, ...]] = {}
        self._trace_offsets_tensor = torch.empty(0, dtype=torch.int64).share_memory_()
        self._trace_positions_tensor = torch.empty(0, dtype=torch.int64).share_memory_()
        self._native_cache_available = False
        self._trace_length = 0
        self._trace_cyclic = True
        self._storage = {
            key: torch.empty((self.max_batches, *spec.shape), dtype=spec.dtype).share_memory_()
            for key, spec in self.tensor_specs.items()
        }
        self._storage_items = tuple(self._storage.items())

    def __len__(self) -> int:
        if not self.enabled or self.max_batches <= 0:
            return 0
        with self._lock:
            return int(self._native().size(self._slot_to_key))

    @property
    def _opt_enabled(self) -> bool:
        return self.enabled and self.policy == "opt"

    @property
    def _opt_active(self) -> bool:
        return self._opt_enabled and self._trace_length > 0

    @property
    def _shared_lru_active(self) -> bool:
        return self.enabled and self.policy == "lru" and self._key_to_slot.numel() > 0

    def configure_key_universe(self, keys: Iterable[Hashable]) -> None:
        if not self.enabled or self.max_batches <= 0:
            return
        key_ids = _key_ids_from_universe(keys)
        with self._lock:
            if key_ids == self._key_ids and self._key_to_slot.numel() == len(key_ids):
                return
            self._reset_index(key_ids)
            self._trace_positions_by_key = {}
            self._trace_offsets_tensor = torch.zeros(len(key_ids) + 1, dtype=torch.int64).share_memory_()
            self._trace_positions_tensor = torch.empty(0, dtype=torch.int64).share_memory_()
            self._trace_length = 0
            self._trace_cyclic = True
            self._native()

    def configure_access_trace(
        self,
        trace: Iterable[Hashable],
        *,
        cyclic: bool = True,
        key_universe: Iterable[Hashable] = (),
    ) -> None:
        if not self.enabled or self.max_batches <= 0:
            return

        if not self._opt_enabled:
            return
        key_ids = _key_ids_from_universe(key_universe)
        positions: dict[int, list[int]] = {}
        trace_length = 0
        for key in trace:
            key_id = key_ids.setdefault(key, len(key_ids))
            positions.setdefault(key_id, []).append(trace_length)
            trace_length += 1
        trace_positions_by_key = {
            key_id: tuple(values) for key_id, values in positions.items()
        }

        with self._lock:
            same_index = key_ids == self._key_ids and self._key_to_slot.numel() == len(key_ids)
            same_trace = (
                trace_length == self._trace_length
                and cyclic == self._trace_cyclic
                and trace_positions_by_key == self._trace_positions_by_key
            )
            if same_index and same_trace:
                return

            self._reset_index(key_ids)
            self._trace_positions_by_key = trace_positions_by_key
            offsets, positions_tensor = _trace_index_tensors(trace_positions_by_key, len(key_ids))
            self._trace_offsets_tensor = offsets.share_memory_()
            self._trace_positions_tensor = positions_tensor.share_memory_()
            self._trace_length = trace_length
            self._trace_cyclic = bool(cyclic)
            self._native()

    def get(self, key: Hashable) -> PCVRBatch | None:
        if not self.enabled or self.max_batches <= 0:
            return None
        with self._lock:
            key_id = self._key_id_for(key)
            slot_index = int(
                self._native().get_slot(
                    self._effective_policy_code(),
                    key_id,
                    self._effective_trace_length(),
                    self._trace_cyclic,
                    self._key_to_slot,
                    self._slot_to_key,
                    self._access_count,
                    self._slot_last_access,
                    self._slot_frequency,
                    self._slot_versions,
                )
            )
            if slot_index < 0:
                self._misses += 1
                return None
            version_before = int(self._slot_versions[slot_index].item())
            if version_before & 1:
                self._misses += 1
                return None
            row_count = int(self._row_counts[slot_index].item())
        batch = self._materialize_slot(slot_index, row_count)
        with self._lock:
            version_after = int(self._slot_versions[slot_index].item())
            if version_after != version_before or (version_after & 1):
                self._misses += 1
                return None
            self._hits += 1
        return batch

    def put(self, key: Hashable, batch: PCVRBatch) -> None:
        if not self.enabled or self.max_batches <= 0:
            return
        with self._lock:
            key_id = self._key_id_for(key)
            slot_index = int(
                self._native().allocate_slot(
                    self._effective_policy_code(),
                    key_id,
                    self._effective_trace_length(),
                    self._trace_cyclic,
                    self._key_to_slot,
                    self._slot_to_key,
                    self._access_count,
                    self._slot_last_access,
                    self._slot_frequency,
                    self._slot_insert_order,
                    self._slot_versions,
                    self._rr_state,
                    self._trace_offsets_tensor,
                    self._trace_positions_tensor,
                )
            )
            if slot_index < 0:
                return
            self._slot_versions[slot_index] = int(self._slot_versions[slot_index].item()) | 1
        try:
            row_count = self._write_slot(slot_index, batch)
        except Exception:
            with self._lock:
                self._slot_versions[slot_index] = int(self._slot_versions[slot_index].item()) + 1
            raise
        with self._lock:
            self._row_counts[slot_index] = row_count
            self._slot_versions[slot_index] = int(self._slot_versions[slot_index].item()) + 1

    def _write_slot(self, slot_index: int, batch: PCVRBatch) -> int:
        row_count = pcvr_batch_row_count(batch)
        for tensor_key, storage in self._storage_items:
            value = batch.get(tensor_key)
            if not isinstance(value, torch.Tensor):
                raise KeyError(f"shared cache tensor key missing from batch: {tensor_key}")
            target = storage[slot_index]
            if row_count > 0:
                target[:row_count].copy_(value)
        return row_count

    def _materialize_slot(self, slot_index: int, row_count: int) -> PCVRBatch:
        batch: PCVRBatch = dict(self.static_values)
        for tensor_key, storage in self._storage_items:
            batch[tensor_key] = storage[slot_index][:row_count].clone()
        return batch

    def _key_id_for(self, key: Hashable) -> int:
        try:
            return self._key_ids[key]
        except KeyError as exc:
            if self._opt_active:
                raise KeyError(
                    f"OPT cache key missing from configured access trace: {key!r}"
                ) from exc
            raise KeyError(f"cache key missing from configured key universe: {key!r}") from exc

    def _effective_policy(self) -> str:
        return "opt" if self._opt_active else ("lru" if self.policy == "opt" else self.policy)

    def _effective_policy_code(self) -> int:
        return _policy_code(self._effective_policy())

    def _effective_trace_length(self) -> int:
        return self._trace_length if self._opt_active else 0

    def _native(self) -> Any:
        module = load_native_cache_index()
        self._native_cache_available = True
        return module

    def _reset_counters(self) -> None:
        self._access_count.zero_()
        self._hits.zero_()
        self._misses.zero_()

    def _reset_index(self, key_ids: dict[Hashable, int]) -> None:
        self._key_ids = dict(key_ids)
        self._key_to_slot = torch.full((len(key_ids),), -1, dtype=torch.int64).share_memory_()
        self._slot_to_key.fill_(-1)
        self._row_counts.zero_()
        self._slot_last_access.zero_()
        self._slot_frequency.zero_()
        self._slot_insert_order.zero_()
        self._slot_versions.zero_()
        self._rr_state.fill_(1)
        self._reset_counters()

    def stats(self) -> dict[str, float | int | str | bool]:
        with self._lock:
            hits = int(self._hits.item())
            misses = int(self._misses.item())
            total = hits + misses
            items = int(self._native().size(self._slot_to_key)) if self.enabled and self.max_batches > 0 else 0
            return {
                "enabled": self.enabled,
                "policy": self.policy,
                "effective_policy": self._effective_policy(),
                "items": items,
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / total if total > 0 else 0.0,
                "shared_lru_active": self._shared_lru_active,
                "opt_active": self._opt_active,
                "native_cache_active": self._native_cache_available,
                "native_opt_active": self._opt_active and self._native_cache_available,
                "trace_length": self._trace_length,
            }


__all__ = ["PCVRMemoryBatchCache", "PCVRSharedBatchCache", "PCVRSharedTensorSpec"]

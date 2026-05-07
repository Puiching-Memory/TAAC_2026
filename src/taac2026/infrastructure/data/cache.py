"""PCVR data cache components."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Hashable, Iterable
from typing import Any

from cachetools import Cache, FIFOCache, LFUCache, LRUCache, RRCache
import torch

from taac2026.domain.config import PCVRDataCacheConfig
from taac2026.infrastructure.data.batches import (
    PCVRBatch,
    PCVRSharedTensorSpec,
    clone_pcvr_batch,
    pcvr_batch_row_count,
)
from taac2026.infrastructure.data.native.opt_cache import load_native_opt_cache


_CACHETOOLS_POLICIES: dict[str, type[Cache]] = {
    "lru": LRUCache,
    "fifo": FIFOCache,
    "lfu": LFUCache,
    "rr": RRCache,
}
_CACHE_POLICIES = (*_CACHETOOLS_POLICIES, "opt")


def _normalize_cache_policy(policy: str) -> str:
    if policy not in _CACHE_POLICIES:
        raise ValueError(f"unsupported cache policy: {policy}")
    return policy


def _new_cachetools_cache(policy: str, max_batches: int) -> Cache:
    try:
        cache_type = _CACHETOOLS_POLICIES[policy]
    except KeyError as exc:
        raise ValueError(f"cache policy is not handled by cachetools: {policy}") from exc
    return cache_type(max_batches)


class PCVRMemoryBatchCache:
    """Small per-process cache for converted base PCVR batches."""

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
        self._items: dict[Hashable, PCVRBatch] | Cache = {}
        self._trace_positions: dict[Hashable, int] = {}
        self._trace_length = 0
        self._trace_cyclic = True
        self._access_count = 0
        self._opt_fallback = False

    @classmethod
    def from_config(cls, config: PCVRDataCacheConfig | None) -> PCVRMemoryBatchCache:
        if config is None:
            return cls()
        return cls(
            enabled=config.enabled,
            max_batches=config.max_batches,
            policy="lru" if config.mode == "none" else config.mode,
        )

    def configure_access_trace(self, trace: Iterable[Hashable], *, cyclic: bool = True) -> None:
        if not self._opt_enabled or self.max_batches <= 0:
            return

        trace_positions: dict[Hashable, int] = {}
        trace_length = 0
        for key in trace:
            if key in trace_positions:
                self._opt_fallback = True
                self._ensure_cachetools_items("lru")
                return
            trace_positions[key] = trace_length
            trace_length += 1

        if trace_length == 0:
            self._trace_positions = {}
            self._trace_length = 0
            self._access_count = 0
            self._opt_fallback = False
            self._ensure_cachetools_items("lru")
            return

        if trace_length == self._trace_length and cyclic == self._trace_cyclic and trace_positions == self._trace_positions:
            return

        self._items = {}
        self._trace_positions = trace_positions
        self._trace_length = trace_length
        self._trace_cyclic = cyclic
        self._access_count = 0
        self._opt_fallback = False

    def get(self, key: Hashable) -> PCVRBatch | None:
        if not self.enabled or self.max_batches <= 0:
            return None
        self._record_access(key)
        try:
            batch = self._items[key]
        except KeyError:
            return None
        return clone_pcvr_batch(batch)

    def put(self, key: Hashable, batch: PCVRBatch) -> None:
        if not self.enabled or self.max_batches <= 0:
            return
        if not self._opt_active:
            self._ensure_cachetools_items("lru" if self.policy == "opt" else self.policy)
            self._items[key] = clone_pcvr_batch(batch)
            return

        self._items[key] = clone_pcvr_batch(batch)
        while self.max_batches > 0 and len(self._items) > self.max_batches:
            victim = self._select_opt_victim()
            del self._items[victim]

    def __len__(self) -> int:
        return len(self._items)

    @property
    def _opt_enabled(self) -> bool:
        return self.enabled and self.policy == "opt"

    @property
    def _opt_active(self) -> bool:
        return self._opt_enabled and self._trace_length > 0 and not self._opt_fallback

    def _record_access(self, key: Hashable) -> None:
        if not self._opt_active:
            return
        expected_position = self._access_count % self._trace_length
        if self._trace_positions.get(key) != expected_position:
            self._opt_fallback = True
            self._ensure_cachetools_items("lru")
            return
        self._access_count += 1

    def _ensure_cachetools_items(self, policy: str) -> None:
        if isinstance(self._items, Cache):
            return
        cache = _new_cachetools_cache(policy, self.max_batches)
        for key, value in self._items.items():
            cache[key] = value
        self._items = cache

    def _next_use(self, key: Hashable) -> float:
        position = self._trace_positions.get(key)
        if position is None or self._trace_length <= 0:
            return float("inf")
        cycle_index, cycle_position = divmod(self._access_count, self._trace_length)
        if position >= cycle_position:
            return float(cycle_index * self._trace_length + position)
        if self._trace_cyclic:
            return float((cycle_index + 1) * self._trace_length + position)
        return float("inf")

    def _select_opt_victim(self) -> Hashable:
        victim: Hashable | None = None
        farthest = float("-inf")
        for key in self._items:
            next_use = self._next_use(key)
            if next_use > farthest:
                farthest = next_use
                victim = key
        if victim is None:
            raise RuntimeError("OPT cache eviction requested for empty cache")
        return victim


class PCVRSharedBatchCache:
    """Shared-memory CPU batch cache for multi-worker single-card training."""

    uses_global_access_trace = False
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
        self._slot_cache = (
            _new_cachetools_cache(self.policy, self.max_batches)
            if self.policy != "opt" and self.max_batches > 0
            else None
        )
        self._free_slots = list(range(self.max_batches))
        self._key_ids: dict[Hashable, int] = {}
        self._key_to_slot = torch.empty(0, dtype=torch.int64).share_memory_()
        self._slot_to_key = torch.full((self.max_batches,), -1, dtype=torch.int64).share_memory_()
        self._row_counts = torch.zeros(self.max_batches, dtype=torch.int64).share_memory_()
        self._last_access = torch.zeros(self.max_batches, dtype=torch.int64).share_memory_()
        self._touch_counter = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._access_count = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._trace_length = 0
        self._trace_cyclic = True
        self._storage = {
            key: torch.empty((self.max_batches, *spec.shape), dtype=spec.dtype).share_memory_()
            for key, spec in self.tensor_specs.items()
        }

    def __len__(self) -> int:
        if self._opt_active:
            with self._lock:
                return int(load_native_opt_cache().size(self._slot_to_key))
        if self._slot_cache is not None:
            return len(self._slot_cache)
        return 0

    @property
    def _opt_enabled(self) -> bool:
        return self.enabled and self.policy == "opt"

    @property
    def _opt_active(self) -> bool:
        return self._opt_enabled and self._trace_length > 0

    def configure_access_trace(self, trace: Iterable[Hashable], *, cyclic: bool = True) -> None:
        if not self._opt_enabled or self.max_batches <= 0:
            return
        key_ids: dict[Hashable, int] = {}
        trace_length = 0
        for key in trace:
            if key in key_ids:
                raise ValueError(f"OPT cache access trace contains duplicate key: {key!r}")
            key_ids[key] = trace_length
            trace_length += 1

        with self._lock:
            if not key_ids:
                self._key_ids = {}
                self._key_to_slot = torch.empty(0, dtype=torch.int64).share_memory_()
                self._slot_to_key.fill_(-1)
                self._row_counts.zero_()
                self._last_access.zero_()
                self._touch_counter.zero_()
                self._access_count.zero_()
                self._trace_length = 0
                self._trace_cyclic = bool(cyclic)
                return

            if trace_length == self._trace_length and cyclic == self._trace_cyclic and key_ids == self._key_ids:
                return

            load_native_opt_cache()
            self._key_ids = key_ids
            self._key_to_slot = torch.full((trace_length,), -1, dtype=torch.int64).share_memory_()
            self._slot_to_key.fill_(-1)
            self._row_counts.zero_()
            self._last_access.zero_()
            self._touch_counter.zero_()
            self._access_count.zero_()
            self._trace_length = trace_length
            self._trace_cyclic = bool(cyclic)
            self._reset_slot_cache()

    def get(self, key: Hashable) -> PCVRBatch | None:
        if not self.enabled or self.max_batches <= 0:
            return None
        with self._lock:
            slot_index = self._get_slot_index(key)
            if slot_index < 0:
                return None
            row_count = int(self._row_counts[slot_index].item())
            return self._materialize_slot(slot_index, row_count)

    def put(self, key: Hashable, batch: PCVRBatch) -> None:
        if not self.enabled or self.max_batches <= 0:
            return
        with self._lock:
            slot_index = self._allocate_slot_for_key(key)
            if slot_index is None:
                return
            self._write_slot(slot_index, key, batch)

    def _allocate_slot_for_key(self, key: Hashable) -> int | None:
        if self._opt_active:
            key_id = self._key_id_for(key)
            slot_index = int(
                load_native_opt_cache().allocate_slot(
                    key_id,
                    self._trace_length,
                    self._trace_cyclic,
                    self._key_to_slot,
                    self._slot_to_key,
                    self._last_access,
                    self._touch_counter,
                    self._access_count,
                )
            )
            if slot_index < 0:
                return None
            return slot_index

        if self.policy == "opt" or self._slot_cache is None:
            return None

        try:
            return int(self._slot_cache[key])
        except KeyError:
            pass

        if self._free_slots:
            slot_index = self._free_slots.pop(0)
        else:
            _victim_key, slot_index = self._slot_cache.popitem()
        self._slot_cache[key] = int(slot_index)
        return int(slot_index)

    def _write_slot(self, slot_index: int, key: Hashable, batch: PCVRBatch) -> None:
        row_count = pcvr_batch_row_count(batch)
        for tensor_key, storage in self._storage.items():
            value = batch.get(tensor_key)
            if not isinstance(value, torch.Tensor):
                raise KeyError(f"shared cache tensor key missing from batch: {tensor_key}")
            target = storage[slot_index]
            target.zero_()
            if row_count > 0:
                target[:row_count].copy_(value)
        self._row_counts[slot_index] = row_count

    def _materialize_slot(self, slot_index: int, row_count: int) -> PCVRBatch:
        batch: PCVRBatch = dict(self.static_values)
        for tensor_key, storage in self._storage.items():
            batch[tensor_key] = storage[slot_index][:row_count].clone()
        return batch

    def _key_id_for(self, key: Hashable) -> int:
        try:
            return self._key_ids[key]
        except KeyError as exc:
            raise KeyError(f"OPT cache key missing from configured access trace: {key!r}") from exc

    def _get_slot_index(self, key: Hashable) -> int:
        if self._opt_active:
            key_id = self._key_id_for(key)
            return int(
                load_native_opt_cache().get_slot(
                    key_id,
                    self._trace_length,
                    self._trace_cyclic,
                    self._key_to_slot,
                    self._last_access,
                    self._touch_counter,
                    self._access_count,
                )
            )

        if self._slot_cache is not None:
            try:
                return int(self._slot_cache[key])
            except KeyError:
                return -1
        return -1

    def _reset_slot_cache(self) -> None:
        if self._slot_cache is not None:
            self._slot_cache.clear()
        self._free_slots = list(range(self.max_batches))


__all__ = ["PCVRMemoryBatchCache", "PCVRSharedBatchCache", "PCVRSharedTensorSpec"]

#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int64_t kEmpty = -1;
constexpr int64_t kNever = std::numeric_limits<int64_t>::max() / 4;

constexpr int64_t kPolicyLru = 0;
constexpr int64_t kPolicyFifo = 1;
constexpr int64_t kPolicyLfu = 2;
constexpr int64_t kPolicyRr = 3;
constexpr int64_t kPolicyOpt = 4;

void check_i64_cpu(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.dtype() == torch::kInt64, name, " must be an int64 tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int64_t* mutable_i64(torch::Tensor& tensor, const char* name) {
  check_i64_cpu(tensor, name);
  return tensor.data_ptr<int64_t>();
}

const int64_t* const_i64(const torch::Tensor& tensor, const char* name) {
  check_i64_cpu(tensor, name);
  return tensor.data_ptr<int64_t>();
}

void check_policy(int64_t policy) {
  TORCH_CHECK(
      policy >= kPolicyLru && policy <= kPolicyOpt,
      "unsupported cache policy code: ",
      policy);
}

void check_key_id(int64_t key_id, int64_t key_count) {
  TORCH_CHECK(key_id >= 0 && key_id < key_count, "key_id out of key universe bounds");
}

void check_slot_metadata(
    const torch::Tensor& slot_to_key,
    const torch::Tensor& slot_last_access,
    const torch::Tensor& slot_frequency,
    const torch::Tensor& slot_insert_order,
    const torch::Tensor& slot_versions) {
  const int64_t slot_count = slot_to_key.numel();
  TORCH_CHECK(slot_last_access.numel() == slot_count, "slot_last_access must match slot count");
  TORCH_CHECK(slot_frequency.numel() == slot_count, "slot_frequency must match slot count");
  TORCH_CHECK(slot_insert_order.numel() == slot_count, "slot_insert_order must match slot count");
  TORCH_CHECK(slot_versions.numel() == slot_count, "slot_versions must match slot count");
}

bool slot_busy(const int64_t* slot_versions, int64_t slot) {
  return (slot_versions[slot] & 1LL) != 0;
}

int64_t lower_bound_position(
    const int64_t* positions,
    int64_t start,
    int64_t end,
    int64_t value) {
  int64_t lo = start;
  int64_t hi = end;
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    if (positions[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

int64_t next_use(
    int64_t key_id,
    int64_t access_count,
    int64_t trace_length,
    bool cyclic,
    const int64_t* offsets,
    const int64_t* positions,
    int64_t key_count) {
  if (key_id < 0 || key_id >= key_count || trace_length <= 0) {
    return kNever;
  }
  const int64_t start = offsets[key_id];
  const int64_t end = offsets[key_id + 1];
  if (start >= end) {
    return kNever;
  }
  const int64_t cycle_index = access_count / trace_length;
  const int64_t cycle_position = access_count % trace_length;
  const int64_t index = lower_bound_position(positions, start, end, cycle_position);
  if (index < end) {
    return cycle_index * trace_length + positions[index];
  }
  if (cyclic) {
    return (cycle_index + 1) * trace_length + positions[start];
  }
  return kNever;
}

int64_t find_empty_slot(const int64_t* slot_to_key, int64_t slot_count) {
  for (int64_t slot = 0; slot < slot_count; ++slot) {
    if (slot_to_key[slot] == kEmpty) {
      return slot;
    }
  }
  return kEmpty;
}

int64_t select_lru_victim(
    const int64_t* slot_last_access,
    const int64_t* slot_versions,
    int64_t slot_count) {
  int64_t victim = kEmpty;
  int64_t best = std::numeric_limits<int64_t>::max();
  for (int64_t slot = 0; slot < slot_count; ++slot) {
    if (slot_busy(slot_versions, slot)) {
      continue;
    }
    if (slot_last_access[slot] < best) {
      best = slot_last_access[slot];
      victim = slot;
    }
  }
  return victim;
}

int64_t select_fifo_victim(
    const int64_t* slot_insert_order,
    const int64_t* slot_versions,
    int64_t slot_count) {
  int64_t victim = kEmpty;
  int64_t best = std::numeric_limits<int64_t>::max();
  for (int64_t slot = 0; slot < slot_count; ++slot) {
    if (slot_busy(slot_versions, slot)) {
      continue;
    }
    if (slot_insert_order[slot] < best) {
      best = slot_insert_order[slot];
      victim = slot;
    }
  }
  return victim;
}

int64_t select_lfu_victim(
    const int64_t* slot_frequency,
    const int64_t* slot_insert_order,
    const int64_t* slot_versions,
    int64_t slot_count) {
  int64_t victim = kEmpty;
  int64_t best_frequency = std::numeric_limits<int64_t>::max();
  int64_t best_order = std::numeric_limits<int64_t>::max();
  for (int64_t slot = 0; slot < slot_count; ++slot) {
    if (slot_busy(slot_versions, slot)) {
      continue;
    }
    const int64_t frequency = slot_frequency[slot];
    const int64_t order = slot_insert_order[slot];
    if (frequency < best_frequency || (frequency == best_frequency && order < best_order)) {
      best_frequency = frequency;
      best_order = order;
      victim = slot;
    }
  }
  return victim;
}

int64_t select_rr_victim(
    int64_t* rr_state,
    const int64_t* slot_versions,
    int64_t slot_count) {
  int64_t available = 0;
  for (int64_t slot = 0; slot < slot_count; ++slot) {
    if (!slot_busy(slot_versions, slot)) {
      available += 1;
    }
  }
  if (available <= 0) {
    return kEmpty;
  }
  uint64_t state = static_cast<uint64_t>(rr_state[0]);
  if (state == 0) {
    state = 0x9E3779B97F4A7C15ULL;
  }
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  rr_state[0] = static_cast<int64_t>(state & 0x7FFFFFFFFFFFFFFFULL);
  int64_t target = static_cast<int64_t>(state % static_cast<uint64_t>(available));
  for (int64_t slot = 0; slot < slot_count; ++slot) {
    if (slot_busy(slot_versions, slot)) {
      continue;
    }
    if (target == 0) {
      return slot;
    }
    target -= 1;
  }
  return kEmpty;
}

int64_t select_opt_victim(
    const int64_t* slot_to_key,
    const int64_t* slot_versions,
    int64_t slot_count,
    int64_t access_count,
    int64_t trace_length,
    bool cyclic,
    const int64_t* offsets,
    const int64_t* positions,
    int64_t key_count,
    int64_t* victim_next_use) {
  int64_t victim_slot = kEmpty;
  int64_t farthest_next_use = std::numeric_limits<int64_t>::min();
  for (int64_t slot = 0; slot < slot_count; ++slot) {
    if (slot_busy(slot_versions, slot)) {
      continue;
    }
    const int64_t cached_key = slot_to_key[slot];
    const int64_t cached_next_use = next_use(
        cached_key,
        access_count,
        trace_length,
        cyclic,
        offsets,
        positions,
        key_count);
    if (cached_next_use > farthest_next_use) {
      farthest_next_use = cached_next_use;
      victim_slot = slot;
    }
  }
  *victim_next_use = farthest_next_use;
  return victim_slot;
}

void assign_slot(
    int64_t policy,
    int64_t key_id,
    int64_t slot,
    int64_t* key_to_slot,
    int64_t* slot_to_key,
    int64_t* access_count,
    int64_t* slot_last_access,
    int64_t* slot_frequency,
    int64_t* slot_insert_order) {
  const int64_t old_key_id = slot_to_key[slot];
  if (old_key_id >= 0) {
    key_to_slot[old_key_id] = kEmpty;
  }
  slot_to_key[slot] = key_id;
  key_to_slot[key_id] = slot;

  if (policy != kPolicyOpt) {
    access_count[0] += 1;
    const int64_t timestamp = access_count[0];
    slot_last_access[slot] = timestamp;
    slot_insert_order[slot] = timestamp;
    slot_frequency[slot] = 1;
  }
}

}  // namespace

int64_t cache_get_slot(
    int64_t policy,
    int64_t key_id,
    int64_t trace_length,
    bool cyclic,
    torch::Tensor key_to_slot,
    torch::Tensor slot_to_key,
    torch::Tensor access_count,
    torch::Tensor slot_last_access,
    torch::Tensor slot_frequency,
    torch::Tensor slot_versions) {
  (void)trace_length;
  (void)cyclic;
  check_policy(policy);
  int64_t* key_to_slot_ptr = mutable_i64(key_to_slot, "key_to_slot");
  const int64_t* slot_to_key_ptr = const_i64(slot_to_key, "slot_to_key");
  int64_t* access_count_ptr = mutable_i64(access_count, "access_count");
  int64_t* slot_last_access_ptr = mutable_i64(slot_last_access, "slot_last_access");
  int64_t* slot_frequency_ptr = mutable_i64(slot_frequency, "slot_frequency");
  const int64_t* slot_versions_ptr = const_i64(slot_versions, "slot_versions");
  TORCH_CHECK(access_count.numel() == 1, "access_count must contain exactly one value");
  TORCH_CHECK(slot_last_access.numel() == slot_to_key.numel(), "slot_last_access must match slot count");
  TORCH_CHECK(slot_frequency.numel() == slot_to_key.numel(), "slot_frequency must match slot count");
  TORCH_CHECK(slot_versions.numel() == slot_to_key.numel(), "slot_versions must match slot count");
  check_key_id(key_id, key_to_slot.numel());

  access_count_ptr[0] += 1;
  const int64_t timestamp = access_count_ptr[0];
  const int64_t slot = key_to_slot_ptr[key_id];
  if (slot < 0) {
    return kEmpty;
  }
  TORCH_CHECK(slot < slot_to_key.numel(), "slot index out of slot table bounds");
  TORCH_CHECK(slot_to_key_ptr[slot] == key_id, "cache slot table is inconsistent");
  if (slot_busy(slot_versions_ptr, slot)) {
    return kEmpty;
  }
  if (policy == kPolicyLru) {
    slot_last_access_ptr[slot] = timestamp;
  } else if (policy == kPolicyLfu) {
    slot_frequency_ptr[slot] += 1;
    slot_last_access_ptr[slot] = timestamp;
  }
  return slot;
}

int64_t cache_allocate_slot(
    int64_t policy,
    int64_t key_id,
    int64_t trace_length,
    bool cyclic,
    torch::Tensor key_to_slot,
    torch::Tensor slot_to_key,
    torch::Tensor access_count,
    torch::Tensor slot_last_access,
    torch::Tensor slot_frequency,
    torch::Tensor slot_insert_order,
    torch::Tensor slot_versions,
    torch::Tensor rr_state,
    torch::Tensor trace_offsets,
    torch::Tensor trace_positions) {
  check_policy(policy);
  int64_t* key_to_slot_ptr = mutable_i64(key_to_slot, "key_to_slot");
  int64_t* slot_to_key_ptr = mutable_i64(slot_to_key, "slot_to_key");
  int64_t* access_count_ptr = mutable_i64(access_count, "access_count");
  int64_t* slot_last_access_ptr = mutable_i64(slot_last_access, "slot_last_access");
  int64_t* slot_frequency_ptr = mutable_i64(slot_frequency, "slot_frequency");
  int64_t* slot_insert_order_ptr = mutable_i64(slot_insert_order, "slot_insert_order");
  const int64_t* slot_versions_ptr = const_i64(slot_versions, "slot_versions");
  int64_t* rr_state_ptr = mutable_i64(rr_state, "rr_state");
  const int64_t key_count = key_to_slot.numel();
  const int64_t slot_count = slot_to_key.numel();
  TORCH_CHECK(access_count.numel() == 1, "access_count must contain exactly one value");
  TORCH_CHECK(rr_state.numel() == 1, "rr_state must contain exactly one value");
  check_slot_metadata(slot_to_key, slot_last_access, slot_frequency, slot_insert_order, slot_versions);
  check_key_id(key_id, key_count);

  const int64_t existing_slot = key_to_slot_ptr[key_id];
  if (existing_slot >= 0) {
    if (slot_busy(slot_versions_ptr, existing_slot)) {
      return kEmpty;
    }
    return existing_slot;
  }

  int64_t slot = find_empty_slot(slot_to_key_ptr, slot_count);
  if (slot >= 0) {
    assign_slot(
        policy,
        key_id,
        slot,
        key_to_slot_ptr,
        slot_to_key_ptr,
        access_count_ptr,
        slot_last_access_ptr,
        slot_frequency_ptr,
        slot_insert_order_ptr);
    return slot;
  }

  if (slot_count <= 0) {
    return kEmpty;
  }

  if (policy == kPolicyLru) {
    slot = select_lru_victim(slot_last_access_ptr, slot_versions_ptr, slot_count);
  } else if (policy == kPolicyFifo) {
    slot = select_fifo_victim(slot_insert_order_ptr, slot_versions_ptr, slot_count);
  } else if (policy == kPolicyLfu) {
    slot = select_lfu_victim(slot_frequency_ptr, slot_insert_order_ptr, slot_versions_ptr, slot_count);
  } else if (policy == kPolicyRr) {
    slot = select_rr_victim(rr_state_ptr, slot_versions_ptr, slot_count);
  } else {
    const int64_t* offsets_ptr = const_i64(trace_offsets, "trace_offsets");
    const int64_t* positions_ptr = const_i64(trace_positions, "trace_positions");
    TORCH_CHECK(trace_offsets.numel() == key_count + 1, "trace_offsets must have key_count + 1 entries");
    int64_t victim_next_use = std::numeric_limits<int64_t>::min();
    slot = select_opt_victim(
        slot_to_key_ptr,
        slot_versions_ptr,
        slot_count,
        access_count_ptr[0],
        trace_length,
        cyclic,
        offsets_ptr,
        positions_ptr,
        key_count,
        &victim_next_use);
    if (slot < 0) {
      return kEmpty;
    }
    const int64_t candidate_next_use = next_use(
        key_id,
        access_count_ptr[0],
        trace_length,
        cyclic,
        offsets_ptr,
        positions_ptr,
        key_count);
    if (candidate_next_use >= victim_next_use) {
      return kEmpty;
    }
  }

  if (slot < 0) {
    return kEmpty;
  }

  assign_slot(
      policy,
      key_id,
      slot,
      key_to_slot_ptr,
      slot_to_key_ptr,
      access_count_ptr,
      slot_last_access_ptr,
      slot_frequency_ptr,
      slot_insert_order_ptr);
  return slot;
}

int64_t cache_size(torch::Tensor slot_to_key) {
  const int64_t* slot_to_key_ptr = const_i64(slot_to_key, "slot_to_key");
  int64_t count = 0;
  const int64_t slot_count = slot_to_key.numel();
  for (int64_t slot = 0; slot < slot_count; ++slot) {
    if (slot_to_key_ptr[slot] != kEmpty) {
      count += 1;
    }
  }
  return count;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_slot", &cache_get_slot, "Record access and return a cache slot");
  m.def("allocate_slot", &cache_allocate_slot, "Allocate a cache slot for any PCVR cache policy");
  m.def("size", &cache_size, "Return the number of occupied cache slots");
}

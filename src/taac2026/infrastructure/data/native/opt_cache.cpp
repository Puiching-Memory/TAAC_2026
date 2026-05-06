#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <string>

namespace {

constexpr int64_t kEmpty = -1;
constexpr int64_t kNever = std::numeric_limits<int64_t>::max() / 4;

void check_int64_cpu_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.dtype() == torch::kInt64, name, " must be an int64 tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int64_t* mutable_i64(torch::Tensor& tensor, const char* name) {
  check_int64_cpu_tensor(tensor, name);
  return tensor.data_ptr<int64_t>();
}

void check_key_id(int64_t key_id, int64_t trace_length) {
  TORCH_CHECK(trace_length > 0, "trace_length must be positive");
  TORCH_CHECK(key_id >= 0 && key_id < trace_length, "key_id out of trace bounds");
}

int64_t next_use(int64_t key_id, int64_t access_count, int64_t trace_length, bool cyclic) {
  if (key_id < 0 || trace_length <= 0) {
    return kNever;
  }
  const int64_t cycle_index = access_count / trace_length;
  const int64_t cycle_position = access_count % trace_length;
  if (key_id >= cycle_position) {
    return cycle_index * trace_length + key_id;
  }
  if (cyclic) {
    return (cycle_index + 1) * trace_length + key_id;
  }
  return kNever;
}

void touch_slot(int64_t slot, torch::Tensor& last_access, torch::Tensor& touch_counter) {
  int64_t* last_access_ptr = mutable_i64(last_access, "last_access");
  int64_t* touch_counter_ptr = mutable_i64(touch_counter, "touch_counter");
  TORCH_CHECK(touch_counter.numel() == 1, "touch_counter must contain exactly one value");
  touch_counter_ptr[0] += 1;
  last_access_ptr[slot] = touch_counter_ptr[0];
}

}  // namespace

int64_t opt_cache_get_slot(
    int64_t key_id,
    int64_t trace_length,
    bool cyclic,
    torch::Tensor key_to_slot,
    torch::Tensor last_access,
    torch::Tensor touch_counter,
    torch::Tensor access_count) {
  (void)cyclic;
  check_key_id(key_id, trace_length);
  int64_t* key_to_slot_ptr = mutable_i64(key_to_slot, "key_to_slot");
  int64_t* access_count_ptr = mutable_i64(access_count, "access_count");
  TORCH_CHECK(key_to_slot.numel() == trace_length, "key_to_slot length must match trace_length");
  TORCH_CHECK(access_count.numel() == 1, "access_count must contain exactly one value");

  access_count_ptr[0] += 1;
  const int64_t slot = key_to_slot_ptr[key_id];
  if (slot >= 0) {
    touch_slot(slot, last_access, touch_counter);
  }
  return slot;
}

int64_t opt_cache_allocate_slot(
    int64_t key_id,
    int64_t trace_length,
    bool cyclic,
    torch::Tensor key_to_slot,
    torch::Tensor slot_to_key,
    torch::Tensor last_access,
    torch::Tensor touch_counter,
    torch::Tensor access_count) {
  check_key_id(key_id, trace_length);
  int64_t* key_to_slot_ptr = mutable_i64(key_to_slot, "key_to_slot");
  int64_t* slot_to_key_ptr = mutable_i64(slot_to_key, "slot_to_key");
  int64_t* last_access_ptr = mutable_i64(last_access, "last_access");
  int64_t* access_count_ptr = mutable_i64(access_count, "access_count");
  TORCH_CHECK(key_to_slot.numel() == trace_length, "key_to_slot length must match trace_length");
  TORCH_CHECK(last_access.numel() == slot_to_key.numel(), "last_access length must match slot_to_key length");

  const int64_t existing_slot = key_to_slot_ptr[key_id];
  if (existing_slot >= 0) {
    touch_slot(existing_slot, last_access, touch_counter);
    return existing_slot;
  }

  const int64_t slot_count = slot_to_key.numel();
  for (int64_t slot = 0; slot < slot_count; ++slot) {
    if (slot_to_key_ptr[slot] == kEmpty) {
      slot_to_key_ptr[slot] = key_id;
      key_to_slot_ptr[key_id] = slot;
      touch_slot(slot, last_access, touch_counter);
      return slot;
    }
  }

  int64_t victim_slot = kEmpty;
  int64_t victim_key = kEmpty;
  int64_t farthest_next_use = std::numeric_limits<int64_t>::min();
  const int64_t current_access_count = access_count_ptr[0];
  for (int64_t slot = 0; slot < slot_count; ++slot) {
    const int64_t cached_key = slot_to_key_ptr[slot];
    const int64_t cached_next_use = next_use(cached_key, current_access_count, trace_length, cyclic);
    if (cached_next_use > farthest_next_use) {
      farthest_next_use = cached_next_use;
      victim_slot = slot;
      victim_key = cached_key;
    }
  }
  TORCH_CHECK(victim_slot >= 0, "OPT cache eviction requested for an empty slot table");

  const int64_t candidate_next_use = next_use(key_id, current_access_count, trace_length, cyclic);
  if (candidate_next_use >= farthest_next_use) {
    return kEmpty;
  }

  if (victim_key >= 0) {
    key_to_slot_ptr[victim_key] = kEmpty;
  }
  slot_to_key_ptr[victim_slot] = key_id;
  key_to_slot_ptr[key_id] = victim_slot;
  last_access_ptr[victim_slot] = 0;
  touch_slot(victim_slot, last_access, touch_counter);
  return victim_slot;
}

int64_t opt_cache_size(torch::Tensor slot_to_key) {
  int64_t* slot_to_key_ptr = mutable_i64(slot_to_key, "slot_to_key");
  int64_t count = 0;
  const int64_t slot_count = slot_to_key.numel();
  for (int64_t slot = 0; slot < slot_count; ++slot) {
    if (slot_to_key_ptr[slot] != kEmpty) {
      count += 1;
    }
  }
  return count;
}

void opt_cache_clear(
    torch::Tensor key_to_slot,
    torch::Tensor slot_to_key,
    torch::Tensor last_access,
    torch::Tensor touch_counter,
    torch::Tensor access_count) {
  key_to_slot.fill_(kEmpty);
  slot_to_key.fill_(kEmpty);
  last_access.zero_();
  touch_counter.zero_();
  access_count.zero_();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_slot", &opt_cache_get_slot, "Record access and return an OPT cache slot");
  m.def("allocate_slot", &opt_cache_allocate_slot, "Allocate an OPT cache slot for a missed key");
  m.def("size", &opt_cache_size, "Return the number of occupied OPT cache slots");
  m.def("clear", &opt_cache_clear, "Clear OPT cache index tensors");
}

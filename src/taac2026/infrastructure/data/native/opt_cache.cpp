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

}  // namespace

int64_t opt_cache_get_slot(
    int64_t key_id,
    int64_t trace_length,
    bool cyclic,
    torch::Tensor key_to_slot,
    torch::Tensor access_count) {
  (void)cyclic;
  check_key_id(key_id, trace_length);
  int64_t* key_to_slot_ptr = mutable_i64(key_to_slot, "key_to_slot");
  int64_t* access_count_ptr = mutable_i64(access_count, "access_count");
  TORCH_CHECK(key_to_slot.numel() == trace_length, "key_to_slot length must match trace_length");
  TORCH_CHECK(access_count.numel() == 1, "access_count must contain exactly one value");

  access_count_ptr[0] += 1;
  return key_to_slot_ptr[key_id];
}

int64_t opt_cache_allocate_slot(
    int64_t key_id,
    int64_t trace_length,
    bool cyclic,
    torch::Tensor key_to_slot,
    torch::Tensor slot_to_key,
    torch::Tensor access_count) {
  check_key_id(key_id, trace_length);
  int64_t* key_to_slot_ptr = mutable_i64(key_to_slot, "key_to_slot");
  int64_t* slot_to_key_ptr = mutable_i64(slot_to_key, "slot_to_key");
  int64_t* access_count_ptr = mutable_i64(access_count, "access_count");
  TORCH_CHECK(key_to_slot.numel() == trace_length, "key_to_slot length must match trace_length");
  TORCH_CHECK(access_count.numel() == 1, "access_count must contain exactly one value");

  const int64_t existing_slot = key_to_slot_ptr[key_id];
  if (existing_slot >= 0) {
    return existing_slot;
  }

  const int64_t slot_count = slot_to_key.numel();
  const int64_t current_access_count = access_count_ptr[0];
  int64_t victim_slot = kEmpty;
  int64_t victim_key = kEmpty;
  int64_t farthest_next_use = std::numeric_limits<int64_t>::min();
  for (int64_t slot = 0; slot < slot_count; ++slot) {
    const int64_t cached_key = slot_to_key_ptr[slot];
    if (cached_key == kEmpty) {
      slot_to_key_ptr[slot] = key_id;
      key_to_slot_ptr[key_id] = slot;
      return slot;
    }

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
    torch::Tensor access_count) {
  key_to_slot.fill_(kEmpty);
  slot_to_key.fill_(kEmpty);
  access_count.zero_();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_slot", &opt_cache_get_slot, "Record access and return an OPT cache slot");
  m.def("allocate_slot", &opt_cache_allocate_slot, "Allocate an OPT cache slot for a missed key");
  m.def("size", &opt_cache_size, "Return the number of occupied OPT cache slots");
  m.def("clear", &opt_cache_clear, "Clear OPT cache index tensors");
}

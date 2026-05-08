// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// This file contains a TAAC project-local rewrite of NVIDIA cuEmbed's
// fixed-hotness embedding lookup idea. It is intentionally scoped to the
// embedding_bag_mean(weight, values[B, hotness]) operation used in this repo:
// one CUDA table, fixed hotness, unweighted mean, and padding id 0 ignored.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

namespace {

constexpr int kThreadsPerBlock = 128;

template <typename scalar_t, typename index_t>
__global__ void taac_cuembed_embedding_bag_mean_kernel(
    const scalar_t* __restrict__ weight,
    const index_t* __restrict__ values,
    const int64_t batch_size,
    const int64_t bag_size,
    const int64_t num_embeddings,
    const int64_t emb_dim,
    scalar_t* __restrict__ output) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t col = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (row >= batch_size || col >= emb_dim) {
    return;
  }

  float accum = 0.0f;
  int valid_count = 0;
  const int64_t value_offset = row * bag_size;
  for (int64_t hot = 0; hot < bag_size; ++hot) {
    const int64_t index = static_cast<int64_t>(values[value_offset + hot]);
    if (index > 0 && index < num_embeddings) {
      accum += static_cast<float>(weight[index * emb_dim + col]);
      ++valid_count;
    }
  }

  const float result = valid_count > 0 ? accum / static_cast<float>(valid_count) : 0.0f;
  output[row * emb_dim + col] = static_cast<scalar_t>(result);
}

template <typename scalar_t>
void dispatch_index_type(
    const torch::Tensor& weight,
    const torch::Tensor& values,
    torch::Tensor& output) {
  const int64_t batch_size = values.size(0);
  const int64_t bag_size = values.size(1);
  const int64_t num_embeddings = weight.size(0);
  const int64_t emb_dim = weight.size(1);
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(batch_size, (emb_dim + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (values.scalar_type() == torch::kInt32) {
    taac_cuembed_embedding_bag_mean_kernel<scalar_t, int32_t><<<grid, block, 0, stream>>>(
        weight.data_ptr<scalar_t>(),
        values.data_ptr<int32_t>(),
        batch_size,
        bag_size,
        num_embeddings,
        emb_dim,
        output.data_ptr<scalar_t>());
  } else if (values.scalar_type() == torch::kInt64) {
    taac_cuembed_embedding_bag_mean_kernel<scalar_t, int64_t><<<grid, block, 0, stream>>>(
        weight.data_ptr<scalar_t>(),
        values.data_ptr<int64_t>(),
        batch_size,
        bag_size,
        num_embeddings,
        emb_dim,
        output.data_ptr<scalar_t>());
  } else {
    TORCH_CHECK(false, "cuembed embedding_bag_mean values must be int32 or int64");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor taac_cuembed_embedding_bag_mean_forward(torch::Tensor weight, torch::Tensor values) {
  TORCH_CHECK(weight.is_cuda(), "cuembed embedding_bag_mean weight must be CUDA");
  TORCH_CHECK(values.is_cuda(), "cuembed embedding_bag_mean values must be CUDA");
  TORCH_CHECK(weight.is_contiguous(), "cuembed embedding_bag_mean weight must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "cuembed embedding_bag_mean values must be contiguous");
  TORCH_CHECK(weight.dim() == 2, "cuembed embedding_bag_mean weight must be 2D");
  TORCH_CHECK(values.dim() == 2, "cuembed embedding_bag_mean values must be 2D");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32 || weight.scalar_type() == torch::kFloat16,
              "cuembed embedding_bag_mean supports float32 and float16 weights");
  TORCH_CHECK(values.scalar_type() == torch::kInt32 || values.scalar_type() == torch::kInt64,
              "cuembed embedding_bag_mean values must be int32 or int64");

  auto output = torch::empty({values.size(0), weight.size(1)}, weight.options());
  if (values.size(0) == 0 || weight.size(1) == 0) {
    return output;
  }

  if (weight.scalar_type() == torch::kFloat32) {
    dispatch_index_type<float>(weight, values, output);
  } else {
    dispatch_index_type<at::Half>(weight, values, output);
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "embedding_bag_mean_forward",
      &taac_cuembed_embedding_bag_mean_forward,
      "TAAC cuEmbed-style fixed-hotness embedding_bag_mean forward");
}

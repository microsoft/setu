//==============================================================================
// Copyright 2025 Vajra Team; Georgia Institute of Technology; Microsoft
// Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//==============================================================================
#pragma once
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "commons/Types.h"
//==============================================================================
// To allow using CUDA IPC types and functions from torch
#define USE_CUDA
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
using setu::commons::BinaryBuffer;
using setu::commons::BinaryRange;
//==============================================================================
struct TensorIPCSpec final {
  // Constructor accepting IntArrayRef (copies data into vectors)
  TensorIPCSpec(torch::IntArrayRef tensor_size_param,
                torch::IntArrayRef tensor_stride_param,
                std::int64_t tensor_offset_param, torch::Dtype dtype_param,
                bool requires_grad_param, std::int32_t storage_device_param,
                std::string storage_handle_param,
                std::uint64_t storage_size_bytes_param,
                std::uint64_t storage_offset_bytes_param,
                std::string ref_counter_handle_param,
                std::uint64_t ref_counter_offset_param,
                cudaIpcEventHandle_t event_handle_param,
                bool event_sync_required_param);

  // Accessors returning IntArrayRef views (for compatibility with torch APIs)
  [[nodiscard]] torch::IntArrayRef GetTensorSize() const;
  [[nodiscard]] torch::IntArrayRef GetTensorStride() const;

  // Serialization methods
  void Serialize(BinaryBuffer& buffer) const;
  static TensorIPCSpec Deserialize(const BinaryRange& range);

  const std::vector<std::int64_t> tensor_size;
  const std::vector<std::int64_t> tensor_stride;
  const std::int64_t tensor_offset;
  const torch::Dtype dtype;
  const bool requires_grad;
  const std::int32_t storage_device;
  const std::string storage_handle;
  const std::uint64_t storage_size_bytes;
  const std::uint64_t storage_offset_bytes;
  const std::string ref_counter_handle;
  const std::uint64_t ref_counter_offset;
  const cudaIpcEventHandle_t event_handle;
  const bool event_sync_required;
};
using TensorIPCSpecPtr = std::shared_ptr<TensorIPCSpec>;
//==============================================================================
TensorIPCSpec PrepareTensorIPCSpec(const torch::Tensor& x);
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
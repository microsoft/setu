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
#include "commons/TorchCommon.h"
#include "commons/StdCommon.h"
//==============================================================================
// To allow using CUDA IPC types and functions from torch
#define USE_CUDA
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
struct TensorIPCSpec final {
    torch::IntArrayRef tensor_size;
    torch::IntArrayRef tensor_stride;
    std::int64_t tensor_offset;
    torch::Dtype dtype;
    bool requires_grad;
    std::int32_t storage_device;
    std::string storage_handle;
    std::uint64_t storage_size_bytes;
    std::uint64_t storage_offset_bytes;
    std::string ref_counter_handle;
    std::uint64_t ref_counter_offset;
    cudaIpcEventHandle_t event_handle;
    bool event_sync_required;

    TensorIPCSpec(
        torch::IntArrayRef tensor_size_param,
        torch::IntArrayRef tensor_stride_param,
        std::int64_t tensor_offset_param,
        torch::Dtype dtype_param,
        bool requires_grad_param,
        std::int32_t storage_device_param,
        std::string storage_handle_param,
        std::uint64_t storage_size_bytes,
        std::uint64_t storage_offset_bytes_param,
        std::string ref_counter_handle_param,
        std::uint64_t ref_counter_offset_param,
        cudaIpcEventHandle_t event_handle_param,
        bool event_sync_required_param);
};
using TensorIPCSpecPtr = std::shared_ptr<TensorIPCSpec>;
//==============================================================================
TensorIPCSpec PrepareTensorIPCSpec(const torch::Tensor& x);
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
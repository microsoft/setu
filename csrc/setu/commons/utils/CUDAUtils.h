//==============================================================================
// Copyright (c) 2025 Vajra Team; Georgia Institute of Technology; Microsoft
// Corporation.
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
#include <cuda_runtime_api.h>
#include <nccl.h>
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================

// clang-format off
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = (call);                                           \
    ASSERT_VALID_RUNTIME(err == cudaSuccess, "CUDA error: {} at {}:{}", \
                         cudaGetErrorString(err), __FILE__, __LINE__);  \
  } while (0)

#define NCCL_CHECK(call)                                                \
  do {                                                                  \
    ncclResult_t res = (call);                                          \
    ASSERT_VALID_RUNTIME(res == ncclSuccess, "NCCL error: {} at {}:{}", \
                         ncclGetErrorString(res), __FILE__, __LINE__);  \
  } while (0)
// clang-format on

//==============================================================================
namespace setu::commons::utils {
//==============================================================================
struct CudaComputeCapability {
  int major;
  int minor;
};
//==============================================================================
[[nodiscard]] CudaComputeCapability GetCudaComputeCapability(
    torch::Device device /*[in]*/);
//==============================================================================
[[nodiscard]] int GetCudaRuntimeVersion();
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================

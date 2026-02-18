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
#include "node_manager/worker/RegisterFile.h"
//==============================================================================
#include <cuda_runtime.h>
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================

#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = (call);                                           \
    ASSERT_VALID_RUNTIME(err == cudaSuccess, "CUDA error: {} at {}:{}", \
                         cudaGetErrorString(err), __FILE__, __LINE__);  \
  } while (0)

void RegisterFile::Allocate() {
  for (std::uint32_t i = 0; i < spec_.NumRegisters(); ++i) {
    ASSERT_VALID_RUNTIME(ptrs_[i] == nullptr, "Register {} already allocated",
                         i);
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, spec_.SizeBytes(i)));
    ptrs_[i] = ptr;
  }

  LOG_DEBUG("Allocated {} registers", spec_.NumRegisters());
}

void RegisterFile::Free() {
  for (auto& ptr : ptrs_) {
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  }
}

#undef CUDA_CHECK

//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================

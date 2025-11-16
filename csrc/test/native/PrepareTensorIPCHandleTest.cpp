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
#include <gtest/gtest.h>
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/utils/TorchTensorIPC.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
namespace {
//==============================================================================
TEST(TensorIPC, PrepareIPCHandleTest) {
    // Create a tensor on the GPU
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(torch::kCUDA, 0)
    ;
    auto a = torch::randn({3,3}, options);

    // Prepare IPC spec
    auto spec = setu::commons::utils::PrepareTensorIPCSpec(a);
    std::cout << "STORAGE HANDLE" << spec.storage_handle << std::endl;
}
//==============================================================================
}  // namespace
//==============================================================================
}  // namespace setu::test::native
//==============================================================================

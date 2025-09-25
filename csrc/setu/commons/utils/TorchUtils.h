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
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
[[nodiscard]] at::ScalarType GetScalarTypeFromString(
    const std::string& type /*[in]*/);
//==============================================================================
void CheckShapeDtypeAndDevice(
    const torch::Tensor& x /*[in]*/,
    const std::vector<int64_t>& expected_shape /*[in]*/,
    at::ScalarType expected_dtype /*[in]*/,
    torch::Device expected_device /*[in]*/, const std::string& name /*[in]*/
);
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================

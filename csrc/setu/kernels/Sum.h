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
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
namespace setu::kernels {
//==============================================================================
/// @brief Computes Sum of the input tensor
///
/// This function sums the input tensor.
///
/// @param output Output tensor
/// @param input Input tensor
/// @throws std::runtime_error if tensor Dims are incompatible
void Sum(torch::Tensor& output /*[out]*/, const torch::Tensor& input /*[in]*/);
//==============================================================================
}  // namespace setu::kernels
//==============================================================================

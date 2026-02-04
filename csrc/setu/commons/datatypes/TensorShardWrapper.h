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
#include "commons/Types.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
/**
 * @brief Setu's internal tensor representation
 *  Wraps types such as torch::Tensor and provides access utility functions
 */
struct TensorShardWrapper {
  TensorShardWrapper(torch::Tensor tensor_param)
      : tensor(std::move(tensor_param)) {
    if (!tensor.defined() || tensor.numel() <= 0)
      RAISE_ERROR(std::invalid_argument, "Invalid tensor argument",
                  "{} is not defined", tensor);
  }

  /**
   * @brief Get read-only pointer to device memory
   *
   * @return Const pointer to device memory
   */
  [[nodiscard]] DevicePtr GetDevicePtr() const { return tensor.data_ptr(); }

  [[nodiscard]] torch::Tensor GetTorchTensor() const { return tensor; }

  /**
   * @brief Returns a string representation of the tensor shard
   *
   * @return String containing metadata and device pointer
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("Tensor={})", tensor);
  }

  torch::Tensor tensor;
};
//==============================================================================
/// @brief Allows multiple ptrs to the same wrapper
using TensorShardWrapperPtr = std::shared_ptr<TensorShardWrapper>;

/// @brief Lookup tensor shard given shard id
using TensorShardsConcurrentMap = ConcurrentMap<ShardId, TensorShardWrapperPtr>;

//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

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
#include "commons/Types.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
/**
 * @brief Represents a named dimension of a tensor with its size
 *
 * TensorDim defines a single dimension of a tensor, including its name
 * and size. This is used to describe the shape and structure of tensors
 * within the Setu system.
 */
struct TensorDim {
  /**
   * @brief Constructs a tensor dimension with the specified name and size
   *
   * @param name_param Name of the dimension (e.g., "batch", "sequence",
   * "hidden")
   * @param size_param Size of the dimension, must be greater than 0
   *
   * @throws std::invalid_argument if size_param is 0
   */
  TensorDim(TensorDimName name_param, std::size_t size_param)
      : name(name_param), size(size_param) {
    ASSERT_VALID_ARGUMENTS(size_param > 0, "Size {} must be greater than 0",
                           size_param);
  }

  /**
   * @brief Returns a string representation of the tensor dimension
   *
   * @return String containing the dimension name and size
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("TensorDim(name={}, size={})", name, size);
  }

  const TensorDimName name;  ///< Name of the tensor dimension
  const std::size_t size;    ///< Size of the tensor dimension
};
//==============================================================================
/// @brief Shared pointer to a map of tensor dimension names to TensorDim
/// objects
using TensorDimMap =
    std::shared_ptr<std::unordered_map<TensorDimName, TensorDim>>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

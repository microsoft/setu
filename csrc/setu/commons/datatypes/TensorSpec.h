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
#include "commons/datatypes/TensorDim.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::datatypes {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================
/**
 * @brief Specification describing a tensor's name, dimensions, and data type
 *
 * TensorSpec is a lightweight descriptor for a tensor's shape and type,
 * without any sharding, device, or ownership information. It contains
 * the minimal information needed to construct a TensorSelection.
 */
struct TensorSpec {
  /**
   * @brief Constructs a tensor specification
   *
   * @param name_param Name/identifier for the tensor
   * @param dims_param Map of dimension names to TensorDim objects
   * @param dtype_param Data type of the tensor elements
   *
   * @throws std::invalid_argument if dims is empty
   */
  TensorSpec(TensorName name_param, TensorDimMap dims_param,
             torch::Dtype dtype_param)
      : name(std::move(name_param)),
        dims(std::move(dims_param)),
        dtype(dtype_param) {
    ASSERT_VALID_ARGUMENTS(!dims.empty(), "Dims must be non-empty");
  }

  /**
   * @brief Returns a string representation of the tensor spec
   *
   * @return String containing tensor name, dimensions, and data type
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("TensorSpec(name={}, dims={}, dtype={})", name, dims,
                       dtype);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static TensorSpec Deserialize(const BinaryRange& range);

  const TensorName name;     ///< Name/identifier for the tensor
  const TensorDimMap dims;   ///< Map of dimension names to TensorDim objects
  const torch::Dtype dtype;  ///< Data type of tensor elements
};
//==============================================================================
using TensorSpecPtr = std::shared_ptr<TensorSpec>;
using TensorSpecMap = std::unordered_map<TensorName, TensorSpec>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

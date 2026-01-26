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
#include "commons/Types.h"
#include "commons/datatypes/Device.h"
#include "commons/datatypes/TensorDim.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::datatypes {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================
/**
 * @brief Specification for creating a tensor shard
 *
 * TensorShardSpec contains the user-provided specification for a tensor shard:
 * - name: identifier for the tensor
 * - dims: list of dimensions with names and sizes
 * - dtype: data type of tensor elements
 * - device: device where the tensor resides
 *
 * This is the simplified input from the client. The coordinator will use this
 * to create the full TensorShard with generated UUIDs and additional metadata.
 */
struct TensorShardSpec {
  /**
   * @brief Constructs a tensor shard specification
   *
   * @param name_param Name/identifier for the tensor
   * @param dims_param Vector of TensorDim specifying each dimension
   * @param dtype_param Data type of the tensor elements
   * @param device_param Device where this tensor resides
   *
   * @throws std::invalid_argument if dims is empty
   */
  TensorShardSpec(TensorName name_param, std::vector<TensorDim> dims_param,
                  torch::Dtype dtype_param, Device device_param)
      : name(std::move(name_param)),
        dims(std::move(dims_param)),
        dtype(dtype_param),
        device(device_param) {
    ASSERT_VALID_ARGUMENTS(!dims.empty(), "Dims must be non-empty");
  }

  /**
   * @brief Calculates the total number of elements in the tensor
   *
   * @return Total number of elements (product of all dimension sizes)
   */
  [[nodiscard]] std::size_t GetNumElements() const {
    std::size_t num_elements = 1;
    for (const auto& dim : dims) {
      num_elements *= dim.size;
    }
    return num_elements;
  }

  /**
   * @brief Returns the number of dimensions
   *
   * @return Number of tensor dimensions
   */
  [[nodiscard]] std::size_t GetNumDims() const { return dims.size(); }

  /**
   * @brief Returns a string representation of the tensor shard spec
   *
   * @return String containing all spec properties
   */
  [[nodiscard]] std::string ToString() const {
    std::string dims_str = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
      if (i > 0) dims_str += ", ";
      dims_str += dims[i].ToString();
    }
    dims_str += "]";
    return std::format("TensorShardSpec(name={}, dims={}, dtype={}, device={})",
                       name, dims_str, dtype, device);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static TensorShardSpec Deserialize(const BinaryRange& range);

  const TensorName name;              ///< Name/identifier for the tensor
  const std::vector<TensorDim> dims;  ///< Dimensions of the tensor
  const torch::Dtype dtype;           ///< Data type of tensor elements
  const Device device;                ///< Device where tensor resides
};
//==============================================================================
/// @brief Shared pointer to a TensorShardSpec object
using TensorShardSpecPtr = std::shared_ptr<TensorShardSpec>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

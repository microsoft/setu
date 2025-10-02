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
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/TensorDim.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
/**
 * @brief Lightweight reference to a tensor shard
 *
 * TensorShardRef provides a handle to a tensor shard without containing the
 * actual device memory pointer or full shard information. It contains only
 * the metadata needed to identify and describe the shard's structure.
 */
struct TensorShardRef {
  /**
   * @brief Constructs a tensor shard reference
   *
   * @param name_param Name of the tensor being sharded
   * @param shard_id_param UUID identifier for this shard
   * @param dims_param Map of dimension names to TensorDim objects
   *
   * @throws std::invalid_argument if dims is empty
   */
  TensorShardRef(TensorName name_param, ShardId shard_id_param,
                 TensorDimMap dims_param)
      : name(name_param), shard_id(shard_id_param), dims(dims_param) {
    ASSERT_VALID_ARGUMENTS(!name_param.empty(), "Tensor name cannot be empty");
    ASSERT_VALID_ARGUMENTS(!shard_id_param.is_nil(),
                           "Shard ID cannot be nil UUID");
    ASSERT_VALID_ARGUMENTS(dims_param.size() > 0, "Dims must be non-empty");
  }

  /**
   * @brief Returns a string representation of the tensor shard reference
   *
   * @return String containing name, shard ID, and dimensions
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("TensorShardRef(name={}, shard_id={}, dims={})", name,
                       shard_id, dims);
  }

  /**
   * @brief Returns the number of dimensions in this shard
   *
   * @return Number of tensor dimensions
   */
  [[nodiscard]] std::size_t GetNumDims() const { return dims.size(); }

  const TensorName name;    ///< Name of the tensor being sharded
  const ShardId shard_id;   ///< UUID identifier for this shard
  const TensorDimMap dims;  ///< Map of dimension names to TensorDim objects
};
//==============================================================================
/// @brief Shared pointer to a TensorShardRef object
using TensorShardRefPtr = std::shared_ptr<TensorShardRef>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

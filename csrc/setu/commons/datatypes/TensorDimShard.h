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
#include "commons/datatypes/TensorSlice.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
/**
 * @brief Represents a shard of a tensor dimension across distributed devices
 *
 * TensorDimShard describes how a single dimension of a tensor is distributed
 * across devices, including the device location, size information, and the
 * specific slice of data that this shard contains.
 */
struct TensorDimShard {
  /**
   * @brief Constructs a tensor dimension shard with all required parameters
   *
   * @param name_param Name of the tensor dimension being sharded
   * @param shard_id_param Unique identifier for this shard
   * @param dim_size_param Total size of the original dimension
   * @param slice_param Slice specification defining which part of the dimension
   * this shard contains
   * @param stride_param Memory stride for accessing elements in this shard
   *
   * @throws std::invalid_argument if any size parameter is 0 or if any pointer
   * is null
   */
  TensorDimShard(TensorDimName name_param, ShardId shard_id_param,
                 std::size_t dim_size_param, TensorSlicePtr slice_param,
                 std::size_t stride_param)
      : name(name_param),
        shard_id(shard_id_param),
        dim_size(dim_size_param),
        shard_size(slice_param->size),
        slice(slice_param),
        stride(stride_param) {
    ASSERT_VALID_ARGUMENTS(dim_size_param > 0,
                           "Dim size {} must be greater than 0",
                           dim_size_param);
    ASSERT_VALID_POINTER_ARGUMENT(slice_param);
    ASSERT_VALID_ARGUMENTS(stride_param > 0, "Stride {} must be greater than 0",
                           stride_param);
  }

  /**
   * @brief Returns a string representation of the tensor dimension shard
   *
   * @return String containing all shard properties including name, device,
   * sizes, slice, and stride
   */
  [[nodiscard]] std::string ToString() const {
    return std::format(
        "TensorDimShard(name={}, dim_size={}, shard_size={}, "
        "slice={}, stride={})",
        name, dim_size, shard_size, slice, stride);
  }

  const TensorDimName name;      ///< Name of the tensor dimension
  const ShardId shard_id;        ///< Unique identifier for this shard
  const std::size_t dim_size;    ///< Total size of the original dimension
  const std::size_t shard_size;  ///< Size of this specific shard
  const TensorSlicePtr slice;    ///< Slice specification for this shard
  const std::size_t stride;      ///< Memory stride for accessing elements
};
//==============================================================================
/// @brief Shared pointer to a map of dimension names to TensorDimShard objects
using TensorDimShardsMap =
    std::shared_ptr<std::unordered_map<TensorDimName, TensorDimShard>>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

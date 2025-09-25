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
#include "commons/Types.h"
#include "commons/datatypes/Device.h"
#include "commons/datatypes/TensorDimShard.h"
#include "commons/enums/Enums.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
// Type aliases for convenience
using setu::commons::enums::DType;
//==============================================================================
/**
 * @brief Represents a shard of a tensor distributed across devices
 *
 * TensorShard encapsulates a portion of a tensor that resides on a specific
 * device, including information about which dimensions are sharded and how they
 * are sliced. This enables distributed tensor operations across multiple
 * devices.
 */
struct TensorShard {
  /**
   * @brief Constructs a tensor shard with all required parameters
   *
   * @param name_param Name of the tensor being sharded
   * @param id_param Unique identifier for this shard
   * @param device_param Device information where this shard resides
   * @param device_ptr_param Pointer to the device memory location
   * @param dtype_param Data type of the tensor elements
   * @param dim_shards_param Map of dimension names to their shard information
   *
   * @throws std::invalid_argument if id is 0, device_ptr is null, dim_shards is
   * null or empty
   */
  TensorShard(TensorName name_param, ShardId id_param, Device device_param,
              DevicePtr device_ptr_param, DType dtype_param,
              TensorDimShardsMap dim_shards_param)
      : id(id_param),
        name(name_param),
        device(device_param),
        device_ptr(device_ptr_param),
        dtype(dtype_param),
        dim_shards(dim_shards_param),
        shard_size(GetShardSize()) {
    ASSERT_VALID_ARGUMENTS(id_param > 0, "Shard ID {} must be greater than 0",
                           id_param);
    ASSERT_VALID_POINTER_ARGUMENT(device_ptr_param);
    ASSERT_VALID_POINTER_ARGUMENT(dim_shards_param);
    ASSERT_VALID_ARGUMENTS(dim_shards_param->size() > 0,
                           "Dim shards must be non-empty");
  }

  /**
   * @brief Calculates the total number of elements in the shard
   *
   * Computes the product of all dimension sizes to determine the total
   * number of elements that would be contained in a shard of this shape.
   *
   * @return Total number of elements across all dimensions
   */
  [[nodiscard]] std::size_t GetShardSize() const {
    std::size_t size = 1;
    for (const auto& [_, dim_shard] : *dim_shards) {
      size *= dim_shard.shard_size;
    }
    return size;
  }

  /**
   * @brief Returns a string representation of the tensor shard
   *
   * @return String containing all shard properties including ID, name, device,
   * and dimensions
   */
  [[nodiscard]] std::string ToString() const {
    return std::format(
        "TensorShard(id={}, name={}, device={}, device_ptr={}, dtype={}, "
        "dim_shards={}, shard_size={})",
        id, name, device, device_ptr, dtype, dim_shards, shard_size);
  }

  /**
   * @brief Returns the number of dimensions in this shard
   *
   * @return Number of tensor dimensions represented in this shard
   */
  [[nodiscard]] std::size_t GetNumDims() const { return dim_shards->size(); }

  /**
   * @brief Retrieves the slice information for a specific dimension
   *
   * @param dim_name Name of the dimension to get slice for
   * @return Shared pointer to the TensorSlice for the specified dimension
   *
   * @throws std::invalid_argument if dimension name is not found in this shard
   */
  [[nodiscard]] TensorSlicePtr GetDimSlice(
      const TensorDimName& dim_name) const {
    ASSERT_VALID_ARGUMENTS(dim_shards->find(dim_name) != dim_shards->end(),
                           "Dim {} not found", dim_name);
    return dim_shards->at(dim_name).slice;
  }

  const ShardId id;            ///< Unique identifier for this shard
  const TensorName name;       ///< Name of the tensor being sharded
  const Device device;         ///< Device where this shard resides
  const DevicePtr device_ptr;  ///< Pointer to device memory location
  const DType dtype;           ///< Data type of tensor elements
  const TensorDimShardsMap
      dim_shards;                ///< Map of dimension names to shard info
  const std::size_t shard_size;  ///< Size of this specific shard
};
//==============================================================================
/// @brief Shared pointer to a TensorShard object
using TensorShardPtr = std::shared_ptr<TensorShard>;

/// @brief Shared pointer to a map of shard IDs to TensorShard objects
using TensorShardsMap =
    std::shared_ptr<std::unordered_map<ShardId, TensorShardPtr>>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

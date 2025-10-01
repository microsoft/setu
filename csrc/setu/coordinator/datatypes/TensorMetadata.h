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
#include "commons/ClassTraits.h"
#include "commons/StdCommon.h"
#include "commons/datatypes/TensorDim.h"
#include "commons/datatypes/TensorSelection.h"
#include "commons/datatypes/TensorShard.h"
#include "commons/enums/Enums.h"
#include "coordinator/datatypes/TensorOwnershipMap.h"
#include "coordinator/datatypes/TensorShardUtils.h"
//==============================================================================
namespace setu::coordinator::datatypes {
//==============================================================================
// Type aliases for convenience
using DType = setu::commons::enums::DType;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorDimMap;
using setu::commons::datatypes::TensorSelectionPtr;
using setu::commons::datatypes::TensorShardsMap;
//==============================================================================
/**
 * @brief Complete metadata for a tensor including dimensions, data type, and
 * sharding information
 *
 * TensorMetadata represents the complete description of a tensor within the
 * Setu system, including its dimensions, data type, and how it is distributed
 * across devices through shards. It provides validation to ensure that the
 * shards completely cover the tensor space.
 */
struct TensorMetadata {
  /**
   * @brief Constructs tensor metadata with all required information
   *
   * @param name_param Name of the tensor
   * @param dims_param Map of dimension names to TensorDim objects describing
   * the tensor shape
   * @param dtype_param Data type of the tensor elements
   * @param shards_param Map of node IDs to tensor shards distributed across
   * devices
   *
   * @throws std::invalid_argument if dims or shards are null, empty, or if
   * shards don't fully cover the tensor
   */
  TensorMetadata(TensorName name_param, TensorDimMap dims_param,
                 DType dtype_param, TensorShardsMap shards_param)
      : name(name_param),
        dims(dims_param),
        dtype(dtype_param),
        shards(shards_param),
        size(GetSize()) {
    ASSERT_VALID_ARGUMENTS(dims_param.size() > 0, "Dims must be non-empty");
    ASSERT_VALID_ARGUMENTS(shards_param.size() > 0, "Shards must be non-empty");

    ValidateShards();
  }

  /**
   * @brief Calculates the total number of elements in the tensor
   *
   * Computes the product of all dimension sizes to determine the total
   * number of elements that would be contained in a tensor of this shape.
   *
   * @return Total number of elements across all dimensions
   */
  [[nodiscard]] std::size_t GetSize() const {
    std::size_t size = 1;
    for (const auto& [dim_name, dim] : dims) {
      size *= dim.size;
    }
    return size;
  }

  /**
   * @brief Creates an ownership map for the given tensor selection
   *
   * Creates a TensorOwnershipMap that describes which shards own which portions
   * of the specified tensor selection. This is used for efficient tensor
   * operation routing and data movement planning.
   *
   * @param selection Selection of the tensor to create ownership map for
   * @return Shared pointer to TensorOwnershipMap containing the mapping
   *
   * @throws std::invalid_argument if selection is null or has mismatched tensor
   * name
   */
  [[nodiscard]] TensorOwnershipMapPtr GetOwnershipMap(
      TensorSelectionPtr selection) const {
    ASSERT_VALID_POINTER_ARGUMENT(selection);
    ASSERT_VALID_ARGUMENTS(
        selection->name == name,
        "Selection tensor name {} does not match metadata tensor name {}",
        selection->name, name);

    return std::make_shared<TensorOwnershipMap>(selection, shards);
  }

  /**
   * @brief Returns a string representation of the tensor metadata
   *
   * @return String containing tensor name, dimensions, data type, and shard
   * information
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("TensorMetadata(name={}, dims={}, dtype={}, shards={})",
                       name, dims, dtype, shards);
  }

  const TensorName name;    ///< Name of the tensor
  const TensorDimMap dims;  ///< Map of dimension names to TensorDim objects
  const DType dtype;        ///< Data type of tensor elements
  const TensorShardsMap shards;  ///< Map of node IDs to tensor shards
  const std::size_t size;        ///< Total number of elements in the tensor

 private:
  /**
   * @brief Validates that the shards completely cover the tensor dimensions
   *
   * Ensures that all tensor dimensions are fully covered by the provided shards
   * with no gaps. This validation is performed during construction to guarantee
   * data integrity.
   *
   * @throws std::invalid_argument if shards don't fully span the tensor
   */
  void ValidateShards() const {
    std::size_t total_shard_size = 0;

    for (const auto& [_, shard] : shards) {
      total_shard_size += shard->shard_size;
    }

    ASSERT_VALID_ARGUMENTS(total_shard_size == size,
                           "Total shard size {} does not match tensor size {}",
                           total_shard_size, size);

    // Ensure that no two shards overlap by using TensorSelection intersections
    for (const auto& [id1, shard1] : shards) {
      for (const auto& [id2, shard2] : shards) {
        if (id1 >= id2) continue;  // Only check each pair once, skip self

        // Create selections from each shard
        TensorSelectionPtr selection1 =
            setu::coordinator::datatypes::CreateSelectionFromShard(shard1);
        TensorSelectionPtr selection2 =
            setu::coordinator::datatypes::CreateSelectionFromShard(shard2);

        // Check if they intersect
        TensorSelectionPtr intersection =
            selection1->GetIntersection(selection2);

        ASSERT_VALID_ARGUMENTS(
            intersection->IsEmpty(),
            "Shards {} and {} overlap - their intersection is non-empty", id1,
            id2);
      }
    }
  }
};
//==============================================================================
}  // namespace setu::coordinator::datatypes
//==============================================================================

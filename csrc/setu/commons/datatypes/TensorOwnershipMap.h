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
#include "commons/datatypes/TensorSelection.h"
#include "commons/datatypes/TensorShard.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
/**
 * @brief Maps tensor selection subsets to their owning shards for distributed
 * operations
 *
 * TensorOwnershipMap provides a mapping that shows which shards own specific
 * portions of a tensor as defined by TensorSelection subsets. This is essential
 * for routing tensor operations to the correct devices and planning efficient
 * data movement in distributed tensor computations.
 */
struct TensorOwnershipMap {
  /**
   * @brief Constructs an ownership map from a tensor selection and available
   * shards
   *
   * Creates a mapping by intersecting the input selection with each shard to
   * determine which portions of the selection are owned by which shards. Only
   * non-empty intersections are included in the final mapping.
   *
   * @param selection_param Tensor selection to create ownership map for
   * @param shards_param Map of available tensor shards to check against
   *
   * @throws std::invalid_argument if selection or shards are null, or if shards
   * is empty
   */
  TensorOwnershipMap(TensorSelectionPtr selection_param,
                     TensorShardsMap shards_param)
      : shard_mapping(BuildOwnershipMapping(selection_param, shards_param)) {
    ASSERT_VALID_POINTER_ARGUMENT(selection_param);
    ASSERT_VALID_ARGUMENTS(shards_param.size() > 0, "Shards must be non-empty");
  }

  /**
   * @brief Returns the number of shard ownership mappings
   *
   * @return Number of (selection subset, shard) pairs in this ownership map
   */
  [[nodiscard]] std::size_t GetNumShards() const {
    return shard_mapping.size();
  }

  /**
   * @brief Returns a string representation of the ownership map
   *
   * @return String containing all ownership mappings
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("TensorOwnershipMap(mappings={})", shard_mapping);
  }

  /// @brief Vector of (selection subset, owning shard) pairs
  const std::vector<std::pair<TensorSelectionPtr, TensorShardPtr>>
      shard_mapping;

 private:
  /**
   * @brief Builds the ownership mapping from selection subsets to shards
   *
   * Analyzes the input TensorSelection and determines which shards own which
   * subsets of the selection by computing intersections. Only creates mappings
   * for non-empty intersections to avoid unnecessary entries.
   *
   * @param selection Tensor selection to analyze
   * @param shards Available tensor shards to check against
   * @return Vector of (selection subset, owning shard) pairs
   */
  static std::vector<std::pair<TensorSelectionPtr, TensorShardPtr>>
  BuildOwnershipMapping(TensorSelectionPtr selection, TensorShardsMap shards);
};
//==============================================================================
/// @brief Shared pointer to a TensorOwnershipMap object
using TensorOwnershipMapPtr = std::shared_ptr<TensorOwnershipMap>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

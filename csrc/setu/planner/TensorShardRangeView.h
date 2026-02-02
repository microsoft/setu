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
//==============================================================================
#include "commons/datatypes/TensorShardMetadata.h"
#include "commons/utils/ContiguousBufferIterator.h"
#include "metastore/datatypes/TensorOwnershipMap.h"
//==============================================================================
namespace setu::planner {
//==============================================================================
using setu::commons::TensorDimName;
using setu::commons::datatypes::TensorShardMetadataPtr;
using setu::commons::utils::ContiguousBufferRange;
using setu::commons::utils::ContiguousBufferRangeView;
using setu::metastore::datatypes::TensorOwnershipMapPtr;
//==============================================================================
/**
 * @brief A buffer range within a specific shard
 *
 * Combines shard metadata with a contiguous buffer range, representing
 * a portion of data within that shard's local buffer.
 */
struct ShardBufferRange {
  TensorShardMetadataPtr metadata;
  ContiguousBufferRange range;

  [[nodiscard]] std::string ToString() const {
    return std::format("ShardBufferRange(shard_id={}, range={})", metadata->id,
                       range.ToString());
  }
};
//==============================================================================
/**
 * @brief View over buffer ranges across shards in row-major order
 *
 * Given a TensorOwnershipMap (mapping selection subsets to owning shards),
 * this class produces ShardBufferRange entries in row-major tensor order.
 *
 * The view:
 * 1. Sorts shards by their row-major start position
 * 2. Localizes each selection subset to the shard's coordinate space
 * 3. Uses ContiguousBufferRangeView to enumerate contiguous ranges
 * 4. Produces ShardBufferRange entries pairing metadata with ranges
 *
 * This is used by the planner to iterate over source/destination buffer
 * regions in matching order for the two-pointer copy algorithm.
 */
class TensorShardRangeView {
 public:
  using Iterator = std::vector<ShardBufferRange>::const_iterator;

  /**
   * @brief Construct a view from an ownership map
   *
   * @param ownership_map Mapping of selection subsets to owning shards
   */
  explicit TensorShardRangeView(TensorOwnershipMapPtr ownership_map);

  [[nodiscard]] Iterator begin() const { return ranges_.begin(); }
  [[nodiscard]] Iterator end() const { return ranges_.end(); }

  [[nodiscard]] std::size_t size() const { return ranges_.size(); }
  [[nodiscard]] bool empty() const { return ranges_.empty(); }

 private:
  std::vector<ShardBufferRange> ranges_;

  void ComputeRanges(TensorOwnershipMapPtr ownership_map);
};
//==============================================================================
}  // namespace setu::planner
//==============================================================================

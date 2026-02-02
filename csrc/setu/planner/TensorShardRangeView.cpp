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
#include "planner/TensorShardRangeView.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::planner {
//==============================================================================
TensorShardRangeView::TensorShardRangeView(
    TensorOwnershipMapPtr ownership_map) {
  ASSERT_VALID_POINTER_ARGUMENT(ownership_map);
  ComputeRanges(ownership_map);
}
//==============================================================================
void TensorShardRangeView::ComputeRanges(TensorOwnershipMapPtr ownership_map) {
  for (const auto& [selection_subset, shard_metadata] :
       ownership_map->shard_mapping) {
    auto localized = selection_subset->Localize(shard_metadata);
    std::vector<TensorDimName> dim_order;
    for (const auto& dim : shard_metadata->spec.dims) {
      dim_order.push_back(dim.name);
    }

    ContiguousBufferRangeView range_view(dim_order, localized);
    for (const auto& range : range_view) {
      ranges_.push_back(
          ShardBufferRange{.metadata = shard_metadata, .range = range});
    }
  }
}
//==============================================================================
}  // namespace setu::planner
//==============================================================================

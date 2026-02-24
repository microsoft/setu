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
  auto to_us = [](auto d) {
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
  };

  long long total_localize_us = 0;
  long long total_range_view_us = 0;

  for (const auto& [selection_subset, shard_metadata] :
       ownership_map->shard_mapping) {
    auto tl0 = std::chrono::steady_clock::now();
    auto localized = selection_subset->Localize(shard_metadata);
    auto tl1 = std::chrono::steady_clock::now();

    std::vector<TensorDimName> dim_order;
    for (const auto& dim : shard_metadata->spec.dims) {
      dim_order.push_back(dim.name);
    }

    auto tr0 = std::chrono::steady_clock::now();
    ContiguousBufferRangeView range_view(dim_order, localized);
    auto tr1 = std::chrono::steady_clock::now();

    for (const auto& range : range_view) {
      ranges_.push_back(
          ShardBufferRange{.metadata = shard_metadata, .range = range});
    }

    total_localize_us += to_us(tl1 - tl0);
    total_range_view_us += to_us(tr1 - tr0);
  }

  LOG_INFO(
      "TensorShardRangeView: shards={}, Localize={}us, "
      "ContiguousBufferRangeView={}us, total_ranges={}",
      ownership_map->shard_mapping.size(), total_localize_us,
      total_range_view_us, ranges_.size());
}
//==============================================================================
}  // namespace setu::planner
//==============================================================================

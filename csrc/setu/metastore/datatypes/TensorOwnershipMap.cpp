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
#include "metastore/datatypes/TensorOwnershipMap.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/Types.h"
//==============================================================================
namespace setu::metastore::datatypes {
//==============================================================================
using setu::commons::datatypes::CreateSelectionFromShardMetadata;
using setu::commons::datatypes::TensorSelectionPtr;
using setu::commons::datatypes::TensorShardMetadataMap;
using setu::commons::datatypes::TensorShardMetadataPtr;
//==============================================================================
std::vector<std::pair<TensorSelectionPtr, TensorShardMetadataPtr>>
TensorOwnershipMap::BuildOwnershipMapping(TensorSelectionPtr selection,
                                          TensorShardMetadataMap shards) {
  ASSERT_VALID_POINTER_ARGUMENT(selection);

  auto to_us = [](auto d) {
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
  };

  std::vector<std::pair<TensorSelectionPtr, TensorShardMetadataPtr>>
      ownership_map;

  long long total_create_us = 0;
  long long total_intersect_us = 0;
  long long total_isempty_us = 0;

  // For each shard, determine which subset of the selection it owns
  for (const auto& [shard_id, shard] : shards) {
    ASSERT_VALID_POINTER_ARGUMENT(shard);

    auto tc0 = std::chrono::steady_clock::now();
    TensorSelectionPtr shard_selection =
        CreateSelectionFromShardMetadata(shard);
    auto tc1 = std::chrono::steady_clock::now();
    TensorSelectionPtr intersection =
        selection->GetIntersection(shard_selection);
    auto tc2 = std::chrono::steady_clock::now();

    if (intersection->IsEmpty()) {
      auto tc3 = std::chrono::steady_clock::now();
      total_create_us += to_us(tc1 - tc0);
      total_intersect_us += to_us(tc2 - tc1);
      total_isempty_us += to_us(tc3 - tc2);
      continue;
    }
    auto tc3 = std::chrono::steady_clock::now();
    total_create_us += to_us(tc1 - tc0);
    total_intersect_us += to_us(tc2 - tc1);
    total_isempty_us += to_us(tc3 - tc2);

    ownership_map.push_back(std::make_pair(intersection, shard));
  }

  auto ts0 = std::chrono::steady_clock::now();
  // Sort by shard's row-major start position for consistent iteration order
  std::sort(ownership_map.begin(), ownership_map.end(),
            [](const auto& a, const auto& b) {
              return a.second->spec < b.second->spec;
            });
  auto ts1 = std::chrono::steady_clock::now();

  LOG_INFO(
      "BuildOwnershipMapping: shards={}, CreateSelection={}us, "
      "Intersection={}us, IsEmpty={}us, Sort={}us",
      shards.size(), total_create_us, total_intersect_us, total_isempty_us,
      to_us(ts1 - ts0));

  return ownership_map;
}
//==============================================================================
}  // namespace setu::metastore::datatypes
//==============================================================================

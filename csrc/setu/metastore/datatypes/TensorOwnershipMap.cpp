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

  std::vector<std::pair<TensorSelectionPtr, TensorShardMetadataPtr>>
      ownership_map;

  std::int64_t total_create_us = 0;
  std::int64_t total_intersect_us = 0;
  std::int64_t total_isempty_us = 0;

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
      total_create_us +=
          std::chrono::duration_cast<std::chrono::microseconds>(tc1 - tc0)
              .count();
      total_intersect_us +=
          std::chrono::duration_cast<std::chrono::microseconds>(tc2 - tc1)
              .count();
      total_isempty_us +=
          std::chrono::duration_cast<std::chrono::microseconds>(tc3 - tc2)
              .count();
      continue;
    }

    auto tc3 = std::chrono::steady_clock::now();
    total_create_us +=
        std::chrono::duration_cast<std::chrono::microseconds>(tc1 - tc0)
            .count();
    total_intersect_us +=
        std::chrono::duration_cast<std::chrono::microseconds>(tc2 - tc1)
            .count();
    total_isempty_us +=
        std::chrono::duration_cast<std::chrono::microseconds>(tc3 - tc2)
            .count();

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
      "BuildOwnershipMapping: create_selection={}us, intersect={}us, "
      "is_empty={}us, sort={}us, shards={}, owned={}",
      total_create_us, total_intersect_us, total_isempty_us,
      std::chrono::duration_cast<std::chrono::microseconds>(ts1 - ts0).count(),
      shards.size(), ownership_map.size());

  return ownership_map;
}
//==============================================================================
}  // namespace setu::metastore::datatypes
//==============================================================================

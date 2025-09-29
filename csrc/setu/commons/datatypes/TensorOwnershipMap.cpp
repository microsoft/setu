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
#include "commons/datatypes/TensorOwnershipMap.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/Types.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
std::vector<std::pair<TensorSelectionPtr, TensorShardPtr>>
TensorOwnershipMap::BuildOwnershipMapping(TensorSelectionPtr selection,
                                          TensorShardsMap shards) {
  ASSERT_VALID_POINTER_ARGUMENT(selection);

  std::vector<std::pair<TensorSelectionPtr, TensorShardPtr>> ownership_map;

  // For each shard, determine which subset of the selection it owns
  for (const auto& [shard_id, shard] : shards) {
    ASSERT_VALID_POINTER_ARGUMENT(shard);

    TensorSelectionPtr intersection = selection->GetIntersection(shard);

    if (intersection->IsEmpty()) {
      continue;
    }

    ownership_map.push_back(std::make_pair(intersection, shard));
  }

  return ownership_map;
}
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================

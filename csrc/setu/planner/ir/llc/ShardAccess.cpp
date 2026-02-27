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
#include "planner/ir/llc/ShardAccess.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================

ShardAccessMap GetShardAccess(const Program& program) {
  ShardAccessMap access_map;

  for (const auto& instruction : program) {
    for (const auto& [shard_id, mode] : instruction.GetShardAccess()) {
      if (mode == ShardAccessMode::kWrite) {
        // Write always wins (override any existing read)
        access_map[shard_id] = ShardAccessMode::kWrite;
      } else {
        // Read: only insert if not already present (don't downgrade write)
        access_map.try_emplace(shard_id, ShardAccessMode::kRead);
      }
    }
  }

  return access_map;
}

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

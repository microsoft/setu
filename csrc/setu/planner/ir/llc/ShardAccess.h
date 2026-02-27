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
#include "planner/ir/llc/Instruction.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================

/**
 * @brief Extract merged, sorted shard access requirements for an entire
 * program.
 *
 * Scans all instructions via Instruction::GetShardAccess(), merges access
 * modes per shard (write takes precedence over read for the same shard),
 * and returns results sorted by ShardId for consistent lock ordering.
 *
 * @param program The IR program to analyze
 * @return Sorted map of shard IDs to access modes, deduplicated
 */
[[nodiscard]] ShardAccessMap GetShardAccess(const Program& program);

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

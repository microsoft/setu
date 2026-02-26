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
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::ShardId;
//==============================================================================

/// @brief Access mode for a tensor shard within a program
enum class ShardAccessMode : std::uint8_t {
  kRead,   ///< Shared read access (source of Copy/Send)
  kWrite,  ///< Exclusive write access (destination of Copy/Receive)
};

/// @brief Sorted map from shard ID to its access mode.
/// Sorted by ShardId (UUID) for consistent lock ordering.
/// Each shard appears at most once; write takes precedence over read.
using ShardAccessMap = std::map<ShardId, ShardAccessMode>;

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

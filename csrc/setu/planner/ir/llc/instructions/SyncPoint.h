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
#include "commons/utils/Serialization.h"
//==============================================================================
#include "planner/ir/llc/ShardAccessTypes.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

/// Marks the completion of a write operation for dependency tracking.
///
/// The executor records a CUDA event (identified by `id`) on whichever
/// stream just ran the preceding write operation.  One or more Wait
/// instructions referencing the same `id` can then declare that a later
/// read must not start until this event fires.
///
/// Stream assignment is entirely the executor's responsibility; SyncPoint
/// carries no stream information.  The invariant is that SyncPoint is always
/// placed immediately after the write op it tracks.
struct SyncPoint {
  explicit SyncPoint(std::uint32_t id_param) : id(id_param) {}

  ~SyncPoint() = default;
  SyncPoint(const SyncPoint&) = default;
  SyncPoint& operator=(const SyncPoint&) = default;
  SyncPoint(SyncPoint&&) = default;
  SyncPoint& operator=(SyncPoint&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static SyncPoint Deserialize(const BinaryRange& range);

  /// @brief Extract shard access requirements for this instruction.
  [[nodiscard]] ShardAccessMap GetShardAccess() const;

  std::uint32_t id;  ///< Unique identifier within the LLC program
};

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

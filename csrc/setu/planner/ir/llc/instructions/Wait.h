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

/// Declares a data dependency on a prior write.
///
/// The executor buffers this id and, when the next data operation starts,
/// issues cudaStreamWaitEvent on that op's stream for the CUDA event
/// recorded by SyncPoint(id).  This ensures the next op cannot begin until
/// the write tracked by SyncPoint(id) is visible.
///
/// Stream assignment is entirely the executor's responsibility; Wait carries
/// no stream information.  The invariant is that Wait is always placed
/// immediately before the read op that depends on SyncPoint(id).
struct Wait {
  explicit Wait(std::uint32_t id_param) : id(id_param) {}

  ~Wait() = default;
  Wait(const Wait&) = default;
  Wait& operator=(const Wait&) = default;
  Wait(Wait&&) = default;
  Wait& operator=(Wait&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static Wait Deserialize(const BinaryRange& range);

  /// @brief Extract shard access requirements for this instruction.
  [[nodiscard]] ShardAccessMap GetShardAccess() const;

  std::uint32_t id;  ///< Matches the SyncPoint this depends on
};

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

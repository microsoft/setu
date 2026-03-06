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
#include "commons/datatypes/Device.h"
#include "commons/utils/Serialization.h"
//==============================================================================
#include "planner/Participant.h"
#include "planner/ir/llc/CommId.h"
#include "planner/ir/llc/ShardAccessTypes.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::DeviceRank;
using setu::commons::NodeId;
using setu::commons::datatypes::Device;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::planner::Participant;
//==============================================================================

/// Initialize a new communicator for a group of participant devices.
///
/// Maps each Participant to a DeviceRank within the communicator.  The
/// `comm_id` is an opaque identifier shared by all participants so they
/// can collectively initialize the communicator.
struct InitComm {
  InitComm(CommId comm_id,
           std::unordered_map<Participant, DeviceRank> participant_to_rank)
      : comm_id(comm_id), participant_to_rank(std::move(participant_to_rank)) {}

  ~InitComm() = default;
  InitComm(const InitComm&) = default;
  InitComm& operator=(const InitComm&) = default;
  InitComm(InitComm&&) = default;
  InitComm& operator=(InitComm&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static InitComm Deserialize(const BinaryRange& range);

  /// @brief Extract shard access requirements for this instruction.
  [[nodiscard]] ShardAccessMap GetShardAccess() const;

  CommId comm_id;
  std::unordered_map<Participant, DeviceRank> participant_to_rank;
};

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

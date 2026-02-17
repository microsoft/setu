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
#include <nccl.h>
//==============================================================================
#include "setu/commons/StdCommon.h"
#include "setu/commons/Types.h"
#include "setu/commons/datatypes/Device.h"
#include "setu/commons/utils/Serialization.h"
#include "setu/planner/Participant.h"
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

/// Initialize a new NCCL communicator for a group of participant devices.
///
/// Maps each Participant to a DeviceRank within the communicator.  The
/// `comm_id` is a globally unique NCCL identifier shared by all participants
/// so they can collectively call ncclCommInitRank.
struct InitComm {
  InitComm(ncclUniqueId comm_id,
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

  ncclUniqueId comm_id;
  std::unordered_map<Participant, DeviceRank> participant_to_rank;
};

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

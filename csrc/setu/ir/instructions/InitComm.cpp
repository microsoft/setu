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
#include "setu/ir/instructions/InitComm.h"
//==============================================================================
#include "setu/planner/Planner.h"
//==============================================================================
namespace setu::ir {
//==============================================================================

std::string InitComm::ToString() const {
  std::string hex;
  for (std::size_t i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    hex += std::format("{:02x}", static_cast<std::uint8_t>(comm_id.internal[i]));
  }
  return std::format("InitComm(comm_id={}, participant_to_rank={})", hex,
                     participant_to_rank);
}

void InitComm::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(comm_id, participant_to_rank);
}

InitComm InitComm::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [comm_id, participant_to_rank] =
      reader.ReadFields<ncclUniqueId,
                        std::unordered_map<Participant, DeviceRank>>();
  return InitComm(comm_id, participant_to_rank);
}

//==============================================================================
}  // namespace setu::ir
//==============================================================================

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
#include "planner/ir/llc/instructions/InitComm.h"
//==============================================================================
#include "setu/planner/Planner.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================

std::string InitComm::ToString() const {
  // Show only the first 8 bytes of the comm_id as a short identifier
  std::string short_hex;
  for (std::size_t i = 0; i < 8; ++i) {
    short_hex +=
        std::format("{:02x}", static_cast<std::uint8_t>(comm_id.internal[i]));
  }

  std::string ranks_str;
  for (const auto& [participant, rank] : participant_to_rank) {
    if (!ranks_str.empty()) ranks_str += ", ";
    ranks_str += std::format("{}={}", participant.ToString(), rank);
  }

  return std::format("InitComm(comm_id={}..., ranks={{{}}})", short_hex,
                     ranks_str);
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
}  // namespace setu::planner::ir::llc
//==============================================================================

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
namespace setu::ir {
//==============================================================================

std::string InitCommInstruction::ToString() const {
  return std::format("InitCommInstruction(device_to_rank_size={})",
                     device_to_rank.size());
}

void InitCommInstruction::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(comm_id, device_to_rank);
}

InitCommInstruction InitCommInstruction::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [comm_id, device_to_rank] =
      reader
          .ReadFields<ncclUniqueId, std::unordered_map<DeviceRank, std::int32_t>>();
  return InitCommInstruction(comm_id, std::move(device_to_rank));
}

//==============================================================================
}  // namespace setu::ir
//==============================================================================

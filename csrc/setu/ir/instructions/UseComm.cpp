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
#include "setu/ir/instructions/UseComm.h"
//==============================================================================
namespace setu::ir {
//==============================================================================

std::string UseCommInstruction::ToString() const {
  return std::format("UseCommInstruction(comm_id_present={})", true);
}

void UseCommInstruction::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(comm_id);
}

UseCommInstruction UseCommInstruction::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [comm_id] = reader.ReadFields<ncclUniqueId>();
  return UseCommInstruction(comm_id);
}

//==============================================================================
}  // namespace setu::ir
//==============================================================================

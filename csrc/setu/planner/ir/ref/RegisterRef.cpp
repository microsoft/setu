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
#include "planner/ir/ref/RegisterRef.h"
//==============================================================================
namespace setu::planner::ir::ref {
//==============================================================================
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

std::string RegisterRef::ToString() const {
  std::string part_str =
      participant.has_value() ? participant->ToString() : "<none>";
  return std::format("RegisterRef(#{}, participant={})", register_index,
                     part_str);
}

void RegisterRef::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(register_index, participant);
}

RegisterRef RegisterRef::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [register_index, participant] =
      reader.ReadFields<std::uint32_t, std::optional<Participant>>();
  return RegisterRef(register_index, std::move(participant));
}

//==============================================================================
}  // namespace setu::planner::ir::ref
//==============================================================================

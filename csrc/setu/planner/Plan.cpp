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
#include "planner/Plan.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::planner {
//==============================================================================

std::unordered_map<NodeId, Plan> Plan::Fragments() {
  std::unordered_map<NodeId, Plan> fragments;

  // Group programs by NodeId, keeping participants the same across all
  // fragments
  for (const auto& [participant, prog] : program) {
    auto& fragment = fragments[participant.node_id];
    fragment.participants = participants;
    fragment.program[participant] = prog;
  }

  return fragments;
}

void Plan::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(participants, program);
}

Plan Plan::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [participants_val, program_val] =
      reader
          .ReadFields<Participants, std::unordered_map<Participant, Program>>();
  Plan plan;
  plan.participants = std::move(participants_val);
  plan.program = std::move(program_val);
  return plan;
}

//==============================================================================
}  // namespace setu::planner
//==============================================================================

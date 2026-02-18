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
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
#include "commons/Types.h"
//==============================================================================
#include "commons/utils/Serialization.h"
#include "planner/Participant.h"
#include "planner/ir/llc/Instruction.h"
//==============================================================================
namespace setu::planner {
//==============================================================================

using setu::commons::BinaryBuffer;
using setu::commons::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::planner::ir::llc::Program;

//==============================================================================

struct Plan {
  std::unordered_map<NodeId, Plan> Fragments();

  [[nodiscard]] std::string ToString() const {
    return std::format("Plan(participants={}, programs={})", participants,
                       program);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static Plan Deserialize(const BinaryRange& range);

  Participants participants;
  std::unordered_map<Participant, Program> program;
};

//==============================================================================
}  // namespace setu::planner
//==============================================================================

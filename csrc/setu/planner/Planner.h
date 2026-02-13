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
#include "commons/datatypes/CopySpec.h"
#include "commons/utils/Serialization.h"
#include "ir/Instruction.h"
#include "metastore/MetaStore.h"
#include "planner/Participant.h"
//==============================================================================
namespace setu::planner {
//==============================================================================

using setu::commons::BinaryBuffer;
using setu::commons::BinaryRange;
using setu::commons::datatypes::CopySpec;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::ir::Program;
using setu::metastore::MetaStore;

//==============================================================================

struct Plan {
  std::unordered_map<NodeId, Plan> Fragments();

  [[nodiscard]] std::string ToString() const {
    return std::format("Plan(participants={}, programs={})", participants,
                       program);
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.WriteFields(participants, program);
  }

  static Plan Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    auto [participants_val, program_val] =
        reader.ReadFields<Participants,
                          std::unordered_map<Participant, Program>>();
    Plan plan;
    plan.participants = std::move(participants_val);
    plan.program = std::move(program_val);
    return plan;
  }

  Participants participants;
  std::unordered_map<Participant, Program> program;
};

class Planner {
 public:
  virtual ~Planner() = default;
  virtual Plan Compile(CopySpec& spec, MetaStore& metastore) = 0;
};
//==============================================================================
}  // namespace setu::planner
//==============================================================================

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
//==============================================================================
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/Device.h"
#include "ir/Instruction.h"
#include "metastore/MetaStore.h"
//==============================================================================
namespace setu::planner {
//==============================================================================

using setu::commons::NodeId;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::Device;
using setu::ir::Program;
using setu::metastore::MetaStore;

using NodeAgentId = std::size_t;
using DeviceId = std::size_t;

using Participant = Device;
using Participants = std::vector<Device>;

struct Plan {
  std::unordered_map<NodeId, Plan> Fragments();

  [[nodiscard]] std::string ToString() const {
    return std::format("Plan(participants={}, programs={})",
                       participants.size(), program.size());
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

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
#include "setu/commons/StdCommon.h"
#include "setu/commons/Types.h"
#include "setu/commons/utils/Serialization.h"
#include "setu/ir/Instruction.h"
//==============================================================================
namespace setu::coordinator::datatypes {
//==============================================================================
using setu::commons::DeviceRank;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::ir::Instruction;
//==============================================================================

struct Program {
  Program() = default;
  ~Program() = default;

  [[nodiscard]] std::string ToString() const {
    return std::format("Program(participating_workers={}, instrs={})",
                       participating_workers.size(), instrs.size());
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.WriteFields(participating_workers, instrs);
  }

  static Program Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    Program program;
    std::tie(program.participating_workers, program.instrs) =
        reader.ReadFields<std::vector<DeviceRank>, std::vector<Instruction>>();
    return program;
  }

  std::vector<DeviceRank> participating_workers;
  std::vector<Instruction> instrs;
};

//==============================================================================
}  // namespace setu::coordinator::datatypes
//==============================================================================

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
#include "planner/ir/llc/Instruction.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================

std::string Instruction::ToString() const {
  return std::visit([](const auto& inst) { return inst.ToString(); }, instr);
}

void Instruction::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);

  std::visit(
      [&writer](const auto& inst) {
        using T = std::decay_t<decltype(inst)>;
        InstructionType type = InstructionType::kInitComm;
        if constexpr (std::is_same_v<T, InitComm>) {
          type = InstructionType::kInitComm;
        } else if constexpr (std::is_same_v<T, UseComm>) {
          type = InstructionType::kUseComm;
        } else if constexpr (std::is_same_v<T, Copy>) {
          type = InstructionType::kCopy;
        } else if constexpr (std::is_same_v<T, Send>) {
          type = InstructionType::kSend;
        } else if constexpr (std::is_same_v<T, Receive>) {
          type = InstructionType::kReceive;
        } else if constexpr (std::is_same_v<T, Barrier>) {
          type = InstructionType::kBarrier;
        }

        writer.Write<std::uint8_t>(static_cast<std::uint8_t>(type));
        writer.Write(inst);
      },
      instr);
}

Instruction Instruction::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);

  const auto type_id = reader.Read<std::uint8_t>();
  switch (static_cast<InstructionType>(type_id)) {
    case InstructionType::kInitComm:
      return Instruction(reader.Read<InitComm>());
    case InstructionType::kUseComm:
      return Instruction(reader.Read<UseComm>());
    case InstructionType::kCopy:
      return Instruction(reader.Read<Copy>());
    case InstructionType::kSend:
      return Instruction(reader.Read<Send>());
    case InstructionType::kReceive:
      return Instruction(reader.Read<Receive>());
    case InstructionType::kBarrier:
      return Instruction(Barrier::Deserialize(range));
    default:
      RAISE_RUNTIME_ERROR("Unknown instruction type id {}", type_id);
  }
}

void Instruction::Embellish(
    const std::function<DevicePtr(const BufferRef&)>& resolver) {
  std::visit(
      [&resolver](auto& inst) {
        // Use a compile-time check to see if the instruction has an Embellish
        // method
        if constexpr (requires { inst.Embellish(resolver); }) {
          inst.Embellish(resolver);
        }
      },
      instr);
}

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

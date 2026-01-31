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
#include "setu/commons/utils/Serialization.h"
//==============================================================================
#include "setu/ir/instructions/Copy.h"
#include "setu/ir/instructions/InitComm.h"
#include "setu/ir/instructions/Receive.h"
#include "setu/ir/instructions/Send.h"
#include "setu/ir/instructions/UseComm.h"
//==============================================================================
namespace setu::ir {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorShardIdentifier;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

enum class InstructionType : std::uint8_t {
  kInitComm = 1,
  kUseComm = 2,
  kCopy = 3,
  kSend = 4,
  kReceive = 5,
};

using InstructionVariant =
    std::variant<InitCommInstruction, UseCommInstruction, CopyInstruction,
                 SendInstruction, ReceiveInstruction>;

struct Instruction {
  Instruction() = delete;

  template <typename T>
  explicit Instruction(T inst) : instr(std::move(inst)) {}

  ~Instruction() = default;
  Instruction(const Instruction&) = default;
  Instruction& operator=(const Instruction&) = default;
  Instruction(Instruction&&) = default;
  Instruction& operator=(Instruction&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static Instruction Deserialize(const BinaryRange& range);

  void Embellish(
      const std::function<DevicePtr(const TensorShardIdentifier&)>& resolver);

  InstructionVariant instr;
};

//==============================================================================
}  // namespace setu::ir
//==============================================================================

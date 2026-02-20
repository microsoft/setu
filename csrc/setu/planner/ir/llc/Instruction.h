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
#include "commons/utils/Serialization.h"
//==============================================================================
#include "planner/ir/llc/instructions/Barrier.h"
#include "planner/ir/llc/instructions/Copy.h"
#include "planner/ir/llc/instructions/InitComm.h"
#include "planner/ir/llc/instructions/Receive.h"
#include "planner/ir/llc/instructions/Send.h"
#include "planner/ir/llc/instructions/UseComm.h"
#include "planner/ir/ref/ShardRef.h"
//==============================================================================
/// Low-Level Copy (LLC) IR — the target IR for backend code generation.
///
/// LLC instructions are concrete, per-device operations that a backend (e.g.
/// NCCL) can execute directly.  A CIR program is lowered into one LLC Program
/// per participating device.  Each instruction operates on physical shard
/// byte offsets and carries enough information for the worker to resolve
/// device pointers and issue the corresponding runtime calls.
///
/// Instruction set:
///   InitComm  — initialize a new NCCL communicator among a set of devices
///   UseComm   — switch to an already-initialised communicator
///   Copy      — local (same-device) memcpy between shard regions
///   Send      — NCCL point-to-point send to a peer rank
///   Receive   — NCCL point-to-point receive from a peer rank
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::planner::ir::ref::ShardRef;
//==============================================================================

enum class InstructionType : std::uint8_t {
  kInitComm = 1,
  kUseComm = 2,
  kCopy = 3,
  kSend = 4,
  kReceive = 5,
  kBarrier = 6,
};

using InstructionVariant =
    std::variant<InitComm, UseComm, Copy, Send, Receive, Barrier>;

/// A single LLC instruction.  Wraps one of the five concrete instruction
/// types in a variant.  Supports serialization for wire transfer and
/// "embellishment" (late-binding of device pointers before execution).
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

  void Embellish(const std::function<DevicePtr(const BufferRef&)>& resolver);

  InstructionVariant instr;
};

/// An LLC Program is a linear sequence of instructions for a single device.
using Program = std::vector<Instruction>;
//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

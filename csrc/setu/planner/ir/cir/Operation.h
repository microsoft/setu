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
//==============================================================================
#include "planner/ir/cir/Value.h"
#include "planner/ir/cir/operations/AllocTmp.h"
#include "planner/ir/cir/operations/CopyOp.h"
#include "planner/ir/cir/operations/Pack.h"
#include "planner/ir/cir/operations/SliceOp.h"
#include "planner/ir/cir/operations/Unpack.h"
#include "planner/ir/cir/operations/View.h"
//==============================================================================
namespace setu::planner::ir::cir {
//==============================================================================

enum class OpType : std::uint8_t {
  kView = 1,
  kAllocTmp = 2,
  kCopy = 3,
  kPack = 4,
  kUnpack = 5,
  kSlice = 6,
};

using OperationVariant =
    std::variant<ViewOp, AllocTmpOp, CopyOp, PackOp, UnpackOp, SliceOp>;

/// Wrapper around OperationVariant, providing uniform access to defs/uses
/// and string representation. Follows the same variant+wrapper pattern
/// as setu::planner::ir::llc::Instruction.
struct Operation {
  Operation() = delete;

  template <typename T>
  explicit Operation(T op_param) : op(std::move(op_param)) {}

  ~Operation() = default;
  Operation(const Operation&) = default;
  Operation& operator=(const Operation&) = default;
  Operation(Operation&&) = default;
  Operation& operator=(Operation&&) = default;

  /// Returns the type tag for this operation
  [[nodiscard]] OpType Type() const;

  /// Returns all Values defined (produced) by this operation
  [[nodiscard]] std::vector<Value> Defs() const;

  /// Returns all Values used (consumed or read) by this operation
  [[nodiscard]] std::vector<Value> Uses() const;

  /// Returns the subset of Uses() whose ownership is transferred
  /// (the value is destroyed and replaced by a new SSA version).
  /// CopyOp: {dst_in}, PackOp: {dst_in}, UnpackOp: {dst_ins...}
  [[nodiscard]] std::vector<Value> ConsumedOperands() const;

  [[nodiscard]] std::string ToString() const;

  OperationVariant op;
};

//==============================================================================
}  // namespace setu::planner::ir::cir
//==============================================================================

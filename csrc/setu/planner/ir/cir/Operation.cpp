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
#include "planner/ir/cir/Operation.h"
//==============================================================================
namespace setu::planner::ir::cir {
//==============================================================================

OpType Operation::Type() const {
  return std::visit(
      [](const auto& op) -> OpType {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, ViewOp>) {
          return OpType::kView;
        } else if constexpr (std::is_same_v<T, AllocTmpOp>) {
          return OpType::kAllocTmp;
        } else if constexpr (std::is_same_v<T, CopyOp>) {
          return OpType::kCopy;
        } else if constexpr (std::is_same_v<T, PackOp>) {
          return OpType::kPack;
        } else if constexpr (std::is_same_v<T, UnpackOp>) {
          return OpType::kUnpack;
        }
      },
      op);
}

std::vector<Value> Operation::Defs() const {
  return std::visit(
      [](const auto& op) -> std::vector<Value> {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, ViewOp>) {
          return {op.out};
        } else if constexpr (std::is_same_v<T, AllocTmpOp>) {
          return {op.out};
        } else if constexpr (std::is_same_v<T, CopyOp>) {
          return {op.dst_out};
        } else if constexpr (std::is_same_v<T, PackOp>) {
          return {op.dst_out};
        } else if constexpr (std::is_same_v<T, UnpackOp>) {
          return op.dst_outs;
        }
      },
      op);
}

std::vector<Value> Operation::Uses() const {
  return std::visit(
      [](const auto& op) -> std::vector<Value> {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, ViewOp>) {
          return {};
        } else if constexpr (std::is_same_v<T, AllocTmpOp>) {
          return {};
        } else if constexpr (std::is_same_v<T, CopyOp>) {
          return {op.src, op.dst_in};
        } else if constexpr (std::is_same_v<T, PackOp>) {
          std::vector<Value> uses = op.srcs;
          uses.push_back(op.dst_in);
          return uses;
        } else if constexpr (std::is_same_v<T, UnpackOp>) {
          std::vector<Value> uses = {op.src};
          uses.insert(uses.end(), op.dst_ins.begin(), op.dst_ins.end());
          return uses;
        }
      },
      op);
}

std::vector<Value> Operation::ConsumedOperands() const {
  return std::visit(
      [](const auto& op) -> std::vector<Value> {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, ViewOp>) {
          return {};
        } else if constexpr (std::is_same_v<T, AllocTmpOp>) {
          return {};
        } else if constexpr (std::is_same_v<T, CopyOp>) {
          return {op.dst_in};
        } else if constexpr (std::is_same_v<T, PackOp>) {
          return {op.dst_in};
        } else if constexpr (std::is_same_v<T, UnpackOp>) {
          return op.dst_ins;
        }
      },
      op);
}

std::string Operation::ToString() const {
  return std::visit([](const auto& op) { return op.ToString(); }, op);
}

//==============================================================================
}  // namespace setu::planner::ir::cir
//==============================================================================

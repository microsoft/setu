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
#include "commons/TorchCommon.h"
//==============================================================================
#include "planner/Participant.h"
//==============================================================================
namespace setu::planner::ir::cir {
//==============================================================================

/// CIR Device is a (NodeId, Device) pair identifying a physical device in the
/// distributed system. We reuse the existing Participant type directly.
using Device = setu::planner::Participant;

//==============================================================================

/// SSA value identifier -- a lightweight handle into Program's value table.
/// Each Value is assigned exactly once (by the operation that defines it)
/// and can be used as an operand by subsequent operations.
struct Value {
  std::uint32_t id;

  [[nodiscard]] bool operator==(const Value& other) const {
    return id == other.id;
  }

  [[nodiscard]] bool operator!=(const Value& other) const {
    return id != other.id;
  }

  [[nodiscard]] bool operator<(const Value& other) const {
    return id < other.id;
  }

  [[nodiscard]] std::string ToString() const { return std::format("%{}", id); }
};

//==============================================================================

/// Metadata associated with a Value, stored in Program's side table.
struct ValueInfo {
  Device device;               ///< Physical device where this value lives
  std::size_t size_elements;   ///< Number of elements in the buffer region
  torch::Dtype dtype;          ///< Element data type
  std::uint32_t def_op_index;  ///< Index of the defining operation in Program

  [[nodiscard]] std::string ToString() const {
    return std::format("ValueInfo(device={}, size={}, dtype={}, def_op={})",
                       device.ToString(), size_elements, torch::toString(dtype),
                       def_op_index);
  }
};

//==============================================================================
}  // namespace setu::planner::ir::cir
//==============================================================================
// Hash specialization for Value
//==============================================================================
namespace std {
template <>
struct hash<setu::planner::ir::cir::Value> {
  std::size_t operator()(
      const setu::planner::ir::cir::Value& v) const noexcept {
    return std::hash<std::uint32_t>{}(v.id);
  }
};
}  // namespace std
//==============================================================================

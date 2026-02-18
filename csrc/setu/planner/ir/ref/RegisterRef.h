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
#include "planner/Participant.h"
//==============================================================================
namespace setu::planner::ir::ref {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

/// Lightweight reference to a physical register (temporary buffer slot) in a
/// device's pre-allocated pool.
///
/// The device is implicit — each LLC Program is per-participant, so the worker
/// executing the program knows which device's pool to index into.  The
/// register_index identifies the specific slot assigned by register allocation.
struct RegisterRef {
  RegisterRef() = default;

  explicit RegisterRef(
      std::uint32_t register_index_param,
      std::optional<Participant> participant_param = std::nullopt)
      : register_index(register_index_param),
        participant(std::move(participant_param)) {}

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static RegisterRef Deserialize(const BinaryRange& range);

  [[nodiscard]] bool operator==(const RegisterRef& other) const {
    return register_index == other.register_index;
  }

  [[nodiscard]] bool operator!=(const RegisterRef& other) const {
    return !(*this == other);
  }

  std::uint32_t register_index{0};  ///< Slot index in the device's pool

  // Debug information
  std::optional<Participant> participant;  ///< Owning device (debug)
};

//==============================================================================
inline std::size_t hash_value(const RegisterRef& ref) {
  return std::hash<std::uint32_t>{}(ref.register_index);
}
//==============================================================================
}  // namespace setu::planner::ir::ref
//==============================================================================

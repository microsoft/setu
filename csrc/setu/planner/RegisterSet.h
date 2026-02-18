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
namespace setu::planner {
//==============================================================================

/// Specification of a set of named registers (temporary buffers).
///
/// Each register is identified by its index (0-based) and has an associated
/// size in bytes.  The RegisterSet is passed to workers at construction time;
/// they allocate device memory for each register during Setup().
///
/// Although all registers currently use a uniform size (kRegisterSize), the
/// abstraction supports per-register sizing for future flexibility.
struct RegisterSet {
  RegisterSet() = default;

  /// Construct a uniform RegisterSet where all registers share the same size.
  static RegisterSet Uniform(std::uint32_t num_registers,
                             std::size_t size_bytes) {
    RegisterSet set;
    set.sizes_.assign(num_registers, size_bytes);
    return set;
  }

  /// Add a register with the given size.  Returns the assigned index.
  std::uint32_t AddRegister(std::size_t size_bytes) {
    auto index = static_cast<std::uint32_t>(sizes_.size());
    sizes_.push_back(size_bytes);
    return index;
  }

  /// Number of registers in the set.
  [[nodiscard]] std::uint32_t NumRegisters() const {
    return static_cast<std::uint32_t>(sizes_.size());
  }

  /// Size in bytes of register at the given index.
  [[nodiscard]] std::size_t SizeBytes(std::uint32_t index) const {
    return sizes_.at(index);
  }

  /// Whether this register set is empty (no registers).
  [[nodiscard]] bool Empty() const { return sizes_.empty(); }

  std::vector<std::size_t> sizes_;  ///< Size in bytes per register index
};

//==============================================================================
}  // namespace setu::planner
//==============================================================================

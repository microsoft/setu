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
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

/// Synchronization barrier between communication stages.
///
/// Instructs the backend to complete all in-flight communication operations
/// before proceeding.  This ensures all receives from a prior stage are
/// visible before subsequent sends read the received data (multi-hop relay
/// correctness).  The concrete synchronization mechanism is backend-defined.
struct Barrier {
  Barrier() = default;

  ~Barrier() = default;
  Barrier(const Barrier&) = default;
  Barrier& operator=(const Barrier&) = default;
  Barrier(Barrier&&) = default;
  Barrier& operator=(Barrier&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static Barrier Deserialize(const BinaryRange& range);
};

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

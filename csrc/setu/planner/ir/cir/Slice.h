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
namespace setu::planner::ir::cir {
//==============================================================================

/// Contiguous region within a buffer, specified in element counts (not bytes).
/// Byte conversion happens only during backend lowering, where we use dtype to
/// do the conversion.
struct Slice {
  std::size_t offset;  ///< Start offset in elements
  std::size_t size;    ///< Number of elements

  [[nodiscard]] std::size_t End() const { return offset + size; }

  [[nodiscard]] bool operator==(const Slice& other) const {
    return offset == other.offset && size == other.size;
  }

  [[nodiscard]] bool operator!=(const Slice& other) const {
    return !(*this == other);
  }

  [[nodiscard]] std::string ToString() const {
    return std::format("[{}, {}]", offset, size);
  }
};

//==============================================================================
}  // namespace setu::planner::ir::cir
//==============================================================================

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
#include "planner/ir/cir/Slice.h"
#include "planner/ir/cir/Value.h"
//==============================================================================
namespace setu::planner::ir::cir {
//==============================================================================

/// %out = slice(%src, [offset, size])
///
/// Creates a new Value representing a contiguous sub-region of src.
/// Does not consume src -- src remains live for other uses.
/// The result inherits the device and dtype from src.
struct SliceOp {
  Value out;    ///< Result value (the sub-region)
  Value src;    ///< Source value to slice (read, not consumed)
  Slice slice;  ///< Sub-region within src (elements)

  [[nodiscard]] std::string ToString() const {
    return std::format("{} = slice({}, {})", out.ToString(), src.ToString(),
                       slice.ToString());
  }
};

//==============================================================================
}  // namespace setu::planner::ir::cir
//==============================================================================

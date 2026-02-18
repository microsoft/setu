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
#include "planner/ir/cir/Value.h"
//==============================================================================
namespace setu::planner::ir::cir {
//==============================================================================

/// %out = alloc_tmp(@node:device, size)
///
/// Allocates a virtual register (temporary buffer) on the specified device.
/// Virtual registers are mapped to physical buffers via register allocation
/// before backend lowering. Sizes are restricted to a pool of pre-defined
/// register sizes.
struct AllocTmpOp {
  Value out;                  ///< Result value (the allocated virtual register)
  Device device;              ///< Physical device where the temp buffer lives
  std::size_t size_elements;  ///< Number of elements to allocate
  torch::Dtype dtype;         ///< Element data type

  [[nodiscard]] std::string ToString() const {
    return std::format("{} = alloc_tmp({}, {}, {})", out.ToString(),
                       device.ToString(), size_elements,
                       torch::toString(dtype));
  }
};

//==============================================================================
}  // namespace setu::planner::ir::cir
//==============================================================================

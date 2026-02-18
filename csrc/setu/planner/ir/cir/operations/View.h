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
#include "planner/ir/cir/Slice.h"
#include "planner/ir/cir/Value.h"
#include "planner/ir/ref/ShardRef.h"
//==============================================================================
namespace setu::planner::ir::cir {
//==============================================================================

/// %out = view(@node:device, &shard_ref, [offset, size])
///
/// Constructs a CIR Value representing a contiguous region of a physical
/// shard buffer. The slice is in element counts; byte conversion happens
/// during backend lowering.
struct ViewOp {
  Value out;      ///< Result value
  Device device;  ///< Physical device where the shard resides
  setu::planner::ir::ref::ShardRef handle;  ///< Reference to the physical shard
  Slice slice;         ///< Region within the shard (elements)
  torch::Dtype dtype;  ///< Element data type

  [[nodiscard]] std::string ToString() const {
    return std::format("{} = view({}, &{}, {}, {})", out.ToString(),
                       device.ToString(), handle.ToString(), slice.ToString(),
                       torch::toString(dtype));
  }
};

//==============================================================================
}  // namespace setu::planner::ir::cir
//==============================================================================

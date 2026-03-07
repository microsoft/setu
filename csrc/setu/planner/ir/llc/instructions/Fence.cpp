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
#include "planner/ir/llc/instructions/Fence.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================

std::string Fence::ToString() const { return "Fence()"; }

void Fence::Serialize(BinaryBuffer& /*buffer*/) const {
  // No fields to serialize — the type tag is written by Instruction::Serialize
}

Fence Fence::Deserialize(const BinaryRange& /*range*/) { return Fence(); }

ShardAccessMap Fence::GetShardAccess() const { return {}; }

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

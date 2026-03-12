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
#include "planner/ir/llc/instructions/SyncPoint.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================

std::string SyncPoint::ToString() const {
  return std::format("SyncPoint(id={})", id);
}

void SyncPoint::Serialize(BinaryBuffer& buffer) const {
  setu::commons::utils::BinaryWriter writer(buffer);
  writer.Write(id);
}

SyncPoint SyncPoint::Deserialize(const BinaryRange& range) {
  setu::commons::utils::BinaryReader reader(range);
  return SyncPoint(reader.Read<std::uint32_t>());
}

ShardAccessMap SyncPoint::GetShardAccess() const { return {}; }

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

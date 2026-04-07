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
  return std::format("SyncPoint(id={}, wait_count={})", id, wait_count);
}

void SyncPoint::Serialize(BinaryBuffer& buffer) const {
  setu::commons::utils::BinaryWriter writer(buffer);
  writer.WriteFields(id, wait_count);
}

SyncPoint SyncPoint::Deserialize(const BinaryRange& range) {
  setu::commons::utils::BinaryReader reader(range);
  auto [id_val, wait_count_val] =
      reader.ReadFields<std::uint32_t, std::uint32_t>();
  return SyncPoint(id_val, wait_count_val);
}

ShardAccessMap SyncPoint::GetShardAccess() const { return {}; }

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

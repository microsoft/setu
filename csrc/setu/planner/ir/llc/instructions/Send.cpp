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
#include "planner/ir/llc/instructions/Send.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================

std::string Send::ToString() const {
  return std::format(
      "Send(peer_rank={}, shard={}, offset_bytes={}, count={}, dtype={}, "
      "src_device_ptr={})",
      peer_rank, src_shard.ToString(), offset_bytes, count,
      static_cast<int>(dtype), src_ptr);
}

void Send::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  const auto src_ptr_value = reinterpret_cast<std::uintptr_t>(src_ptr);
  writer.WriteFields(peer_rank, src_shard, offset_bytes, count, dtype,
                     src_ptr_value);
}

Send Send::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [peer_rank, src_shard, offset_bytes, count, dtype, src_ptr_val] =
      reader.ReadFields<DeviceRank, ShardRef, std::size_t, std::size_t,
                        torch::Dtype, std::uintptr_t>();
  auto src_ptr = reinterpret_cast<DevicePtr>(src_ptr_val);
  return Send(std::move(src_shard), offset_bytes, count, dtype, peer_rank,
              src_ptr);
}

void Send::Embellish(
    const std::function<DevicePtr(const ShardRef&)>& resolver) {
  src_ptr = resolver(src_shard);
}

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

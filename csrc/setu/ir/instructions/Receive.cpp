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
#include "setu/ir/instructions/Receive.h"
//==============================================================================
namespace setu::ir {
//==============================================================================

std::string Receive::ToString() const {
  return std::format(
      "Receive(peer_rank={}, shard={}, offset_bytes={}, count={}, dtype={}, "
      "dst_device_ptr={})",
      peer_rank, dst_shard.ToString(), offset_bytes, count,
      static_cast<int>(dtype), dst_ptr);
}

void Receive::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  const auto dst_ptr_value = reinterpret_cast<std::uintptr_t>(dst_ptr);
  writer.WriteFields(peer_rank, dst_shard, offset_bytes, count, dtype,
                     dst_ptr_value);
}

Receive Receive::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [peer_rank, dst_shard, offset_bytes, count, dtype, dst_ptr_value] =
      reader.ReadFields<DeviceRank, ShardRef, std::size_t, std::size_t,
                        torch::Dtype, std::uintptr_t>();
  const auto dst_ptr = reinterpret_cast<DevicePtr>(dst_ptr_value);
  return Receive(std::move(dst_shard), offset_bytes, count, dtype, peer_rank,
                 dst_ptr);
}

void Receive::Embellish(
    const std::function<DevicePtr(const ShardRef&)>& resolver) {
  dst_ptr = resolver(dst_shard);
}

//==============================================================================
}  // namespace setu::ir
//==============================================================================

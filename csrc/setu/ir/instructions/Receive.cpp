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
      peer_rank.has_value() ? std::to_string(peer_rank.value()) : "unset",
      dst_shard.ToString(), offset_bytes, count, static_cast<int>(dtype),
      dst_ptr);
}

void Receive::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  const auto dst_ptr_value = reinterpret_cast<std::uintptr_t>(dst_ptr);
  // Serialize peer_rank as optional (bool + value if present)
  writer.WriteFields(peer_rank.has_value(), peer_rank.value_or(0), dst_shard,
                     offset_bytes, count, dtype, dst_ptr_value);
}

Receive Receive::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [has_peer_rank, peer_rank_val, dst_shard, offset_bytes, count, dtype,
        dst_ptr_value] =
      reader.ReadFields<bool, DeviceRank, ShardRef, std::size_t, std::size_t,
                        torch::Dtype, std::uintptr_t>();
  const auto dst_ptr = reinterpret_cast<DevicePtr>(dst_ptr_value);
  Receive recv(std::move(dst_shard), offset_bytes, count, dtype, dst_ptr);
  if (has_peer_rank) {
    recv.SetPeerRank(peer_rank_val);
  }
  return recv;
}

void Receive::Embellish(
    const std::function<DevicePtr(const ShardRef&)>& resolver) {
  dst_ptr = resolver(dst_shard);
}

//==============================================================================
}  // namespace setu::ir
//==============================================================================

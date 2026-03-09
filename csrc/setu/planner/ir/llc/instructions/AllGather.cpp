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
#include "planner/ir/llc/instructions/AllGather.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================

std::string AllGather::ToString() const {
  return std::format(
      "AllGather(comm_id={}, send_ref={}, send_offset_bytes={}, "
      "recv_ref={}, recv_offset_bytes={}, send_count={}, dtype={}, "
      "num_ranks={}, send_ptr={}, recv_ptr={})",
      comm_id.ToString(), send_ref.ToString(), send_offset_bytes,
      recv_ref.ToString(), recv_offset_bytes, send_count,
      static_cast<int>(dtype), num_ranks, send_ptr, recv_ptr);
}

void AllGather::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  const auto send_ptr_value = reinterpret_cast<std::uintptr_t>(send_ptr);
  const auto recv_ptr_value = reinterpret_cast<std::uintptr_t>(recv_ptr);
  writer.WriteFields(comm_id, send_ref, send_offset_bytes, recv_ref,
                     recv_offset_bytes, send_count, dtype, num_ranks,
                     send_ptr_value, recv_ptr_value);
}

AllGather AllGather::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [comm_id, send_ref, send_offset_bytes, recv_ref, recv_offset_bytes,
        send_count, dtype, num_ranks, send_ptr_val, recv_ptr_val] =
      reader.ReadFields<CommId, BufferRef, std::size_t, BufferRef, std::size_t,
                        std::size_t, torch::Dtype, DeviceRank, std::uintptr_t,
                        std::uintptr_t>();
  auto send_ptr = reinterpret_cast<DevicePtr>(send_ptr_val);
  auto recv_ptr = reinterpret_cast<DevicePtr>(recv_ptr_val);
  return AllGather(comm_id, std::move(send_ref), send_offset_bytes,
                   std::move(recv_ref), recv_offset_bytes, send_count, dtype,
                   num_ranks, send_ptr, recv_ptr);
}

void AllGather::Embellish(
    const std::function<DevicePtr(const BufferRef&)>& resolver) {
  send_ptr = resolver(send_ref);
  recv_ptr = resolver(recv_ref);
}

ShardAccessMap AllGather::GetShardAccess() const {
  ShardAccessMap access_map;
  if (send_ref.IsShard()) {
    access_map.try_emplace(send_ref.AsShard().shard_id, ShardAccessMode::kRead);
  }
  if (recv_ref.IsShard()) {
    access_map.try_emplace(recv_ref.AsShard().shard_id,
                           ShardAccessMode::kWrite);
  }
  return access_map;
}

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

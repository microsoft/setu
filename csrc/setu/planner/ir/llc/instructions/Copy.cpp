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
#include "planner/ir/llc/instructions/Copy.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================

std::string Copy::ToString() const {
  std::string result = std::format("Copy(num_entries={})", entries.size());
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const auto& e = entries[i];
    result += std::format(
        "\n  [{}] src_ref={}, src_offset_bytes={}, dst_ref={}, "
        "dst_offset_bytes={}, count={}, dtype={}, src_ptr={}, dst_ptr={}",
        i, e.src_ref.ToString(), e.src_offset_bytes, e.dst_ref.ToString(),
        e.dst_offset_bytes, e.count, static_cast<int>(e.dtype), e.src_ptr,
        e.dst_ptr);
  }
  return result;
}

void Copy::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.Write<std::size_t>(entries.size());
  for (const auto& e : entries) {
    const auto src_ptr_value = reinterpret_cast<std::uintptr_t>(e.src_ptr);
    const auto dst_ptr_value = reinterpret_cast<std::uintptr_t>(e.dst_ptr);
    writer.WriteFields(e.src_ref, e.src_offset_bytes, e.dst_ref,
                       e.dst_offset_bytes, e.count, e.dtype, src_ptr_value,
                       dst_ptr_value);
  }
}

Copy Copy::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  const auto num_entries = reader.Read<std::size_t>();

  std::vector<CopyEntry> entries;
  entries.reserve(num_entries);
  for (std::size_t i = 0; i < num_entries; ++i) {
    auto [src_ref, src_offset_bytes, dst_ref, dst_offset_bytes, count, dtype,
          src_ptr_val, dst_ptr_val] =
        reader.ReadFields<BufferRef, std::size_t, BufferRef, std::size_t,
                          std::size_t, torch::Dtype, std::uintptr_t,
                          std::uintptr_t>();

    entries.emplace_back(std::move(src_ref), src_offset_bytes,
                         std::move(dst_ref), dst_offset_bytes, count, dtype,
                         reinterpret_cast<DevicePtr>(src_ptr_val),
                         reinterpret_cast<DevicePtr>(dst_ptr_val));
  }

  return Copy(std::move(entries));
}

void Copy::Embellish(
    const std::function<DevicePtr(const BufferRef&)>& resolver) {
  for (auto& e : entries) {
    e.src_ptr = resolver(e.src_ref);
    e.dst_ptr = resolver(e.dst_ref);
  }
}

ShardAccessMap Copy::GetShardAccess() const {
  ShardAccessMap access_map;
  for (const auto& e : entries) {
    // Write on dst always wins; read on src only if not already written
    if (e.dst_ref.IsShard()) {
      access_map[e.dst_ref.AsShard().shard_id] = ShardAccessMode::kWrite;
    }
    if (e.src_ref.IsShard()) {
      access_map.try_emplace(e.src_ref.AsShard().shard_id,
                             ShardAccessMode::kRead);
    }
  }
  return access_map;
}

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

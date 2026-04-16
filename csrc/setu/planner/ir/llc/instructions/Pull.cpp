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
#include "planner/ir/llc/instructions/Pull.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================

std::string Pull::ToString() const {
  std::string result = std::format("Pull(num_entries={})", entries.size());
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const auto& e = entries[i];
    result += std::format(
        "\n  [{}] src_ref={}, src_offset_bytes={}, dst_ref={}, "
        "dst_offset_bytes={}, count={}, dtype={}, src_device={}, src_ptr={}, "
        "dst_ptr={}",
        i, e.src_ref.ToString(), e.src_offset_bytes, e.dst_ref.ToString(),
        e.dst_offset_bytes, e.count, static_cast<int>(e.dtype), e.src_device,
        e.src_ptr, e.dst_ptr);
  }
  return result;
}

void Pull::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.Write<std::size_t>(entries.size());
  for (const auto& e : entries) {
    auto src_ptr_val = reinterpret_cast<std::uintptr_t>(e.src_ptr);
    auto dst_ptr_val = reinterpret_cast<std::uintptr_t>(e.dst_ptr);
    writer.WriteFields(e.src_ref, e.src_offset_bytes, e.dst_ref,
                       e.dst_offset_bytes, e.count, e.dtype, e.src_device,
                       src_ptr_val, dst_ptr_val);
  }
}

Pull Pull::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  const auto num_entries = reader.Read<std::size_t>();

  std::vector<PullEntry> entries;
  entries.reserve(num_entries);
  for (std::size_t i = 0; i < num_entries; ++i) {
    auto [src_ref, src_offset, dst_ref, dst_offset, count, dtype, src_dev,
          src_ptr_val, dst_ptr_val] =
        reader.ReadFields<BufferRef, std::size_t, BufferRef, std::size_t,
                          std::size_t, torch::Dtype, std::int32_t,
                          std::uintptr_t, std::uintptr_t>();
    entries.emplace_back(std::move(src_ref), src_offset, std::move(dst_ref),
                         dst_offset, count, dtype, src_dev,
                         reinterpret_cast<DevicePtr>(src_ptr_val),
                         reinterpret_cast<DevicePtr>(dst_ptr_val));
  }

  return Pull(std::move(entries));
}

void Pull::Embellish(
    const std::function<DevicePtr(const BufferRef&)>& resolver) {
  for (auto& entry : entries) {
    entry.src_ptr = resolver(entry.src_ref);
    entry.dst_ptr = resolver(entry.dst_ref);
  }
}

ShardAccessMap Pull::GetShardAccess() const {
  ShardAccessMap access_map;
  for (const auto& entry : entries) {
    if (entry.src_ref.IsShard()) {
      access_map.try_emplace(entry.src_ref.AsShard().shard_id,
                             ShardAccessMode::kRead);
    }
    if (entry.dst_ref.IsShard()) {
      access_map.try_emplace(entry.dst_ref.AsShard().shard_id,
                             ShardAccessMode::kWrite);
    }
  }
  return access_map;
}

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================

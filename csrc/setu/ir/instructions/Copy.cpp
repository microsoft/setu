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
#include "setu/ir/instructions/Copy.h"
//==============================================================================
namespace setu::ir {
//==============================================================================

std::string CopyInstruction::ToString() const {
  return std::format(
      "CopyInstruction(src_shard={}, src_offset={}, dst_shard={}, "
      "dst_offset={}, dtype={}, num_elements={}, src_ptr={}, dst_ptr={})",
      src_shard.ToString(), src_memory_offset_bytes, dst_shard.ToString(),
      dst_memory_offset_bytes, static_cast<int>(dtype), num_elements, src_ptr,
      dst_ptr);
}

void CopyInstruction::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  const auto src_ptr_value = reinterpret_cast<std::uintptr_t>(src_ptr);
  const auto dst_ptr_value = reinterpret_cast<std::uintptr_t>(dst_ptr);
  writer.WriteFields(src_shard, src_memory_offset_bytes, dst_shard,
                     dst_memory_offset_bytes, dtype, num_elements,
                     src_ptr_value, dst_ptr_value);
}

CopyInstruction CopyInstruction::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [src_shard, src_memory_offset_bytes, dst_shard, dst_memory_offset_bytes,
        dtype, num_elements, src_ptr_val, dst_ptr_val] =
      reader.ReadFields<ShardRef, std::size_t, ShardRef, std::size_t,
                        torch::Dtype, std::size_t, std::uintptr_t,
                        std::uintptr_t>();

  auto src_ptr = reinterpret_cast<DevicePtr>(src_ptr_val);
  auto dst_ptr = reinterpret_cast<DevicePtr>(dst_ptr_val);
  return CopyInstruction(std::move(src_shard), src_memory_offset_bytes,
                         std::move(dst_shard), dst_memory_offset_bytes, dtype,
                         num_elements, src_ptr, dst_ptr);
}

void CopyInstruction::Embellish(
    const std::function<DevicePtr(const ShardRef&)>& resolver) {
  src_ptr = resolver(src_shard);
  dst_ptr = resolver(dst_shard);
}

//==============================================================================
}  // namespace setu::ir
//==============================================================================

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
#include "setu/ir/instructions/Send.h"
//==============================================================================
namespace setu::ir {
//==============================================================================

std::string SendInstruction::ToString() const {
  return std::format(
      "SendInstruction(dst_rank={}, tensor=({}, {}), dtype={}, "
      "memory_offset={}, num_elements={}, src_device_ptr={})",
      dst_device_id, src_tensor.tensor_name, src_tensor.shard_id,
      static_cast<int>(dtype), memory_offset_bytes, num_elements, src_ptr);
}

void SendInstruction::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  const auto src_ptr_value = reinterpret_cast<std::uintptr_t>(src_ptr);
  writer.WriteFields(dst_device_id, src_tensor, dtype, memory_offset_bytes,
                     num_elements, src_ptr_value);
}

SendInstruction SendInstruction::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [dst_device_id, src_tensor, dtype, memory_offset_bytes, num_elements,
        src_ptr_val] =
      reader.ReadFields<DeviceRank, TensorShardIdentifier, torch::Dtype,
                        std::size_t, std::size_t, std::uintptr_t>();
  auto src_ptr = reinterpret_cast<DevicePtr>(src_ptr_val);
  return SendInstruction(dst_device_id, std::move(src_tensor), dtype,
                         memory_offset_bytes, num_elements, src_ptr);
}

void SendInstruction::Embellish(
    const std::function<DevicePtr(const TensorShardIdentifier&)>& resolver) {
  src_ptr = resolver(src_tensor);
}

//==============================================================================
}  // namespace setu::ir
//==============================================================================
